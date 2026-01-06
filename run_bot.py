import os
import html
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ФУНКЦІЇ ---
def get_technical_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if len(df) < 50: return None, None

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['TR'] = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))))
        
        last = df.iloc[-1]
        bias = "NEUTRAL"
        side = "LONG"

        if last['Close'] > last['SMA_50']:
            bias = "BULLISH TREND" if last['Close'] > last['SMA_20'] else "WEAK BULL (Retrace)"
            side = "LONG"
        else:
            bias = "BEARISH TREND" if last['Close'] < last['SMA_20'] else "WEAK BEAR (Bounce)"
            side = "SHORT"

        atr_multiplier = 2.0 
        if side == "LONG":
            stop_loss = last['Close'] - (atr_multiplier * last['ATR'])
            target = last['Close'] + (3.0 * last['ATR'])
            resistance = last['Close'] + last['ATR']
            risk = last['Close'] - stop_loss
            reward = target - last['Close']
        else:
            stop_loss = last['Close'] + (atr_multiplier * last['ATR'])
            target = last['Close'] - (3.0 * last['ATR'])
            resistance = last['Close'] - last['ATR']
            risk = stop_loss - last['Close']
            reward = last['Close'] - target
        
        ratio = reward / risk if risk > 0 else 0
        stats = {"Price": last['Close'], "RSI": last['RSI'], "ATR": last['ATR'], "SMA_20": last['SMA_20'], "Stop": stop_loss, "Target": target, "Resistance": resistance, "Ratio": ratio, "Side": side, "Bias": bias}
        return df.tail(100), stats
    except: return None, None

def agent_pipeline(ticker, stats, llm):
    search = TavilySearchResults(max_results=5)
    news = search.invoke({"query": f"{ticker} crypto price news trading analysis"})
    analyst_prompt = ChatPromptTemplate.from_template("Ти аналітик. {ticker}. Bias: {bias}. RSI {rsi:.1f}. Новини: {news}. Score (0-100) та ризики українською.")
    analyst_chain = analyst_prompt | llm | StrOutputParser()
    report = analyst_chain.invoke({"ticker": ticker, "bias": stats['Bias'], "rsi": stats['RSI'], "news": news})
    
    limit_level = stats['SMA_20']
    buffer = 1.0 * stats['ATR']
    if abs(limit_level - stats['Stop']) < buffer:
        safe_entry = stats['Stop'] + buffer if stats['Side'] == "LONG" else stats['Stop'] - buffer
    else: safe_entry = limit_level

    ceo_prompt = ChatPromptTemplate.from_template(
        "Ти CEO. Вхід: Bias {bias}, Ratio {ratio:.2f}, RSI {rsi:.1f}, Price ${price:.0f}, Safe Entry ${safe_entry:.0f}, Target ${target:.0f}.\n"
        "Аналітика: {report}\nWAIT: Ratio < 1.5. LIMIT: Ratio > 1.5. MARKET: Ratio > 2.0 AND Score > 80.\n"
        "ФОРМАТ:\nACTION: [MARKET {side} / LIMIT {side} / WAIT]\nCONFIDENCE: [0-100]%\n"
        "PLAN: \n🔵 Вхід (Entry): ${safe_entry:.0f}\n🛡 Стоп: ${stop_loss:.0f}\n💰 Тейк: ${target:.0f}\n🚀 Альтернатива: [Умова пробою ${resistance:.0f}]\nREASON: [Суть]"
    )
    ceo_chain = ceo_prompt | llm | StrOutputParser()
    raw = ceo_chain.invoke({"report": report, "bias": stats['Bias'], "side": stats['Side'], "ratio": stats['Ratio'], "price": stats['Price'], "safe_entry": safe_entry, "stop_loss": stats['Stop'], "resistance": stats['Resistance'], "rsi": stats['RSI'], "target": stats['Target']})
    return raw.replace("[", "").replace("]", ""), safe_entry

def send_telegram(ticker, decision_data, image_bytes):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token: return
    
    stats = decision_data['stats']
    plan_text = decision_data['plan']
    alt_text = "N/A"
    for line in plan_text.split('\n'):
        if "🚀" in line or "Альтернатива" in line: alt_text = line.replace("🚀", "").replace("Альтернатива:", "").strip()

    caption = f"""
<b>AUTO-HUNTER: {ticker}</b>
<b>СИГНАЛ:</b> {decision_data['action']} ({decision_data['confidence']}%)

🔵 <b>Вхід:</b> <b>${decision_data['entry_price']:,.0f}</b>
🛡 <b>Стоп:</b> ${stats['Stop']:,.0f}
💰 <b>Тейк:</b> ${stats['Target']:,.0f}

🚀 <b>АЛЬТЕРНАТИВА:</b> {alt_text}

🧠 <b>АНАЛІЗ:</b> {html.escape(decision_data['reason'])}
"""
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", files=files, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'})

def plot_chart(df, ticker, stats, entry):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#198754', decreasing_line_color='#dc3545')])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='#fd7e14', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='#6610f2', width=1)))
    fig.add_hline(y=stats['Target'], line_dash="dot", line_color='green')
    fig.add_hline(y=stats['Stop'], line_dash="dot", line_color='red')
    fig.add_hline(y=entry, line_dash="solid", line_color='blue')
    fig.update_layout(title=f"{ticker}", height=600, width=1000, template="plotly_white", xaxis_rangeslider_visible=False)
    return fig

# --- ЗАПУСК ---
if __name__ == "__main__":
    TICKERS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    for name, symbol in TICKERS.items():
        print(f"Scanning {name}...")
        df, stats = get_technical_data(symbol)
        if stats:
            verdict, safe_entry = agent_pipeline(symbol, stats, llm)
            lines = verdict.split("\n")
            action, confidence, plan, reason = "WAIT", "0", "N/A", "N/A"
            for line in lines:
                if "ACTION:" in line: action = line.split("ACTION:")[1].strip()
                elif "CONFIDENCE:" in line: confidence = line.split("CONFIDENCE:")[1].replace("%","").strip()
                elif "REASON:" in line: reason = line.split("REASON:")[1].strip()
            try: plan = verdict.split("PLAN:")[1].split("REASON:")[0].strip()
            except: plan = "N/A"
            
            entry = safe_entry if "LIMIT" in action else stats['Price']
            fig = plot_chart(df, name, stats, entry)
            img = fig.to_image(format="png", width=1000, height=600, scale=2)
            
            send_telegram(name, {"action": action, "confidence": confidence, "entry_price": entry, "stats": stats, "plan": plan, "reason": reason}, img)
