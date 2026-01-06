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
import os
from dotenv import load_dotenv # Додайте цей імпорт

# Явно завантажуємо файл .env
load_dotenv()

# ... решта ваших імпортів
# ... далі у коді:
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- НАЛАШТУВАННЯ ---
RISK_PER_TRADE = 100


def get_technical_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if len(df) < 50: return None, None

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['TR'] = pd.concat(
            [df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())],
            axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr14 = df['TR'].rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr14)
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(window=14).mean()

        last = df.iloc[-1]

        bias = "NEUTRAL"
        if last['ADX'] < 20:
            bias = "CHOPPY (NO TRADE)"
            side = "WAIT"
        elif last['Close'] > last['SMA_50']:
            bias = "BULLISH TREND"
            side = "LONG"
        else:
            bias = "BEARISH TREND"
            side = "SHORT"

        atr_multiplier = 2.0
        if side == "LONG":
            stop_loss = last['Close'] - (atr_multiplier * last['ATR'])
            target = last['Close'] + (3.0 * last['ATR'])
            resistance = last['Close'] + last['ATR']
        elif side == "SHORT":
            stop_loss = last['Close'] + (atr_multiplier * last['ATR'])
            target = last['Close'] - (3.0 * last['ATR'])
            resistance = last['Close'] - last['ATR']
        else:
            stop_loss = last['Close'] * 0.95
            target = last['Close'] * 1.05
            resistance = last['Close']

        risk_dist = abs(last['Close'] - stop_loss)
        reward = abs(target - last['Close'])
        ratio = reward / risk_dist if risk_dist > 0 else 0

        pos_size = 0
        if risk_dist > 0 and side != "WAIT":
            pos_size = (RISK_PER_TRADE / risk_dist) * last['Close']

        stats = {
            "Price": last['Close'], "RSI": last['RSI'], "ATR": last['ATR'], "ADX": last['ADX'],
            "SMA_20": last['SMA_20'], "Stop": stop_loss, "Target": target,
            "Resistance": resistance, "Ratio": ratio, "Side": side, "Bias": bias,
            "PosSize": pos_size
        }
        return df.tail(100), stats
    except:
        return None, None


def agent_pipeline(ticker, stats, llm):
    search = TavilySearchResults(max_results=3)
    try:
        news = search.invoke({"query": f"{ticker} crypto price news trading analysis"})
    except:
        news = "No news available."

    if stats['ADX'] < 20:
        return f"ACTION: WAIT\nCONFIDENCE: 100%\nPLAN: N/A\nREASON: Ринок у фазі флету (ADX {stats['ADX']:.1f} < 20).", \
        stats['Price']

    analyst_prompt = ChatPromptTemplate.from_template(
        "Аналітик. {ticker}. Trend: {bias}. ADX: {adx:.1f}. RSI: {rsi:.1f}. News: {news}. Оцінка ризиків українською."
    )
    analyst_chain = analyst_prompt | llm | StrOutputParser()
    report = analyst_chain.invoke(
        {"ticker": ticker, "bias": stats['Bias'], "adx": stats['ADX'], "rsi": stats['RSI'], "news": news})

    limit_level = stats['SMA_20']
    buffer = 1.0 * stats['ATR']
    if abs(limit_level - stats['Stop']) < buffer:
        safe_entry = stats['Stop'] + buffer if stats['Side'] == "LONG" else stats['Stop'] - buffer
    else:
        safe_entry = limit_level

    # ВИПРАВЛЕНО: .4f для точності XRP та українська мова
    ceo_prompt = ChatPromptTemplate.from_template(
        "Ти CEO хедж-фонду. Відповідай УКРАЇНСЬКОЮ.\n"
        "Data: Price ${price:.4f}, ADX {adx:.1f}, Bias {bias}.\n"
        "Sizing: Position size ${pos_size:.0f} (Risk ${risk}).\n"
        "Analyst Report: {report}\n\n"
        "LOGIC:\n1. ADX < 20 -> WAIT.\n2. LONG: Ratio > 1.5, RSI < 70.\n3. SHORT: Ratio > 2.5, RSI > 50, ADX > 25.\n"
        "OUTPUT FORMAT:\nACTION: [MARKET {side} / LIMIT {side} / WAIT]\nCONFIDENCE: [0-100]%\n"
        "PLAN: \n🔵 Вхід: ${safe_entry:.4f}\n🛡 Стоп: ${stop_loss:.4f}\n💰 Тейк: ${target:.4f}\n⚖️ Позиція: ${pos_size:.0f} (Ризик ${risk})\n🚀 Альтернатива: [Пробій ${resistance:.4f}]\n"
        "REASON: [Пояснення українською]"
    )
    ceo_chain = ceo_prompt | llm | StrOutputParser()
    raw = ceo_chain.invoke({
        "report": report, "ticker": ticker, "bias": stats['Bias'], "side": stats['Side'],
        "ratio": stats['Ratio'], "price": stats['Price'], "safe_entry": safe_entry,
        "stop_loss": stats['Stop'], "resistance": stats['Resistance'], "rsi": stats['RSI'],
        "target": stats['Target'], "adx": stats['ADX'], "pos_size": stats['PosSize'], "risk": RISK_PER_TRADE
    })
    return raw.replace("[", "").replace("]", ""), safe_entry


def send_telegram(ticker, data, img_bytes):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token: return

    try:
        plan = data['raw'].split("PLAN:")[1].split("REASON:")[0].strip()
    except:
        plan = "N/A"

    try:
        reason = data['raw'].split("REASON:")[1].strip()
    except:
        reason = "Analysis complete."

    icon = "🛡"
    if "MARKET" in data['action']:
        icon = "⚡"
    elif "LIMIT" in data['action']:
        icon = "🔵"

    caption = f"""
{icon} <b>AUTO-HUNTER: {ticker}</b>
<b>СИГНАЛ:</b> {data['action']} ({data['conf']}%)
<b>РЕЖИМ:</b> {data['stats']['Bias']} (ADX: {data['stats']['ADX']:.1f})

📋 <b>ПЛАН:</b>
{plan}

🧠 <b>АНАЛІЗ:</b>
{html.escape(reason)}
"""
    requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", files={'photo': img_bytes},
                  data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'})


def plot_chart(df, ticker, stats, entry):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                         increasing_line_color='#198754', decreasing_line_color='#dc3545')])
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='#fd7e14', width=1), name="SMA 20"))
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='#6610f2', width=1), name="SMA 50"))

    if stats['Side'] != "WAIT":
        fig.add_hline(y=stats['Target'], line_dash="dot", line_color='green')
        fig.add_hline(y=stats['Stop'], line_dash="dot", line_color='red')
        fig.add_hline(y=entry, line_dash="solid", line_color='blue')

    title_text = f"{ticker} | ADX: {stats['ADX']:.1f}"
    fig.update_layout(title=title_text, height=600, width=1000, template="plotly_white",
                      xaxis_rangeslider_visible=False)
    return fig


if __name__ == "__main__":
    # Повний список для GitHub
    TICKERS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD", "Cardano": "ADA-USD"}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    for name, symbol in TICKERS.items():
        print(f"Scanning {name}...")
        df, stats = get_technical_data(symbol)
        if stats:
            raw_verdict, safe_entry = agent_pipeline(name, stats, llm)

            action = "WAIT"
            if "MARKET" in raw_verdict:
                action = "MARKET"
            elif "LIMIT" in raw_verdict:
                action = "LIMIT"

            conf = "0"
            if "CONFIDENCE:" in raw_verdict: conf = raw_verdict.split("CONFIDENCE:")[1].split("%")[0].strip()

            entry = safe_entry if "LIMIT" in action else stats['Price']
            fig = plot_chart(df, name, stats, entry)
            img = fig.to_image(format="png", width=1000, height=600, scale=2)

            send_telegram(name,
                          {"action": action, "conf": conf, "entry_price": entry, "stats": stats, "raw": raw_verdict},
                          img)
