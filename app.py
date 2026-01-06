import streamlit as st
import requests
import os
import io
import html
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- 1. SETTINGS & LIGHT THEME CSS ---
load_dotenv()
st.set_page_config(page_title="NARRATIVE HUNTER | PRO TERMINAL", page_icon="🦅", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL --- */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 800 !important;
        color: #111 !important;
        text-transform: none; 
        letter-spacing: -0.5px;
    }

    /* --- METRIC CARDS --- */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-color: #ced4da;
    }
    div[data-testid="metric-container"] label {
        color: #6c757d !important;
        font-size: 14px;
        font-weight: 500;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #212529 !important;
        font-weight: 700;
    }

    /* --- VERDICT BOX --- */
    .verdict-box {
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-radius: 8px;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .v-green { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .v-blue { background-color: #cce5ff; border: 1px solid #b8daff; color: #004085; }
    .v-grey { background-color: #e2e3e5; border: 1px solid #d6d8db; color: #383d41; }

    /* --- PLAN/LOGS (UPDATED SPACING) --- */
    .terminal-box {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-left: 5px solid #0d6efd;
        color: #212529;
        font-family: 'Courier New', monospace;
        padding: 20px;
        border-radius: 4px;
        font-size: 14px;
        line-height: 2.0; /* Increased line height for readability */
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        color: #495057;
        font-weight: 700;
        border-radius: 8px;
        height: 50px;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
        color: #212529;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stButton > button:active {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. TELEGRAM ---
def send_telegram_photo_alert(ticker, decision_data, image_bytes):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return False, "❌ SYSTEM ERROR: Telegram config missing"

    action = decision_data['action']
    entry_price = decision_data['entry_price']
    stats = decision_data['stats']

    fmt_entry = f"<b>${entry_price:,.2f}</b>"
    fmt_stop = f"<b>${stats['Stop']:,.2f}</b>"
    fmt_target = f"<b>${stats['Target']:,.2f}</b>"
    fmt_ratio = f"{stats['Ratio']:.2f}"

    plan_text = decision_data['plan']
    alt_text = "Немає даних"
    for line in plan_text.split('\n'):
        # Шукаємо рядок з альтернативою
        if "🚀" in line or "Breakout" in line:
            clean_line = line.replace("🚀", "").replace("Альтернатива (Breakout):", "").strip()
            alt_text = clean_line
            break

    reason_safe = html.escape(decision_data['reason'])

    if "WAIT" in action:
        header = f"🛡 <b>MARKET HUNTER: {ticker}</b>"
        status_line = "✋ <b>ОЧІКУВАННЯ (WAIT)</b>"
    elif "LIMIT" in action:
        header = f"🔵 <b>MARKET HUNTER: {ticker}</b>"
        status_line = "🔵 <b>ЛІМІТНИЙ ВХІД (LIMIT LONG)</b>"
    else:
        header = f"🟢 <b>MARKET HUNTER: {ticker}</b>"
        status_line = "⚡ <b>РИНКОВИЙ ВХІД (MARKET LONG)</b>"

    caption = f"""
{header}

<b>НАСТРІЙ:</b> {stats['Bias']}
<b>СИГНАЛ:</b> {status_line}
<b>ВПЕВНЕНІСТЬ:</b> {decision_data['confidence']}%

📋 <b>ТОРГОВИЙ ПЛАН:</b>
🔵 <b>Вхід:</b> {fmt_entry}
🛡 <b>Стоп-лос:</b> {fmt_stop}
💰 <b>Тейк-профіт:</b> {fmt_target}
⚖️ <b>R/R Ratio:</b> {fmt_ratio}

🚀 <b>АЛЬТЕРНАТИВА (BREAKOUT):</b>
{alt_text}

🧠 <b>АНАЛІЗ ШІ:</b>
{reason_safe}

<i>Disclaimer: AI Generated. DYOR.</i>
"""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('chart.png', image_bytes, 'image/png')}
    data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            return True, "✅ REPORT SENT"
        else:
            return False, f"FAILED: {response.text}"
    except Exception as e:
        return False, f"ERROR: {e}"


# --- 3. TECHNICAL CORE ---
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
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / (
            -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))))

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

        stats = {
            "Price": last['Close'], "RSI": last['RSI'], "ATR": last['ATR'],
            "SMA_20": last['SMA_20'], "Stop": stop_loss, "Target": target,
            "Resistance": resistance, "Ratio": ratio, "Side": side, "Bias": bias,
            "ChangePct": ((last['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100
        }
        return df.tail(100), stats
    except:
        return None, None


def plot_pro_chart(df, ticker, stats, action_type, display_entry):
    chart_data = df.reset_index()
    increase_color = '#198754'
    decrease_color = '#dc3545'

    fig = go.Figure(data=[
        go.Candlestick(x=chart_data.iloc[:, 0], open=chart_data['Open'], high=chart_data['High'], low=chart_data['Low'],
                       close=chart_data['Close'], increasing_line_color=increase_color,
                       decreasing_line_color=decrease_color,
                       name="Asset")])
    fig.add_trace(go.Scatter(x=chart_data.iloc[:, 0], y=chart_data['SMA_20'], mode='lines', name='SMA 20',
                             line=dict(color='#fd7e14', width=1.5)))
    fig.add_trace(go.Scatter(x=chart_data.iloc[:, 0], y=chart_data['SMA_50'], mode='lines', name='SMA 50',
                             line=dict(color='#6610f2', width=1.5)))

    entry_label = "LIMIT ENTRY" if "LIMIT" in action_type else "MARKET ENTRY"
    fig.add_hline(y=stats['Target'], line_dash="dot", line_color=increase_color, annotation_text="TARGET 🎯")
    fig.add_hline(y=stats['Stop'], line_dash="dot", line_color=decrease_color, annotation_text="INVALID ❌")
    fig.add_hline(y=display_entry, line_dash="solid", line_color="#0d6efd", line_width=1, annotation_text=entry_label,
                  annotation_position="left")

    fig.update_layout(title=f"<b>{ticker}</b> // {stats['Bias']}", xaxis_rangeslider_visible=False, height=600,
                      margin=dict(l=10, r=60, t=50, b=20), plot_bgcolor='white', paper_bgcolor='white',
                      font=dict(color="#212529"))
    return fig


# --- 4. AGENTS ---
def agent_pipeline(ticker, stats, llm):
    search = TavilySearchResults(max_results=5)
    news = search.invoke({"query": f"{ticker} crypto price news trading analysis"})

    analyst_prompt = ChatPromptTemplate.from_template(
        "Ти аналітик. {ticker}. Bias: {bias}. RSI {rsi:.1f}. Новини: {news}. Score (0-100) та ризики українською.")
    analyst_chain = analyst_prompt | llm | StrOutputParser()
    report = analyst_chain.invoke({"ticker": ticker, "bias": stats['Bias'], "rsi": stats['RSI'], "news": news})

    limit_level = stats['SMA_20']
    buffer_distance = 1.0 * stats['ATR']

    if abs(limit_level - stats['Stop']) < buffer_distance:
        if stats['Side'] == "LONG":
            safe_entry = stats['Stop'] + buffer_distance
        else:
            safe_entry = stats['Stop'] - buffer_distance
    else:
        safe_entry = limit_level

    # --- CEO PROMPT (ВИПРАВЛЕНО ОПИС АЛЬТЕРНАТИВИ) ---
    ceo_prompt = ChatPromptTemplate.from_template(
        "Ти — CEO Хедж-фонду. Фінальне рішення.\n"
        "Вхід: Bias {bias}, Ratio {ratio:.2f}, RSI {rsi:.1f}, Price ${price:.0f}, Safe Entry ${safe_entry:.0f}, Target ${target:.0f}.\n"
        "Аналітика: {report}\n\n"
        "ПРАВИЛА:\n"
        "1. WAIT: Ratio < 1.5 АБО Score < 60.\n"
        "2. LIMIT {side}: Ratio > 1.5, але Score 60-80 (або RSI > 70).\n"
        "3. MARKET {side}: Ratio > 2.0 І Score > 80 (RSI < 70).\n\n"
        "ЗАДАЧА: Сформуй план українською.\n"
        "ВАЖЛИВО: Альтернатива (Breakout) - це вхід на пробої опору, якщо ціна НЕ піде до нашого ліміту.\n\n"
        "ФОРМАТ ВІДПОВІДІ:\n"
        "ACTION: [MARKET {side} / LIMIT {side} / WAIT]\nCONFIDENCE: [0-100]%\n"
        "PLAN: \n"
        "🔵 Вхід (Entry): ${safe_entry:.0f} (Лімітний ордер)\n"
        "🛡 Стоп-лосс (Stop): ${stop_loss:.0f}\n"
        "💰 Тейк-профіт (Target): ${target:.0f}\n"
        "🚀 Альтернатива (Breakout): Вхід тільки при пробої ${resistance:.0f} вгору (якщо не було відкату)\n"
        "REASON: [Синтез новин та стратегії без зайвих цифр]"
    )
    ceo_chain = ceo_prompt | llm | StrOutputParser()
    raw = ceo_chain.invoke({
        "report": report, "bias": stats['Bias'], "side": stats['Side'], "ratio": stats['Ratio'],
        "price": stats['Price'], "safe_entry": safe_entry, "stop_loss": stats['Stop'],
        "resistance": stats['Resistance'], "rsi": stats['RSI'],
        "target": stats['Target']
    })
    return raw.replace("[", "").replace("]", ""), safe_entry


# --- 5. INTERFACE ---
def main():
    with st.sidebar:
        st.markdown("## 🦅 CONTROL PANEL")
        tickers = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD"}
        option = st.selectbox("ASSET", list(tickers.keys()))
        if st.button("⚡ START ANALYSIS", type="primary"): st.session_state.scan_requested = True

        st.markdown("---")
        st.markdown("### SYSTEM STATUS")
        api_status = "✅ ONLINE" if os.getenv("OPENAI_API_KEY") else "❌ OFFLINE"
        tavily_status = "✅ ONLINE" if os.getenv("TAVILY_API_KEY") else "❌ OFFLINE"
        st.info(f"AI CORE: {api_status}")
        st.info(f"DATA FEED: {tavily_status}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if "decision" not in st.session_state: st.session_state.decision = None

    if st.session_state.get("scan_requested", False):
        st.session_state.scan_requested = False
        with st.spinner("Analyzing market structure..."):
            df, stats = get_technical_data(tickers[option])
            if stats:
                full_verdict, safe_entry = agent_pipeline(tickers[option], stats, llm)

                lines = full_verdict.split("\n")
                action, confidence, plan, reason = "WAIT", "0", "N/A", "N/A"
                for line in lines:
                    if "ACTION:" in line:
                        action = line.split("ACTION:")[1].strip()
                    elif "CONFIDENCE:" in line:
                        confidence = line.split("CONFIDENCE:")[1].replace("%", "").strip()
                    elif "REASON:" in line:
                        reason = line.split("REASON:")[1].strip()
                try:
                    plan = full_verdict.split("PLAN:")[1].split("REASON:")[0].strip()
                except:
                    plan = "N/A"

                if "LIMIT" in action:
                    display_entry = safe_entry
                else:
                    display_entry = stats['Price']

                fig = plot_pro_chart(df, option, stats, action, display_entry)

                st.session_state.decision = {
                    "stats": stats, "action": action, "confidence": confidence,
                    "plan": plan, "reason": reason, "bias": stats['Bias'],
                    "ticker": option, "fig": fig,
                    "entry_price": display_entry
                }
            else:
                st.error("❌ Data connection failed.")

    if st.session_state.decision:
        data = st.session_state.decision
        stats = data['stats']

        cls = "v-green" if "MARKET" in data['action'] else "v-blue" if "LIMIT" in data['action'] else "v-grey"
        st.markdown(
            f'<div class="verdict-box {cls}">{data["action"]} <span style="font-size:0.6em">({data["confidence"]}%)</span></div>',
            unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry Price", f"${data['entry_price']:,.2f}")
        m2.metric("Target", f"${stats['Target']:,.2f}")
        m3.metric("Invalidation", f"${stats['Stop']:,.2f}")
        m4.metric("Risk Ratio", f"{stats['Ratio']:.2f}", f"RSI: {stats['RSI']:.1f}")

        # TABS CORRECTION
        tab1, tab2 = st.tabs(["📊 CHART ANALYTICS", "📋 EXECUTION PLAN"])

        with tab1:
            st.plotly_chart(data['fig'], use_container_width=True)

        with tab2:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("### 📝 Strategy Protocol")
                # REPLACING NEWLINES WITH <br> FOR CLEARER FORMATTING
                st.markdown(f'<div class="terminal-box">{data["plan"].replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown("### 🧠 AI Rationale")
                st.info(data['reason'])
                st.markdown(f"**BIAS:** `{data['bias']}`")

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("📤 SEND REPORT"):
                with st.spinner("Generating Report..."):
                    img_bytes = data['fig'].to_image(format="png", width=1200, height=800, scale=2)
                    res, msg = send_telegram_photo_alert(data['ticker'], data, img_bytes)
                    if res:
                        st.success(msg)
                    else:
                        st.error(msg)


if __name__ == "__main__":
    main()
