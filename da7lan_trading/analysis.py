# =======================================================
# analysis.py — Gold AI Trader (Full ~450 lines)
# تحليل فني + منطقي + نماذج + موجات + جلسات + خلاصة ذكية
# بيانات حقيقية عبر TwelveData API
# =======================================================
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Tuple, List, Optional

# ---------------- Env & Const ----------------
load_dotenv()
API_KEY = os.getenv("TWELVEDATA_KEY")
SYMBOL = os.getenv("SYMBOL", "XAU/USD")
BASE_URL = "https://api.twelvedata.com"

API_ERROR_NOTICE = "⚠️ تعذّر جلب البيانات. تأكد من TWELVEDATA_KEY والحد اليومي (quota)."

# =======================================================
#                    Data Fetch Layer
# =======================================================
def _assert_key():
    if not API_KEY:
        raise RuntimeError("❌ TWELVEDATA_KEY مفقود في .env")

def fetch_realtime_price() -> Optional[float]:
    """Price endpoint — returns last price rounded to 2 decimals."""
    try:
        _assert_key()
        r = requests.get(f"{BASE_URL}/price", params={"symbol": SYMBOL, "apikey": API_KEY}, timeout=10)
        j = r.json()
        if "price" in j:
            return round(float(j["price"]), 2)
    except:
        pass
    return None

def fetch_candles(interval: str, outputsize: int = 300) -> Optional[pd.DataFrame]:
    """
    time_series endpoint — returns DataFrame with datetime(Open/High/Low/Close/Volume)
    interval: '1min','5min','15min','1h','4h','1day'
    """
    try:
        _assert_key()
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "timezone": "UTC",
            "order": "ASC",
        }
        r = requests.get(f"{BASE_URL}/time_series", params=params, timeout=15)
        j = r.json()
        if "values" not in j:
            return None
        df = pd.DataFrame(j["values"])
        for c in ["datetime", "open", "high", "low", "close"]:
            if c not in df.columns:
                return None
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["Open"]  = df["open"].astype(float)
        df["High"]  = df["high"].astype(float)
        df["Low"]   = df["low"].astype(float)
        df["Close"] = df["close"].astype(float)
        if "volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        else:
            df["Volume"] = 0.0
        return df[["datetime","Open","High","Low","Close","Volume"]].sort_values("datetime").reset_index(drop=True)
    except:
        return None

def now_utc_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# =======================================================
#                   Indicators & Math
# =======================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n//2)).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_dn = dn.rolling(n, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def macd(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    e12 = ema(s, 12); e26 = ema(s, 26)
    line = e12 - e26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def stochastic(df: pd.DataFrame, k_period=14, d_period=3) -> Tuple[pd.Series, pd.Series]:
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d

def bollinger_bands(s: pd.Series, n=20, k=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = sma(s, n)
    std = s.rolling(n, min_periods=n).std()
    upper = ma + k*std
    lower = ma - k*std
    return lower, ma, upper

# =======================================================
#               Trend / Structure / Levels
# =======================================================
def detect_trend_label(df: pd.DataFrame) -> Tuple[str, float, float, float, float]:
    close = df["Close"]
    e20, e50 = ema(close, 20), ema(close, 50)
    rsi14 = rsi(close, 14)
    n = min(30, len(close)-1)
    pct = ((close.iloc[-1] - close.iloc[-n]) / (close.iloc[-n])) * 100 if n > 1 else 0.0
    if e20.iloc[-1] > e50.iloc[-1] and rsi14.iloc[-1] > 55 and pct > 0.2:
        lbl = "صاعد قوي"
    elif e20.iloc[-1] < e50.iloc[-1] and rsi14.iloc[-1] < 45 and pct < -0.2:
        lbl = "هابط قوي"
    elif abs(pct) < 0.1:
        lbl = "جانبي متذبذب"
    else:
        lbl = "اتجاه ضعيف/متغير"
    return lbl, float(e20.iloc[-1]), float(e50.iloc[-1]), float(rsi14.iloc[-1]), float(pct)

def support_resistance_swing(df: pd.DataFrame, lookback=120) -> Tuple[float, float]:
    r = df.tail(min(lookback, len(df)))
    return round(float(r["Low"].min()), 2), round(float(r["High"].max()), 2)

def pivots_from_daily(d1: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    row = d1.iloc[-1]
    H,L,C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P = (H+L+C)/3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H-L)
    S2 = P - (H-L)
    return round(P,2), round(S1,2), round(S2,2), round(R1,2), round(R2,2)

def breakout_level(df: pd.DataFrame, window=48) -> float:
    return round(float(df["High"].tail(window).max()), 2)

def structure_note(h4: pd.DataFrame) -> str:
    highs = h4["High"].tail(6).values
    lows  = h4["Low"].tail(6).values
    if len(highs) < 2: return "هيكل محدود البيانات"
    hh = highs[-1] > max(highs[:-1])
    hl = lows[-1]  > min(lows[:-1])
    if hh and hl: return "HH/HL محافظ عليه"
    if hh: return "Higher High مُسجّل"
    if hl: return "Higher Low مُسجّل"
    return "هيكل محايد"

def fib_levels_from_swings(df: pd.DataFrame, lookback=120) -> Tuple[float,float,float,float,float]:
    """إرجاع مستويات فيبوناتشي (0.382/0.5/0.618) بين آخر قاع/قمة رئيسية داخل lookback."""
    recent = df.tail(min(lookback, len(df)))
    low = float(recent["Low"].min())
    high = float(recent["High"].max())
    if high == low:
        return low, high, low, low, low
    # لو الاتجاه صاعد: قياسات على الهبوط؛ لو هابط بالعكس
    if recent["Close"].iloc[-1] > (low + high)/2:
        # اتجاه صاعد
        r382 = round(high - 0.382*(high-low), 2)
        r500 = round(high - 0.5*(high-low), 2)
        r618 = round(high - 0.618*(high-low), 2)
    else:
        # اتجاه هابط
        r382 = round(low + 0.382*(high-low), 2)
        r500 = round(low + 0.5*(high-low), 2)
        r618 = round(low + 0.618*(high-low), 2)
    return round(low,2), round(high,2), r382, r500, r618

def simple_elliott_hint(df: pd.DataFrame) -> str:
    """
    تلميح مبسط جدًا لموجات إليوت: يعتمد على تسلسل قمم/قيعان.
    هذا ليس كاشف كامل، بس يعطي إشارة “موجة دافعة/تصحيح”.
    """
    close = df["Close"].values[-40:]
    if len(close) < 10:
        return "معلومات موجية غير كافية"
    # احسب قمم/قيعان بسيطة
    swings = []
    for i in range(2, len(close)-2):
        if close[i] > close[i-1] and close[i] > close[i+1]:
            swings.append(("H", i, close[i]))
        if close[i] < close[i-1] and close[i] < close[i+1]:
            swings.append(("L", i, close[i]))
    if len(swings) < 4:
        return "بنية موجية محدودة"
    # عد نمط HL/HH متتالي
    highs = [v for t,_,v in swings if t=="H"]
    lows  = [v for t,_,v in swings if t=="L"]
    if len(highs)>=2 and len(lows)>=2 and highs[-1]>highs[-2] and lows[-1]>lows[-2]:
        return "🌀 موجة دافعة (ميل صاعد)"
    if len(highs)>=2 and len(lows)>=2 and highs[-1]<highs[-2] and lows[-1]<lows[-2]:
        return "🌀 موجة دافعة (ميل هابط)"
    return "🔄 موجة تصحيح/جانبية"

# =======================================================
#                    Candle & Patterns
# =======================================================
def detect_candle_patterns(df: pd.DataFrame) -> List[str]:
    out=[]
    if len(df)<3: return out
    o = df["Open"].values; h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    # آخر 3
    o1,o2,o3 = o[-3],o[-2],o[-1]
    c1,c2,c3 = c[-3],c[-2],c[-1]
    h3,l3 = h[-1],l[-1]
    # Engulfing
    if c2<o2 and c3>o3 and (c3-o3)>(o2-c2): out.append("✨ Bullish Engulfing (ابتلاع شرائي)")
    if c2>o2 and c3<o3 and (o3-c3)>(c2-o2): out.append("⚠️ Bearish Engulfing (ابتلاع بيعي)")
    # Pin Bar
    body = abs(c3 - o3); rng = h3 - l3
    if rng>0 and body/rng < 0.3:
        upper = h3 - max(c3,o3); lower = min(c3,o3) - l3
        if upper > 2*body: out.append("🔧 Shooting Star / Pin Bar علوي")
        if lower > 2*body: out.append("🔧 Hammer / Pin Bar سفلي")
    # Morning/Evening Star (مبسطة)
    if c1>o1 and abs(c2-o2)/max(1e-9,(h[-2]-l[-2]))<0.25 and c3>o3 and c3>c1:
        out.append("🌅 Morning Star (إيجابي)")
    if c1<o1 and abs(c2-o2)/max(1e-9,(h[-2]-l[-2]))<0.25 and c3<o3 and c3<c1:
        out.append("🌆 Evening Star (سلبي)")
    return out

def detect_chart_pattern(df: pd.DataFrame) -> str:
    # تقلص المدى -> مثلث/علم
    recent = df.tail(30)
    rng_ma = (recent["High"] - recent["Low"]).rolling(5).mean()
    if len(rng_ma.dropna())==0: return "لا يوجد نمط واضح"
    if rng_ma.iloc[-1] < 0.65 * rng_ma.mean():
        return "🔺 Consolidation (مثلث/علم) — راقب الاختراق"
    return "لا يوجد نمط قوي حاليًا"

# =======================================================
#                 Session & Sentiment
# =======================================================
def session_tag() -> str:
    h = datetime.utcnow().hour
    if 0 <= h < 7: return "آسيا"
    if 7 <= h < 12: return "أوروبا المبكرة"
    if 12 <= h < 20: return "أمريكا"
    return "أوروبا المتأخرة"

def sentiment_logic(df: pd.DataFrame) -> str:
    lbl,e20,e50,rsi14,_ = detect_trend_label(df)
    macd_line,signal,hist = macd(df["Close"])
    s = 0
    if lbl.startswith("صاعد"): s+=2
    if rsi14>55: s+=1
    if macd_line.iloc[-1] > signal.iloc[-1]: s+=1
    if abs(hist.tail(5).mean()) < abs(hist).quantile(0.3): s-=1
    if s>=3: return "إيجابي قوي"
    if s==2: return "إيجابي"
    if s==1: return "محايد مائل للصعود"
    if s==0: return "محايد"
    if s==-1: return "محايد مائل للهبوط"
    return "سلبي"

# =======================================================
#                 Probability Matrix
# =======================================================
def probability_matrix_from_atr(price: float, atr_val: float) -> Tuple[float,float,float,float,float,float]:
    """
    مصفوفة احتمالات بسيطة بناءً على مضاعفات ATR من السعر الحالي.
    (ليس توقعًا يقينيًا؛ فقط قياس احتمالي مبسط)
    """
    if atr_val <= 0: return 50,60,65,50,40,30
    p24_up   = min(90, 55 + (atr_val*0.0 + 10))   # baseline
    p7d_up   = min(92, p24_up + 5)
    p1m_up   = min(95, p7d_up + 5)
    p24_down = max(10, 100 - p24_up)
    p7d_down = max(8, 100 - p7d_up)
    p1m_down = max(5, 100 - p1m_up)
    return p24_up, p7d_up, p1m_up, p24_down, p7d_down, p1m_down

# =======================================================
#                    Reports: Price Brief
# =======================================================
def generate_price_brief() -> str:
    m15 = fetch_candles("15min", 60)
    if m15 is None or m15.empty:
        return API_ERROR_NOTICE
    last = m15.iloc[-1]
    price = float(last["Close"])
    now = last["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"🟡 **XAU/USD – السعر اللحظي**\n"
        f"💵 ${price:.2f}\n"
        f"🕒 {now}\n"
        f"المصدر: TwelveData"
    )

# =======================================================
#                  Reports: Sniper (M15)
# =======================================================
def generate_sniper_report() -> str:
    m15 = fetch_candles("15min", 300)
    if m15 is None or m15.empty:
        return API_ERROR_NOTICE
    trend, e20, e50, rsi15, pct = detect_trend_label(m15)
    m15["ATR14"] = atr(m15, 14)
    a = float(m15["ATR14"].iloc[-1])
    price = float(m15["Close"].iloc[-1])

    if e20>e50 and rsi15>55: signal, side = "شراء 🔼", 1
    elif e20<e50 and rsi15<45: signal, side = "بيع 🔽", -1
    else: signal, side = "انتظار ⚪", 0

    if side!=0 and a>0:
        tp1 = round(price + side*0.6*a, 2)
        tp2 = round(price + side*1.0*a, 2)
        tp3 = round(price + side*1.4*a, 2)
        sl  = round(price - side*0.8*a, 2)
    else:
        tp1=tp2=tp3=sl=None

    conf = max(0.0, min(100.0, abs(rsi15-50)*2))
    lines = [
        "🎯 **القنّاص – M15**",
        f"💵 السعر: ${price:.2f}",
        f"📈 الاتجاه (M15): {trend} | Δ{pct:.2f}%",
        f"EMA20/50: {e20:.2f}/{e50:.2f}",
        f"RSI(14): {rsi15:.1f}",
        f"ATR(14): {a:.2f}",
        f"🧠 الإشارة: {signal} | الثقة: {conf:.1f}%"
    ]
    if tp1:
        lines += ["🎯 الأهداف:", f"• TP1: {tp1}", f"• TP2: {tp2}", f"• TP3: {tp3}", f"🛡️ SL: {sl}"]
    else:
        lines += ["⚠️ السوق غير واضح الآن — يُفضل الانتظار."]
    lines += [f"🕒 {m15.iloc[-1]['datetime'].strftime('%Y-%m-%d %H:%M UTC')}", "المصدر: TwelveData"]
    return "\n".join(lines)

# =======================================================
#          Reports: Secret Scalping (1m) — Enhanced
# =======================================================
def generate_secret_scalp_report() -> str:
    m1 = fetch_candles("1min", 300)
    if m1 is None or m1.empty:
        return API_ERROR_NOTICE

    # مؤشرات
    m1["RSI5"] = rsi(m1["Close"], 5)
    k, d = stochastic(m1, k_period=5, d_period=3)
    m1["K"], m1["D"] = k, d
    m1["ATR14"] = atr(m1, 14)

    price = float(m1["Close"].iloc[-1])
    now = m1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    rsi5 = float(m1["RSI5"].iloc[-1])
    kL, dL = float(m1["K"].iloc[-1]), float(m1["D"].iloc[-1])
    atr1 = float(m1["ATR14"].iloc[-1]) if not np.isnan(m1["ATR14"].iloc[-1]) else 0.0

    vol_recent = m1["Volume"].tail(30)
    if vol_recent.sum() > 0:
        vol_rel = (m1["Volume"].iloc[-1] / (vol_recent.mean()+1e-9)) * 100
    else:
        vol_rel = 100  # fallback

    last_open = m1["Open"].iloc[-1]; last_high = m1["High"].iloc[-1]
    last_low  = m1["Low"].iloc[-1];  last_close = m1["Close"].iloc[-1]
    body = last_close - last_open; rng = last_high - last_low
    bullish = (body > 0) and (body >= 0.5*rng)
    bearish = (body < 0) and (abs(body) >= 0.5*rng)

    buy_cond = (rsi5 > 60 and kL > dL and bullish and vol_rel > 110)
    sell_cond= (rsi5 < 40 and kL < dL and bearish and vol_rel > 110)

    if atr1 <= 0:
        atr1 = max(0.5, abs(m1["Close"].pct_change().tail(30).mean()) * price)

    entry = price; tp1=tp2=sl=None; signal="انتظار ⚪"; confidence=0.0; note=[]
    if buy_cond:
        signal="سكالبينج شراء ⚡🔼"
        tp1=round(entry+0.6*atr1,2); tp2=round(entry+1.1*atr1,2); sl=round(entry-0.5*atr1,2)
        confidence=min(98.0,(rsi5-50)*1.5+max(0,vol_rel-100)*0.15+max(0,(kL-dL))*0.8)
        note.append("شروط قوية: RSI5/Stoch/شمعة دفع/حجم مؤكد")
    elif sell_cond:
        signal="سكالبينج بيع ⚡🔽"
        tp1=round(entry-0.6*atr1,2); tp2=round(entry-1.1*atr1,2); sl=round(entry+0.5*atr1,2)
        confidence=min(98.0,(50-rsi5)*1.5+max(0,vol_rel-100)*0.15+max(0,(dL-kL))*0.8)
        note.append("شروط قوية: RSI5/Stoch/شمعة دفع/حجم مؤكد")
    else:
        signal="فرصة محتملة — انتظار تأكيد 🔎"
        tp1=round(entry+0.6*atr1,2); tp2=round(entry+1.0*atr1,2); sl=round(entry-0.3*atr1,2)
        confidence=max(10.0,min(60.0,abs(rsi5-50)*1.2+max(0,vol_rel-100)*0.1))
        note.append("الظروف غير مكتملة — خطة احترازية")

    lines = [
        "⚡ **سكالبينج سري – 1m (محسّن)**",
        "بدء التحليل...\n🔄 جاري تحديث السعر اللحظي...",
        f"✅ السعر اللحظي المحدث: ${price:.2f}",
        f"🕒 {now}",
        "",
        f"📍 الإشارة: {signal}",
        f"📈 RSI5: {rsi5:.1f} | Stoch K/D: {kL:.1f}/{dL:.1f} | VolumeRel: {vol_rel:.0f}%",
        f"📍 Entry: {entry:.2f} | 🎯 TP1: {tp1} | 🎯 TP2: {tp2} | 🛡️ SL: {sl}",
        "⏱️ المدة المتوقعة: ≈ 5-15 دقيقة",
        f"⭐ الثقة: {confidence:.1f}%",
    ]
    if note: lines += ["", "ملاحظات:", *note]
    lines += ["", "المصدر: TwelveData"]
    return "\n".join(lines)

# =======================================================
#                Reports: Titan Executive (H1/H4/D1)
# =======================================================
def generate_titan_executive_report() -> str:
    price = fetch_realtime_price()
    if price is None: return API_ERROR_NOTICE
    h4 = fetch_candles("4h", 260); d1 = fetch_candles("1day", 260); h1 = fetch_candles("1h", 260)
    if h4 is None or d1 is None or h1 is None: return API_ERROR_NOTICE

    tr_h4, e20_h4, e50_h4, rsi_h4, _ = detect_trend_label(h4)
    tr_d1, _, _, rsi_d1, _ = detect_trend_label(d1)

    h1["ATR14"] = atr(h1, 14); atr_h1 = float(h1["ATR14"].iloc[-1])
    atr_h4 = float(atr(h4, 14).iloc[-1])

    sup, res = support_resistance_swing(h4, 120)
    P, S1, S2, R1, R2 = pivots_from_daily(d1)
    brk = breakout_level(h4, 48)
    struct = structure_note(h4)
    fib_lo, fib_hi, f382, f500, f618 = fib_levels_from_swings(h4, 140)

    magnet = P
    nuclear_sl = round(sup - 0.6*atr_h4, 2)
    t1 = round(price + 0.8*atr_h1, 2); t2 = round(price + 1.6*atr_h1, 2); t3 = round(price + 2.6*atr_h1, 2)

    flow_idx = float(np.clip(100*(atr_h1/np.nanmedian(h1["ATR14"].tail(60))), 60, 180))
    flow_line = f"شراء مكثف {flow_idx:.0f}% من المتوسط" if tr_h4.startswith("صاعد") else f"بيع نشط {flow_idx:.0f}% من المتوسط"

    score = 0
    if tr_h4.startswith("صاعد"): score+=2
    if price > P: score+=1
    if rsi_h4 > 55: score+=1
    if price > brk: score+=1
    if tr_d1.startswith("صاعد"): score+=1

    alpha = min(92, 60 + score*6)
    charlie = max(6, 20 - score*3)
    bravo = max(8, 100 - alpha - charlie)

    entry = price; add_on = brk
    take30 = round(entry + max(0.5*atr_h1, 0.5*atr_h4), 2)
    take40 = t2; runner = t3
    rr = (take30 - entry) / max(entry - nuclear_sl, 1e-6)
    rr_txt = f"1:{rr:.1f}"
    bias_daily = "إيجابي" if tr_d1.startswith("صاعد") else ("سلبي" if tr_d1.startswith("هابط") else "حيادي")
    vol_note = "متوسط مرتفع" if atr_h1 > np.nanmedian(h1["ATR14"].tail(60)) else "متوسط"

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "بدء التحليل...","🔄 جاري تحديث السعر اللحظي...",f"✅ السعر اللحظي المحدث: ${price:.2f}",
        "═══════════════════════════════════════════════════════════════════════════════",
        "🦅 تحليل التيتان الذهبي - XAUUSD | الجلسة: نيويورك | الأولوية: عاجل",
        "═══════════════════════════════════════════════════════════════════════════════",
        "🎯 **الملخص التنفيذي:**",
        f"الذهب يختبر مناطق حرجة قرب ${price:.2f} بزخم {('قوي' if rsi_h4>55 else 'متوازن')}, مع احتمالية صعود نحو ${t2:.2f} خلال الساعات القادمة.",
        "⚡️ **المعلومات الفورية:**",
        f"• السعر: ${price:.2f} | ترند 4H: {tr_h4} | انحياز يومي: {bias_daily}",
        f"• هيكل السوق: {struct}",
        f"• نظام التقلب: {vol_note}",
        f"• التدفق المؤسسي: {flow_line}",
        "🔥 **مستويات احترافية:**",
        f"• Pivot (المغناطيس): ${P} | R1: {R1} | S1: {S1}",
        f"• دعم/مقاومة تأريخية: {sup} / {res}",
        f"• فيبوناتشي (آخر موجة): 38.2%={f382} | 50%={f500} | 61.8%={f618}",
        f"• انفجار سعري محتمل (Breakout): ${brk}",
        f"• الإيقاف النووي: ${nuclear_sl}",
        "⚖️ **مصفوفة الاحتمال:**",
        f"• السيناريو ألفا [{alpha}%]: صعود لـ ${round((t1+t2)/2,2)}-${t2}",
        f"• برافو [{bravo}%]: توطيد ${sup}-${brk}",
        f"• تشارلي [{charlie}%]: تصحيح ${sup}",
        "💰 **استراتيجية:**",
        f"• دخول: ${entry:.2f} | إضافة عند كسر ${add_on}",
        f"• TP: {take30} / {take40} / {runner}",
        f"• SL: {nuclear_sl} | **R:R** {rr_txt}",
        "═══════════════════════════════════════════════════════════════════════════════",
        f"⚡️ **الثقة:** {min(98,int(alpha))}/100 | 🕒 آخر تحديث: {now}",
    ]
    return "\n".join(lines)

# =======================================================
#                 Reports: Oracle Prophet
# =======================================================
def generate_oracle_prophet_report() -> str:
    h4 = fetch_candles("4h", 300); d1 = fetch_candles("1day", 300)
    if h4 is None or d1 is None or h4.empty or d1.empty:
        return API_ERROR_NOTICE
    tr_h4,_,_,rsi_h4,_ = detect_trend_label(h4)
    tr_d1,_,_,rsi_d1,_ = detect_trend_label(d1)
    atr_h4 = float(atr(h4, 14).iloc[-1]); atr_d1 = float(atr(d1, 14).iloc[-1])
    price = float(h4["Close"].iloc[-1]); now = h4.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    sup,res = support_resistance_swing(h4, 160); P,_,_,R1,R2 = pivots_from_daily(d1)

    score=0
    if tr_h4.startswith("صاعد"): score+=2
    if tr_d1.startswith("صاعد"): score+=2
    if price>P: score+=1
    if rsi_h4>55: score+=1

    alpha=min(90, 50 + score*8); bravo=max(8, 100-alpha-10); charlie=100-alpha-bravo
    t_short_lo=round(price - 1.0*atr_h4, 2); t_short_hi=round(price + 1.0*atr_h4, 2)
    target_mid=round(price + 1.2*atr_h4, 2); target_far=round(price + 2.8*atr_d1, 2)

    lines=[
        "═══════════════════════════════════════════════════════════════",
        "🔮 Oracle Prophet – XAUUSD",
        "═══════════════════════════════════════════════════════════════",
        "📊 الوضع الحالي:",
        f"• السعر اللحظي: ${price:.2f}",
        f"• الزخم: {tr_h4}",
        "• الموقع في الدورة: متوسط/قوي",
        f"🕒 آخر تحديث: {now}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⏰ التوقعات الزمنية:",
        "📍 قصير المدى (1-24 ساعة):",
        "╔════════════════════════════════════════╗",
        f"║ الاتجاه المتوقع: {'صاعد مع تذبذب' if score>1 else 'تذبذب'}     ║",
        f"║ النطاق المحتمل: ${t_short_lo} - ${t_short_hi} ║",
        f"║ احتمال التحقق: {min(90, 60 + score*5)}%                     ║",
        "╚════════════════════════════════════════╝",
        "📍 متوسط المدى (2-7 أيام):",
        "╔════════════════════════════════════════╗",
        f"║ السيناريو الأساسي: اختراق نحو ${target_mid}          ║",
        f"║ الاحتمال: {alpha}%                                   ║",
        "╚════════════════════════════════════════╝",
        "📍 طويل المدى (1-4 أسابيع):",
        f"• الاتجاه العام: {'صاعد قوي' if score>2 else 'معتدل'}",
        f"• النطاق الأوسع: ${round(price-4*atr_d1,2)} - ${round(price+6*atr_d1,2)}",
        f"• الهدف الاستراتيجي: ${target_far}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "المستويات الاستراتيجية:",
        f"• دعم رئيسي: ${sup}",
        f"• مقاومة رئيسية: ${res}",
        f"• Pivot: ${P}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 مصفوفة الاحتمالات:",
        f"• فوق {round(price+1.0*atr_h4,2)} خلال 24 ساعة: {min(90, alpha)}%",
        f"• فوق {round(price+2.0*atr_h4,2)} خلال أسبوع: {min(90, alpha+5)}%",
        f"• فوق {target_far} خلال شهر: {min(95, alpha+10)}%",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 نقاط التحول:",
        f"1️⃣ المستوى: ${round(price+1.5*atr_h4,2)} | اختراق صاعد | الاحتمال: {min(85, alpha)}%",
        f"2️⃣ المستوى: ${round(price+3.0*atr_d1,2)} | مقاومة قوية/اختراق تاريخي",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚡ التوصية الاستراتيجية:",
        "• يومي: شراء الارتدادات داخل النطاق",
        "• سوينج: تمسّك بالصعود حول Pivot",
        "• استثمار: تجميع عند الدعوم الرئيسية",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🕒 آخر تحديث: {now}",
        "═══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)

# =======================================================
#            Reports: Patterns & Candle Analysis
# =======================================================
def generate_patterns_report() -> str:
    h1 = fetch_candles("1h", 220)
    if h1 is None or h1.empty:
        return API_ERROR_NOTICE
    pats = detect_candle_patterns(h1)
    chart = detect_chart_pattern(h1)
    lower, mid, upper = bollinger_bands(h1["Close"], n=20, k=2.0)
    price = float(h1["Close"].iloc[-1]); now = h1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    bl = round(float(lower.iloc[-1]),2); bm = round(float(mid.iloc[-1]),2); bu = round(float(upper.iloc[-1]),2)
    lines = [
        "🧩 **تحليل الأنماط والشموع – H1**",
        f"💵 السعر: ${price:.2f}",
        f"📎 بولنجر: أسفل={bl} | وسط={bm} | أعلى={bu}",
        f"⏱️ {now}",
        "—"*40
    ]
    if pats:
        lines += ["📌 شموع مكتشفة:"] + [f"• {p}" for p in pats]
    else:
        lines += ["📌 لا توجد شموع انعكاسية قوية مؤخرًا."]
    lines += ["", f"📈 النمط العام: {chart}", "المصدر: TwelveData"]
    return "\n".join(lines)

# =======================================================
#              Reports: RSI Divergence (H1)
# =======================================================
def generate_divergence_report() -> str:
    h1 = fetch_candles("1h", 260)
    if h1 is None or h1.empty:
        return API_ERROR_NOTICE
    h1["RSI14"] = rsi(h1["Close"], 14)
    price=float(h1["Close"].iloc[-1]); rsi_last=float(h1["RSI14"].iloc[-1])

    # قمة/قاع متأخرين بسيطين
    note = "لا يوجد انحراف واضح."
    try:
        highs = h1["High"].rolling(5).max()
        lows  = h1["Low"].rolling(5).min()
        swing_high = highs.iloc[-20:-1].idxmax(); prev_high = highs.iloc[-40:-21].idxmax()
        swing_low  = lows.iloc[-20:-1].idxmin(); prev_low  = lows.iloc[-40:-21].idxmin()

        # سلبي
        if h1["High"].iloc[swing_high] > h1["High"].iloc[prev_high] and \
           h1["RSI14"].iloc[swing_high] < h1["RSI14"].iloc[prev_high]:
            note = "⚠️ انحراف سلبي (احتمال تصحيح)"
        # إيجابي
        if h1["Low"].iloc[swing_low] < h1["Low"].iloc[prev_low] and \
           h1["RSI14"].iloc[swing_low] > h1["RSI14"].iloc[prev_low]:
            note = "✨ انحراف إيجابي (احتمال ارتداد)"
    except:
        pass

    now = h1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    return (
        "🔀 **تحليل الانحرافات – RSI/H1**\n"
        f"💵 السعر: ${price:.2f} | RSI(14): {rsi_last:.1f}\n"
        f"📌 النتيجة: {note}\n"
        f"🕒 {now}\n"
        "المصدر: TwelveData"
    )

# =======================================================
#                 Reports: Supply / Demand
# =======================================================
def find_supply_demand_zones(df: pd.DataFrame, lookback=160) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    recent = df.tail(min(lookback, len(df)))
    lows  = recent.nsmallest(3, "Low")["Low"].values
    highs = recent.nlargest(3, "High")["High"].values
    if len(lows)<3 or len(highs)<3:
        return (0,0),(0,0)
    demand = (round(float(np.mean(lows)*0.997),2), round(float(np.mean(lows)*1.003),2))
    supply = (round(float(np.mean(highs)*0.997),2), round(float(np.mean(highs)*1.003),2))
    return demand, supply

def generate_supply_demand_report() -> str:
    h1 = fetch_candles("1h", 260)
    if h1 is None or h1.empty:
        return API_ERROR_NOTICE
    price=float(h1["Close"].iloc[-1]); now=h1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    demand, supply = find_supply_demand_zones(h1, 160)
    sup, res = support_resistance_swing(h1, 160)
    return (
        "📦 **مناطق العرض والطلب – H1**\n"
        f"💵 السعر: ${price:.2f}\n"
        f"🟢 منطقة الطلب: {demand[0]} – {demand[1]}\n"
        f"🔴 منطقة العرض: {supply[0]} – {supply[1]}\n"
        f"🛡️ دعم: {sup} | 🧱 مقاومة: {res}\n"
        f"🕒 {now}\n"
        "المصدر: TwelveData"
    )

# =======================================================
#             Reports: Session & Sentiment
# =======================================================
def generate_session_sentiment_report() -> str:
    h1 = fetch_candles("1h", 260)
    if h1 is None or h1.empty:
        return API_ERROR_NOTICE
    sess = session_tag()
    senti = sentiment_logic(h1)
    a = float(atr(h1,14).iloc[-1]); price=float(h1["Close"].iloc[-1]); now=h1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    return (
        "🕒 **تحليل الجلسة & المزاج – H1**\n"
        f"🌍 الجلسة الحالية: {sess}\n"
        f"🧠 مزاج السوق (Sentiment): {senti}\n"
        f"📊 ATR(14): {a:.2f}\n"
        f"💵 السعر: ${price:.2f}\n"
        f"🕒 {now}\n"
        "المصدر: TwelveData"
    )

# =======================================================
#                  Reports: Meta AI (خلاصة)
# =======================================================
def generate_meta_ai_report() -> str:
    h4 = fetch_candles("4h", 260); d1 = fetch_candles("1day", 260); h1 = fetch_candles("1h", 240)
    if h4 is None or d1 is None or h1 is None or h4.empty or d1.empty or h1.empty:
        return API_ERROR_NOTICE
    price=float(h1["Close"].iloc[-1]); now=h1.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M UTC")
    tr_h4,_,_,rsi_h4,_ = detect_trend_label(h4); tr_d1,_,_,rsi_d1,_ = detect_trend_label(d1)
    pats = detect_candle_patterns(h1); pattern = detect_chart_pattern(h1)
    # انحراف مبسط:
    h1["RSI14"]=rsi(h1["Close"],14)
    rsinote="—"
    try:
        highs=h1["High"].rolling(5).max(); swing_high = highs.iloc[-20:-1].idxmax(); prev_high = highs.iloc[-40:-21].idxmax()
        if h1["High"].iloc[swing_high] > h1["High"].iloc[prev_high] and h1["RSI14"].iloc[swing_high] < h1["RSI14"].iloc[prev_high]:
            rsinote="⚠️ انحراف سلبي"
        lows=h1["Low"].rolling(5).min(); swing_low=lows.iloc[-20:-1].idxmin(); prev_low=lows.iloc[-40:-21].idxmin()
        if h1["Low"].iloc[swing_low] < h1["Low"].iloc[prev_low] and h1["RSI14"].iloc[swing_low] > h1["RSI14"].iloc[prev_low]:
            rsinote="✨ انحراف إيجابي"
    except:
        pass
    demand,supply = find_supply_demand_zones(h1,160)
    sess=session_tag(); senti=sentiment_logic(h1)
    ell = simple_elliott_hint(h4)
    fib_lo,fib_hi,f382,f500,f618 = fib_levels_from_swings(h4, 140)

    bullets=[
        f"ترند H4: {tr_h4} | RSI(H4): {rsi_h4:.1f}",
        f"ترند D1: {tr_d1} | RSI(D1): {rsi_d1:.1f}",
        f"جلسة: {sess} | مزاج: {senti}",
        f"نمط/شموع: {pattern}"+ (f" | إشارات: {', '.join(pats)}" if pats else ""),
        f"انحراف RSI: {rsinote}",
        f"موجة إليوت (تلميح): {ell}",
        f"فيبوناتشي: 38.2%={f382} | 50%={f500} | 61.8%={f618}",
        f"منطقة طلب: {demand[0]}–{demand[1]} | منطقة عرض: {supply[0]}–{supply[1]}",
    ]

    # توصية ذكية:
    rec="انتظار تأكيد ✅"
    if tr_h4.startswith("صاعد") and "انحراف سلبي" not in rsinote:
        rec="شراء الارتدادات من الطلب مع تأكيد كسر قمة قريبة"
    if tr_h4.startswith("هابط") and "انحراف إيجابي" not in rsinote:
        rec="بيع الارتدادات من العرض مع كسر قاع قريب"

    lines=[
        "🧠 **Meta AI – الخلاصة الذكية**",
        f"💵 السعر: ${price:.2f}",
        f"🕒 {now}",
        "—"*40,
        "📌 نقاط رئيسية:"
    ] + [f"• {b}" for b in bullets] + [
        "—"*40,
        f"🎯 توصية منطقية: {rec}",
        "⚠️ تنبيه: هذه قراءة تحليلية وليست نصيحة استثمارية.",
        "المصدر: TwelveData"
    ]
    return "\n".join(lines)

# =======================================================
# analysis.py — Smart Behavioral + Technical Gold Trader
# =======================================================
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TWELVEDATA_KEY")
SYMBOL = os.getenv("SYMBOL", "XAU/USD")
BASE_URL = "https://api.twelvedata.com"

API_ERROR_NOTICE = "⚠️ تعذّر جلب البيانات حالياً. جرّب بعد لحظات."

# -------------------------------------------------------
# Data Fetch
# -------------------------------------------------------
def _assert():
    if not API_KEY:
        raise RuntimeError("❌ TWELVEDATA_KEY مفقود")

def fetch_candles(interval="1h", size=300):
    try:
        _assert()
        r = requests.get(f"{BASE_URL}/time_series", params={
            "symbol": SYMBOL, "interval": interval,
            "apikey": API_KEY, "outputsize": size,
            "order": "ASC", "timezone": "UTC"
        }, timeout=10)
        j = r.json()
        if "values" not in j:
            return None
        df = pd.DataFrame(j["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ["open","high","low","close"]:
            df[c.capitalize()] = df[c].astype(float)
        return df[["datetime","Open","High","Low","Close"]].sort_values("datetime")
    except:
        return None

def fetch_last_price():
    h1 = fetch_candles("1h", 2)
    if h1 is None or h1.empty:
        return None
    return float(h1["Close"].iloc[-1])

# -------------------------------------------------------
# Indicators
# -------------------------------------------------------
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs= up.rolling(n).mean() / dn.rolling(n).mean()
    return 100-(100/(1+rs))
def atr(df,n=14):
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-df["Close"].shift()).abs(),
        (df["Low"]-df["Close"].shift()).abs()
    ],axis=1).max(axis=1)
    return tr.rolling(n).mean()

# -------------------------------------------------------
# Smart Behavioral Report
# -------------------------------------------------------
def generate_behavior_report():
    h4 = fetch_candles("4h",260)
    h1 = fetch_candles("1h",260)
    if h4 is None or h1 is None: return API_ERROR_NOTICE

    price = float(h1["Close"].iloc[-1])
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Trend logic
    e20_h4 = ema(h4["Close"],20).iloc[-1]
    e50_h4 = ema(h4["Close"],50).iloc[-1]
    rsi_h1 = rsi(h1["Close"],14).iloc[-1]

    trend_score = 1 if e20_h4>e50_h4 else -1
    momentum_score = (rsi_h1-50)/25

    # Volume Proxy via ATR Scale
    v_h1 = atr(h1,14).iloc[-1]
    medium = np.nanmedian(atr(h1,14).tail(60))
    vol_score = (v_h1/medium)-1

    # Price Position Logic
    high = h4["High"].tail(80).max()
    low  = h4["Low"].tail(80).min()
    pos_score = ((price-low)/(high-low))*2-1

    # Behavioral Ensemble Score
    score = (0.45*trend_score + 0.30*momentum_score + 0.25*pos_score)

    # Confidence
    confidence = round(50 + score*40 + vol_score*10,1)
    confidence = max(15,min(95,confidence))

    # Decision
    if confidence>=65 and score>0:
        decision="شراء"
        side=1
    elif confidence>=65 and score<0:
        decision="بيع"
        side=-1
    else:
        decision="انتظار"
        side=0

    # TP/SL
    r=atr(h1,14).iloc[-1] or 1
    if side!=0:
        tp1 = round(price + side*0.6*r,2)
        tp2 = round(price + side*1.2*r,2)
        sl  = round(price - side*0.9*r,2)
    else:
        tp1=tp2=sl=None

    tone=""
    if decision=="شراء":
        tone="• بصراحة المشترين مسيطرين عالسوق أكثر\n• كل هبوط صغير يتم استيعابه بسرعة"
    elif decision=="بيع":
        tone="• البياعة شايفين الفرصة… كل طلعة عم تنباع\n• كأن السوق متوتر ومش مستقر فوق"
    else:
        tone="• الوضع مش مريح للدخول الآن\n• الأفضل نراقب رد الفعل عند مستويات قريبة"

    lines=[
        f"🧠 التحليل السلوكي الذكي — مود B",
        f"⏱ {now} | 💵 السعر: ${price:.2f}",
        "——————————————————————————",
        tone,
        "",
        f"📌 نقاط مختصرة:",
        f"• الترند العام (H4): {'صاعد' if trend_score>0 else 'هابط'}",
        f"• زخم H1: {rsi_h1:.1f}",
        f"• موقع السعر داخل النطاق: {pos_score:.2f}",
        f"• حجم الحركة: {('قوي' if vol_score>0 else 'ضعيف')}",
        "",
        f"🎯 القرار: {decision} | الثقة: {confidence}%",
    ]
    if side!=0:
        lines+=[
            f"📍 Entry: {price:.2f}",
            f"🎯 TP1: {tp1} | 🎯 TP2: {tp2}",
            f"🛡️ SL: {sl}",
        ]
    lines+=[
        "",
        "⚠️ هذا تحليل سلوكي احترافي. إدارة رأس المال مسؤوليتك.",
        "المصدر السعري: TwelveData"
    ]
    return "\n".join(lines)



    # =======================================================
# توصية افتتاح الجلسات (أوتوماتيك بحسب توقيت فلسطين) — مود B
# =======================================================
from typing import Tuple
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except:
    ZoneInfo = None

def _palestine_now():
    # نحاول Asia/Gaza، وإن لم تتوفر ن fallback إلى UTC+3
    try:
        if ZoneInfo:
            return datetime.now(ZoneInfo("Asia/Gaza"))
    except:
        pass
    # Fallback مبسط على UTC+3
    return datetime.utcnow() + pd.Timedelta(hours=3)

def _detect_pal_session(ts_local) -> Tuple[str, str]:
    """
    يُرجع (اسم الجلسة, سبب/وصف موجز)
    أوقات فلسطين التقريبية:
      - آسيا: 01:00–10:00
      - لندن: 10:00–15:30
      - نيويورك: 15:30–23:59
    """
    h = ts_local.hour + ts_local.minute/60.0
    if 1.0 <= h < 10.0:
        return "آسيا", "افتتاح آسيا عادةً يحدد نبرة اليوم المبكرة"
    if 10.0 <= h < 15.5:
        return "لندن", "افتتاح لندن يزيد السيولة ويرسم اتجاه الظهيرة"
    return "نيويورك", "افتتاح نيويورك هو الأقوى تأثيرًا على الذهب"

def _session_open_levels(h1: pd.DataFrame, d1: pd.DataFrame) -> Tuple[float,float,float,float,float]:
    """Pivot + R1/S1 + ATR"""
    P, S1, S2, R1, R2 = pivots_from_daily(d1)
    a = float(atr(h1, 14).iloc[-1])
    return P, R1, S1, a, float(h1["Close"].iloc[-1])

def _entry_targets_from_context(side: int, price: float, atr_h1: float) -> Tuple[float,float,float]:
    if side == 0 or atr_h1 <= 0:
        return None, None, None
    tp1 = round(price + side*0.9*atr_h1, 2)
    tp2 = round(price + side*1.6*atr_h1, 2)
    sl  = round(price - side*1.1*atr_h1, 2)
    return tp1, tp2, sl



def generate_session_open_reco() -> str:

    """
    توصية افتتاح الجلسة الحالية تلقائياً حسب توقيت فلسطين (مود B).
    تُنتج: قراءة سلوكية + خطة دخول/خروج + إلغاء فكرة منطقي.
    """
    # جلب البيانات
    h4 = fetch_candles("4h", 300)
    h1 = fetch_candles("1h", 400)
    m15 = fetch_candles("15min", 600)
    d1 = fetch_candles("1day", 260)

    if any(x is None or x.empty for x in [h4, h1, m15, d1]):
        return API_ERROR_NOTICE

    # توقيت فلسطين وتحديد الجلسة
    now_local = _palestine_now()
    sess_name, sess_note = _detect_pal_session(now_local)

    # أساسيات القراءة
    price = float(h1["Close"].iloc[-1])
    tr_h4, _, _, rsi_h4, _ = detect_trend_label(h4)
    rsi_h1 = float(rsi(h1["Close"], 14).iloc[-1])
    P, R1, S1, atr_h1, _ = _session_open_levels(h1, d1)

    # حجم/رفض أولي على M15
    last = m15.iloc[-1]
    body = float(abs(last["Close"] - last["Open"]))
    rng  = float(last["High"] - last["Low"]) if (last["High"] - last["Low"]) != 0 else 1e-9
    upper = float(last["High"] - max(last["Close"], last["Open"]))
    lower = float(min(last["Close"], last["Open"]) - last["Low"])
    is_reject_up   = (body/rng <= 0.35) and (upper >= 2.2*body) and (upper >= 0.4*rng)
    is_reject_down = (body/rng <= 0.35) and (lower >= 2.2*body) and (lower >= 0.4*rng)

    # اتجاه افتتاحي منطقي:
    # - ترند H4 يعطي الانحياز العام
    # - موقع السعر من Pivot اليومي يزيد/ينقص الانحياز
    # - RSI(H1) يضيف وزن للزخم
    score = 0.0
    score += 0.9 if tr_h4.startswith("صاعد") else (-0.9 if tr_h4.startswith("هابط") else 0.0)
    score += 0.5 if price > P else -0.5
    score += (rsi_h1 - 50.0) / 18.0  # تقريبًا ±1.5 كحد أقصى
    if is_reject_down:  # رفض سفلي يدعم الشراء
        score += 0.6
    if is_reject_up:    # رفض علوي يدعم البيع
        score -= 0.6

    # قرار افتتاحي
    if score >= 0.85:
        decision, side = "شراء", 1
    elif score <= -0.85:
        decision, side = "بيع", -1
    else:
        decision, side = "انتظار", 0

    # مناطق دخول منطقية:
    # - الشراء: فوق Pivot/قمة قريبة
    # - البيع: تحت Pivot/قاع قريب
    # - نستخدم ATR لصناعة TP/SL
    if side == 1:
        trigger = max(P, breakout_level(h1, 36))
    elif side == -1:
        # قاع قريب منطقي
        trigger = min(P, round(float(h1["Low"].tail(36).min()), 2))
    else:
        trigger = None

    tp1, tp2, sl = _entry_targets_from_context(side, price, atr_h1)

    # ثقة افتتاحية معقولة
    base_conf = 55 + min(30, abs(score)*20)
    if is_reject_down and decision == "شراء":
        base_conf += 6
    if is_reject_up and decision == "بيع":
        base_conf += 6
    confidence = int(np.clip(base_conf, 30, 92))

    now_utc = now_utc_str()
    lines = []
    lines.append(f"🚀 توصية افتتاح الجلسة — {sess_name}")
    lines.append(f"⏱ {now_local.strftime('%Y-%m-%d %H:%M')} 🇵🇸 | {now_utc} | 💵 السعر: ${price:.2f}")
    lines.append("—"*50)
    # نبرة بشرية مختصرة حسب القرار
    if decision == "شراء":
        lines += [
            "• المشترين ظاهرين على الفريمات الكبيرة، والدفاع عند الدعوم واضح.",
            "• الأفضل يكون الدخول بعد تأكيد بسيط لتجنّب مصيدة افتتاح."
        ]
    elif decision == "بيع":
        lines += [
            "• البائعين ماسكين الصورة العامة عند الافتتاح.",
            "• كل ارتداد ضعيف يُقابله تصريف سريع — هذا يدعم البيع بعد الكسر."
        ]
    else:
        lines += [
            "• السوق متوازن عند الافتتاح ولم تظهر نية حاسمة بعد.",
            "• نراقب كسر/اختراق واضح قبل أي دخول."
        ]

    lines.append("")
    lines.append("📌 معطيات سريعة:")
    lines.append(f"• {sess_note}")
    lines.append(f"• ترند H4: {tr_h4} | RSI(H1): {rsi_h1:.1f}")
    lines.append(f"• Pivot اليومي: {P} | R1={R1} | S1={S1}")
    if is_reject_down:
        lines.append("• رفض سعري سفلي على 15min (إشارة إيجابية للشراء)")
    if is_reject_up:
        lines.append("• رفض سعري علوي على 15min (إشارة سلبية للشراء)")

    lines.append("")
    if side != 0:
        # خطة دخول افتتاحية
        enter_hint = (f"فوق {trigger}" if side==1 else f"تحت {trigger}") if trigger else "بعد تأكيد شمعة دفع"
        lines.append("🎯 خطة الدخول:")
        lines.append(f"• {decision} {enter_hint} ✅")
        lines.append(f"• TP1: {tp1} | TP2: {tp2}")
        lines.append(f"🛡️ SL: {sl}")
        # إلغاء الفكرة
        cancel_lvl = round(P if side==-1 else P, 2)
        # لو شراء وفشل وثبت تحت Pivot → إلغاء. لو بيع وثبت فوق Pivot → إلغاء.
        cancel_note = "إلغاء الفكرة إذا عاد السعر وثبت " + ("تحت Pivot" if side==1 else "فوق Pivot")
        lines.append(f"❌ إلغاء الفكرة: {cancel_note}")
        lines.append(f"⭐ الثقة: {confidence}%")
    else:
        lines.append("🎯 القرار: انتظار تأكيد — السوق لم يمنح أفضلية واضحة")
        lines.append("• راقب تفعيل الشراء فوق القمم القريبة أو البيع تحت القيعان القريبة.")
    lines.append("")
    lines.append("⚠️ تذكير: هذه توصية معلوماتية. نفّذ بعقلانية ومع إدارة مخاطر.")
    lines.append("المصدر: TwelveData")
    return "\n".join(lines)

def generate_final_decision_report() -> str:
    h4 = fetch_candles("4h", 260)
    h1 = fetch_candles("1h", 260)
    d1 = fetch_candles("1day", 260)
    price = fetch_realtime_price()

    if any(x is None or x.empty for x in [h4, h1, d1]) or price is None:
        return API_ERROR_NOTICE

    # Trend & Levels
    tr_h4, _, _, rsi_h4, _ = detect_trend_label(h4)
    rsi_h1 = float(rsi(h1["Close"], 14).iloc[-1])
    P, S1, S2, R1, R2 = pivots_from_daily(d1)
    h1["ATR14"] = atr(h1,14)
    atr_val = max(0.1, float(h1["ATR14"].iloc[-1]))

    # Session intelligent impact
    sess = session_tag()
    tf_signal = "5m" if sess in ["أوروبا المبكرة", "أمريكا"] else "15m"
    session_boost = {
        "آسيا": -6,
        "أوروبا المبكرة": +3,
        "أمريكا": +7,
        "أوروبا المتأخرة": -2,
    }.get(sess, 0)

    score = 0
    score += 2 if tr_h4.startswith("صاعد") else (-2 if tr_h4.startswith("هابط") else 0)
    score += 1 if price > P else -1
    score += 1 if rsi_h1 > 52 else (-1 if rsi_h1 < 48 else 0)
    score += round(session_boost / 4)

    if score >= 2:
        side = 1
        decision = "شراء"
        trigger = round(P + 0.25*atr_val, 2)
        tp1 = round(trigger + 1.1*atr_val, 2)
        tp2 = round(trigger + 2.2*atr_val, 2)
        sl  = round(P - 0.8*atr_val, 2)
        reason = "📈 المشترون مسيطرين بزخم أعلى من Pivot"
        cancel = f"إذا عاد السعر وثبت تحت Pivot {P}"
    elif score <= -2:
        side = -1
        decision = "بيع"
        trigger = round(P - 0.25*atr_val, 2)
        tp1 = round(trigger - 1.1*atr_val, 2)
        tp2 = round(trigger - 2.2*atr_val, 2)
        sl  = round(P + 0.8*atr_val, 2)
        reason = "📉 البائعون ضاغطين أسفل Pivot بزخم هابط"
        cancel = f"إذا عاد السعر وثبت فوق Pivot {P}"
    else:
        side = 0
        decision = "انتظار"
        trigger = tp1 = tp2 = sl = None
        reason = "⚪ السوق متردد ولا توجد أفضلية واضحة"
        cancel = "بانتظار كسر/اختراق واضح قبل الدخول."

    conf = int(np.clip(62 + score*10 + session_boost, 30, 92))
    now = now_utc_str()

    lines = [
        "📌 **قرار السوق الآن – التوصية النهائية**",
        f"🕒 {now}",
        f"🌍 الجلسة: {sess}",
        f"💵 السعر الحالي: ${price:.2f}",
        "",
        f"🎯 القرار: {decision}",
    ]

    if side != 0:
        lines += [
            f"• التفعيل: {'فوق' if side==1 else 'تحت'} {trigger}",
            f"↳ يتم التفعيل فقط بعد إغلاق شمعة تأكيد {tf_signal} ✅",
            f"• TP1: {tp1}",
            f"• TP2: {tp2}",
            f"🛡 SL: {sl}",
        ]

    lines += [
        "",
        f"⭐ الثقة: {conf}%",
        "",
        "📌 سبب القرار:",
        reason,
        f"• ترند H4: {tr_h4}",
        f"• RSI(H1): {rsi_h1:.1f}",
        f"• Pivot: {P}",
        "",
        f"❌ إلغاء الفكرة: {cancel}",
        "",
        "⚠️ توصية معلوماتية. إدارة المخاطر مسؤوليتك.",
        "المصدر: TwelveData"
    ]

# زر توصية افتتاح الجلسة


def generate_session_open_reco() -> str:

    """
    توصية افتتاح الجلسة الحالية تلقائياً حسب توقيت فلسطين (مود B).
    تُنتج: قراءة سلوكية + خطة دخول/خروج + إلغاء فكرة منطقي.
    """
    # جلب البيانات
    h4 = fetch_candles("4h", 300)
    h1 = fetch_candles("1h", 400)
    m15 = fetch_candles("15min", 600)
    d1 = fetch_candles("1day", 260)

    if any(x is None or x.empty for x in [h4, h1, m15, d1]):
        return API_ERROR_NOTICE

    # توقيت فلسطين وتحديد الجلسة
    now_local = _palestine_now()
    sess_name, sess_note = _detect_pal_session(now_local)

    # أساسيات القراءة
    price = float(h1["Close"].iloc[-1])
    tr_h4, _, _, rsi_h4, _ = detect_trend_label(h4)
    rsi_h1 = float(rsi(h1["Close"], 14).iloc[-1])
    P, R1, S1, atr_h1, _ = _session_open_levels(h1, d1)

    # حجم/رفض أولي على M15
    last = m15.iloc[-1]
    body = float(abs(last["Close"] - last["Open"]))
    rng  = float(last["High"] - last["Low"]) if (last["High"] - last["Low"]) != 0 else 1e-9
    upper = float(last["High"] - max(last["Close"], last["Open"]))
    lower = float(min(last["Close"], last["Open"]) - last["Low"])
    is_reject_up   = (body/rng <= 0.35) and (upper >= 2.2*body) and (upper >= 0.4*rng)
    is_reject_down = (body/rng <= 0.35) and (lower >= 2.2*body) and (lower >= 0.4*rng)

    # اتجاه افتتاحي منطقي:
    # - ترند H4 يعطي الانحياز العام
    # - موقع السعر من Pivot اليومي يزيد/ينقص الانحياز
    # - RSI(H1) يضيف وزن للزخم
    score = 0.0
    score += 0.9 if tr_h4.startswith("صاعد") else (-0.9 if tr_h4.startswith("هابط") else 0.0)
    score += 0.5 if price > P else -0.5
    score += (rsi_h1 - 50.0) / 18.0  # تقريبًا ±1.5 كحد أقصى
    if is_reject_down:  # رفض سفلي يدعم الشراء
        score += 0.6
    if is_reject_up:    # رفض علوي يدعم البيع
        score -= 0.6

    # قرار افتتاحي
    if score >= 0.85:
        decision, side = "شراء", 1
    elif score <= -0.85:
        decision, side = "بيع", -1
    else:
        decision, side = "انتظار", 0

    # مناطق دخول منطقية:
    # - الشراء: فوق Pivot/قمة قريبة
    # - البيع: تحت Pivot/قاع قريب
    # - نستخدم ATR لصناعة TP/SL
    if side == 1:
        trigger = max(P, breakout_level(h1, 36))
    elif side == -1:
        # قاع قريب منطقي
        trigger = min(P, round(float(h1["Low"].tail(36).min()), 2))
    else:
        trigger = None

    tp1, tp2, sl = _entry_targets_from_context(side, price, atr_h1)

    # ثقة افتتاحية معقولة
    base_conf = 55 + min(30, abs(score)*20)
    if is_reject_down and decision == "شراء":
        base_conf += 6
    if is_reject_up and decision == "بيع":
        base_conf += 6
    confidence = int(np.clip(base_conf, 30, 92))

    now_utc = now_utc_str()
    lines = []
    lines.append(f"🚀 توصية افتتاح الجلسة — {sess_name}")
    lines.append(f"⏱ {now_local.strftime('%Y-%m-%d %H:%M')} 🇵🇸 | {now_utc} | 💵 السعر: ${price:.2f}")
    lines.append("—"*50)
    # نبرة بشرية مختصرة حسب القرار
    if decision == "شراء":
        lines += [
            "• المشترين ظاهرين على الفريمات الكبيرة، والدفاع عند الدعوم واضح.",
            "• الأفضل يكون الدخول بعد تأكيد بسيط لتجنّب مصيدة افتتاح."
        ]
    elif decision == "بيع":
        lines += [
            "• البائعين ماسكين الصورة العامة عند الافتتاح.",
            "• كل ارتداد ضعيف يُقابله تصريف سريع — هذا يدعم البيع بعد الكسر."
        ]
    else:
        lines += [
            "• السوق متوازن عند الافتتاح ولم تظهر نية حاسمة بعد.",
            "• نراقب كسر/اختراق واضح قبل أي دخول."
        ]

    lines.append("")
    lines.append("📌 معطيات سريعة:")
    lines.append(f"• {sess_note}")
    lines.append(f"• ترند H4: {tr_h4} | RSI(H1): {rsi_h1:.1f}")
    lines.append(f"• Pivot اليومي: {P} | R1={R1} | S1={S1}")
    if is_reject_down:
        lines.append("• رفض سعري سفلي على 15min (إشارة إيجابية للشراء)")
    if is_reject_up:
        lines.append("• رفض سعري علوي على 15min (إشارة سلبية للشراء)")

    lines.append("")
    if side != 0:
        # خطة دخول افتتاحية
        enter_hint = (f"فوق {trigger}" if side==1 else f"تحت {trigger}") if trigger else "بعد تأكيد شمعة دفع"
        lines.append("🎯 خطة الدخول:")
        lines.append(f"• {decision} {enter_hint} ✅")
        lines.append(f"• TP1: {tp1} | TP2: {tp2}")
        lines.append(f"🛡️ SL: {sl}")
        # إلغاء الفكرة
        cancel_lvl = round(P if side==-1 else P, 2)
        # لو شراء وفشل وثبت تحت Pivot → إلغاء. لو بيع وثبت فوق Pivot → إلغاء.
        cancel_note = "إلغاء الفكرة إذا عاد السعر وثبت " + ("تحت Pivot" if side==1 else "فوق Pivot")
        lines.append(f"❌ إلغاء الفكرة: {cancel_note}")
        lines.append(f"⭐ الثقة: {confidence}%")
    else:
        lines.append("🎯 القرار: انتظار تأكيد — السوق لم يمنح أفضلية واضحة")
        lines.append("• راقب تفعيل الشراء فوق القمم القريبة أو البيع تحت القيعان القريبة.")
    lines.append("")
    lines.append("⚠️ تذكير: هذه توصية معلوماتية. نفّذ بعقلانية ومع إدارة مخاطر.")
    lines.append("المصدر: TwelveData")
    return "\n".join(lines)

def generate_final_decision_report() -> str:
    h4 = fetch_candles("4h", 260)
    h1 = fetch_candles("1h", 260)
    d1 = fetch_candles("1day", 260)
    price = fetch_realtime_price()

    if any(x is None or x.empty for x in [h4, h1, d1]) or price is None:
        return API_ERROR_NOTICE

    # Trend & Levels
    tr_h4, _, _, rsi_h4, _ = detect_trend_label(h4)
    rsi_h1 = float(rsi(h1["Close"], 14).iloc[-1])
    P, S1, S2, R1, R2 = pivots_from_daily(d1)
    h1["ATR14"] = atr(h1,14)
    atr_val = max(0.1, float(h1["ATR14"].iloc[-1]))

    # Session intelligent impact
    sess = session_tag()
    tf_signal = "5m" if sess in ["أوروبا المبكرة", "أمريكا"] else "15m"
    session_boost = {
        "آسيا": -6,
        "أوروبا المبكرة": +3,
        "أمريكا": +7,
        "أوروبا المتأخرة": -2,
    }.get(sess, 0)

    score = 0
    score += 2 if tr_h4.startswith("صاعد") else (-2 if tr_h4.startswith("هابط") else 0)
    score += 1 if price > P else -1
    score += 1 if rsi_h1 > 52 else (-1 if rsi_h1 < 48 else 0)
    score += round(session_boost / 4)

    if score >= 2:
        side = 1
        decision = "شراء"
        trigger = round(P + 0.25*atr_val, 2)
        tp1 = round(trigger + 1.1*atr_val, 2)
        tp2 = round(trigger + 2.2*atr_val, 2)
        sl  = round(P - 0.8*atr_val, 2)
        reason = "📈 المشترون مسيطرين بزخم أعلى من Pivot"
        cancel = f"إذا عاد السعر وثبت تحت Pivot {P}"
    elif score <= -2:
        side = -1
        decision = "بيع"
        trigger = round(P - 0.25*atr_val, 2)
        tp1 = round(trigger - 1.1*atr_val, 2)
        tp2 = round(trigger - 2.2*atr_val, 2)
        sl  = round(P + 0.8*atr_val, 2)
        reason = "📉 البائعون ضاغطين أسفل Pivot بزخم هابط"
        cancel = f"إذا عاد السعر وثبت فوق Pivot {P}"
    else:
        side = 0
        decision = "انتظار"
        trigger = tp1 = tp2 = sl = None
        reason = "⚪ السوق متردد ولا توجد أفضلية واضحة"
        cancel = "بانتظار كسر/اختراق واضح قبل الدخول."

    conf = int(np.clip(62 + score*10 + session_boost, 30, 92))
    now = now_utc_str()

    lines = [
        "📌 **قرار السوق الآن – التوصية النهائية**",
        f"🕒 {now}",
        f"🌍 الجلسة: {sess}",
        f"💵 السعر الحالي: ${price:.2f}",
        "",
        f"🎯 القرار: {decision}",
    ]

    if side != 0:
        lines += [
            f"• التفعيل: {'فوق' if side==1 else 'تحت'} {trigger}",
            f"↳ يتم التفعيل فقط بعد إغلاق شمعة تأكيد {tf_signal} ✅",
            f"• TP1: {tp1}",
            f"• TP2: {tp2}",
            f"🛡 SL: {sl}",
        ]

    lines += [
        "",
        f"⭐ الثقة: {conf}%",
        "",
        "📌 سبب القرار:",
        reason,
        f"• ترند H4: {tr_h4}",
        f"• RSI(H1): {rsi_h1:.1f}",
        f"• Pivot: {P}",
        "",
        f"❌ إلغاء الفكرة: {cancel}",
        "",
        "⚠️ توصية معلوماتية. إدارة المخاطر مسؤوليتك.",
        "المصدر: TwelveData"
    ]

    return "\n".join(lines)
