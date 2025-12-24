import os, json, random, nest_asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)

from analysis import (
    generate_price_brief,
    generate_sniper_report,
    generate_secret_scalp_report,
    generate_titan_executive_report,
    generate_oracle_prophet_report,
    generate_patterns_report,
    generate_divergence_report,
    generate_supply_demand_report,
    generate_session_sentiment_report,
    generate_meta_ai_report,
    generate_behavior_report,
    generate_session_open_reco,
    generate_final_decision_report
)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

USERS_FILE = "users.json"
CODES_FILE = "codes.json"
TRIAL_FIELD = "trial_used"

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN Missing")

def load_json(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def is_user_active(uid: int) -> bool:
    users = load_json(USERS_FILE)
    u = users.get(str(uid))
    if not u or not u.get("expiry"): return False
    try:
        return datetime.fromisoformat(u["expiry"]) > datetime.utcnow()
    except:
        return False

def activate_user(uid: int, code: str) -> str:
    codes = load_json(CODES_FILE)
    users = load_json(USERS_FILE)

    if code not in codes:
        return "❌ كود غير صالح."
    if codes[code]["used"]:
        return "⚠️ الكود مستخدم سابقاً."

    expiry = datetime.utcnow() + timedelta(days=30)
    users[str(uid)] = {"expiry": expiry.isoformat(), TRIAL_FIELD: True}
    codes[code]["used"] = True
    codes[code]["user_id"] = uid

    save_json(USERS_FILE, users)
    save_json(CODES_FILE, codes)
    return f"✅ اشتراكك فعال حتى {expiry.date()}"

def generate_codes(n=10):
    codes = load_json(CODES_FILE)
    out=[]
    for _ in range(n):
        code = "GOLD_" + ''.join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
        while code in codes:
            code = "GOLD_" + ''.join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
        codes[code] = {"used": False, "user_id": None}
        out.append(code)
    save_json(CODES_FILE, codes)
    return out

def main_keyboard():
    return InlineKeyboardMarkup([

        [InlineKeyboardButton("القنّاص 🎯", callback_data="sniper"),
         InlineKeyboardButton("سكالبينج سري ⚡", callback_data="scalp")],

        [InlineKeyboardButton("تحليل التيتان 🦅", callback_data="titan"),
         InlineKeyboardButton("Oracle Prophet 🔮", callback_data="oracle")],

        [InlineKeyboardButton("🧠 Meta AI", callback_data="meta"),
         InlineKeyboardButton(" 🧩 الأنماط والشموع ", callback_data="patterns")],

        [InlineKeyboardButton("العرض والطلب 📦", callback_data="supply"),
         InlineKeyboardButton("  🔀 الانحرافات RSI ", callback_data="divergence")],

        [InlineKeyboardButton("التحليل السلوكي الذكي 🧠", callback_data="behavior"),
         InlineKeyboardButton("📌 قرار السوق الآن  ", callback_data="market_now")],

        [InlineKeyboardButton("توصيات افتتاح الجلسات 🕒", callback_data="session_open")],

        [InlineKeyboardButton("الجلسة والمزاج 🕒", callback_data="session")],

        [InlineKeyboardButton("تفعيل الاشتراك 🔐", callback_data="activate"),
         InlineKeyboardButton("حالة الاشتراك 📅", callback_data="status")],

        [InlineKeyboardButton(" 💬 تواصل مع المشرف …", url="https://t.me/da7lan")]

    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 أهلاً بك في *Titan Gold Bot*\n"
        "🎟️ الاشتراك وطلب الأكواد عبر المشرف:\n"
        "@da7lan\n\n"
        "اختَر نوع التحليل 👇",
        reply_markup=main_keyboard(), parse_mode="Markdown"
    )

async def activate_cmd(update: Update, context):
    if not context.args:
        return await update.message.reply_text("⚠️ استخدم:\n/activate GOLD_XXXXXX")
    code = context.args[0].upper().strip()
    uid = update.effective_user.id
    res = activate_user(uid, code)
    await update.message.reply_text(res)

async def gen_codes_cmd(update: Update, context):
    if update.effective_user.id != ADMIN_ID:
        return await update.message.reply_text("🚫 غير مسموح.")
    n = int(context.args[0]) if context.args else 10
    codes = generate_codes(n)
    await update.message.reply_text("\n".join(codes))

async def run_report(data, uid):
    if data == "price": return generate_price_brief()
    if data == "sniper": return generate_sniper_report()
    if data == "scalp": return generate_secret_scalp_report()
    if data == "titan": return generate_titan_executive_report()
    if data == "oracle": return generate_oracle_prophet_report()
    if data == "patterns": return generate_patterns_report()
    if data == "divergence": return generate_divergence_report()
    if data == "supply": return generate_supply_demand_report()
    if data == "session": return generate_session_sentiment_report()
    if data == "meta": return generate_meta_ai_report()
    if data == "behavior": return generate_behavior_report()
    if data == "market_now": return generate_final_decision_report()
    if data == "session_open": return generate_session_open_reco()
    return "❌ أمر غير معروف"

async def handle_buttons(update, context):
    q = update.callback_query
    data = q.data
    uid = q.from_user.id

    users = load_json(USERS_FILE)
    u = users.get(str(uid))

    if data not in ["activate", "status"]:
        if not is_user_active(uid):
            if not u or not u.get(TRIAL_FIELD):
                users[str(uid)] = {TRIAL_FIELD: True, "trial_at": datetime.utcnow().isoformat()}
                save_json(USERS_FILE, users)
                await q.message.reply_text("🎁 تم منحك تجربة مجانية!")
            else:
                return await q.message.reply_text(
                    "🚫 الاشتراك غير فعّال\nراسل: @da7lan",
                    reply_markup=main_keyboard()
                )

    if data == "status":
        if not u:
            msg = "❌ لم يتم التسجيل بعد."
        else:
            expiry = u.get("expiry")
            trial = u.get(TRIAL_FIELD)
            msg = f"📅 الاشتراك: {expiry if expiry else 'غير مفعّل'}\n🎁 التجربة: {'مستخدمة' if trial else 'متاحة'}"
        return await q.message.reply_text(msg, reply_markup=main_keyboard())

    if data == "activate":
        return await q.message.reply_text(
            "🔑 راسل المشرف للحصول على كود:\n@da7lan\nثم استخدم:\n/activate GOLD_XXXXXX",
            reply_markup=main_keyboard()
        )

    msg = await run_report(data, uid)
    await q.message.reply_text(msg[:4090], reply_markup=main_keyboard())

def main():
    nest_asyncio.apply()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("activate", activate_cmd))
    app.add_handler(CommandHandler("gen_codes", gen_codes_cmd))
    app.add_handler(CallbackQueryHandler(handle_buttons))
    print("🤖 Bot Running...")
    app.run_polling()

if __name__ == "__main__":
    main()