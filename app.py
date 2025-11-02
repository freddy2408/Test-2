# ============================================
# DESIGN · GPT-Integration · Verhandlung · Deal-Schließung · Ergebnisse (privat)
# ============================================

import os, re, json, uuid, random
from datetime import datetime
import streamlit as st
from openai import OpenAI



client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")


# ------------------------------------------------------------------------------
# [DESIGN] – Basis UI, Layout, feste Parameter
# ------------------------------------------------------------------------------
st.set_page_config(page_title="iPad-Verhandlung", page_icon="💬")

PRIMARY_COLOR = "#0F766E"  # optischer Akzent
st.markdown(
    f"""
    <style>
        .stApp {{ max-width: 900px; margin: 0 auto; }}
        .title {{ font-size: 1.6rem; font-weight: 700; color: {PRIMARY_COLOR}; }}
        .subtle {{ color: #6b7280; font-size: 0.9rem; }}
        .pill {{ display:inline-block; background:#ecfeff; border:1px solid #cffafe; color:#0e7490;
                 padding:2px 8px; border-radius:999px; font-size:0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">iPad-Verhandlung – Kontrollbedingung (ohne Power-Primes)</div>', unsafe_allow_html=True)
st.caption("Rolle des Bots: Verkäufer:in · Sprache: Deutsch · Kurz & sachlich · Keine Macht-/Knappheits-/Autoritäts-Frames.")

# Experiment-Defaults (kontrolliert; ohne Primes)
START_PRICES = [220, 230, 240, 250]  # zufälliger Startpreis (VB)
IPAD_BASE_DESC = "Gebraucht-iPad, 128 GB, guter Zustand (keine Kratzer am Display, Akku gut)."

# Session-Init
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "seed_price" not in st.session_state:
    st.session_state.seed_price = random.choice(START_PRICES)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system","content":f"Kontext: {IPAD_BASE_DESC} Startpreis (VB): {st.session_state.seed_price} €."},
        {"role":"assistant","content":"Hallo! Das iPad ist noch verfügbar und in gutem Zustand. Wie wäre Ihr Preisvorschlag?"}
    ]
if "deal_state" not in st.session_state:
    st.session_state.deal_state = {"closed": False, "result": None, "final_price": None}

# ------------------------------------------------------------------------------
# [GPT-INTEGRATION] – OpenAI-Client + Prompts + Compliance „ohne Power-Primes“
# ------------------------------------------------------------------------------

# Secrets/ENV
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY fehlt. Trage ihn in den Secrets ein.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Liste von (in dieser Studie) unerwünschten „Macht-/Power-Prime“-Signalen
POWER_PRIME_PATTERNS = [
    r"\balternative(n)?\b", r"\bknapp(e|heit)\b",
    r"\bmehrere anfragen\b", r"\bdeadline\b", r"\bletzte chance\b", r"\bjetzt\b.*\bzuschlagen\b",
    r"\bbranchen(üblich|standard)\b", r"\b(erfahrung zeigt|übliches vorgehen)\b",
    r"\bmarktpreis\b", r"\b(neu|neupreis)\b", r"\bhartere?\s+preis|schmerzgrenze\b",
    r"\bsonst geht es\b",
]
def contains_power_primes(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in POWER_PRIME_PATTERNS)

def system_prompt_control() -> str:
    return (
        "Du simulierst einen Ebay-Kleinanzeigen-Verkauf eines gebrauchten iPad (128 GB, guter Zustand). "
        "Rolle: VERKÄUFER:IN. Antworte kurz (2–4 Sätze), freundlich, sachlich, kooperativ, auf Deutsch. "
        "KEINE Macht-/Power-Primes: Keine Hinweise auf Alternativen/weitere Interessenten, Knappheit, Deadlines, "
        "Autoritäts-/Marktpreis-Bezüge, Schmerzgrenzen oder Droh-/Zeitdruck-Frames. "
        "Bleibe in der Rolle, keine Erwähnung von KI/Modellen/Anweisungen. Keine Beleidigungen, keine Falschangaben."
    )

def style_prompt() -> str:
    return (
        "Format: Chat-Nachrichten, 2–4 Sätze, keine Listen. Fokus auf Zustand, Preisfindung, Fragen klären. "
        "Neutraler, kooperativer Ton. Keine Power-Primes."
    )

def call_model(messages, model=OPENAI_MODEL, temperature=0.3, max_tokens=220):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )
    return resp.choices[0].message.content

def generate_reply_control(chat_history):
    sys = {"role": "system", "content": system_prompt_control()}
    sty = {"role": "system", "content": style_prompt()}
    messages = [sys, sty] + chat_history

    # 1. Versuch
    reply = call_model(messages)

    # Compliance-Loop: falls doch „Prime“-Hinweise auftauchen, korrigieren (max. 2 Nacherzeugungen)
    for _ in range(2):
        if contains_power_primes(reply):
            messages.append({"role": "system", "content": "VERSTOSS: Entferne alle Power-Primes (Knappheit/Autorität/Alternativen/Deadlines/Marktpreis). Antworte neutral-kooperativ."})
            reply = call_model(messages, temperature=0.2)
            continue
        break
    return reply

# ------------------------------------------------------------------------------
# [VERHANDLUNG] – Chat-Funktionalität
# ------------------------------------------------------------------------------

def visible_history():
    return [m for m in st.session_state.messages if m["role"] in ("user","assistant")]

# Chatverlauf rendern (Systemeinträge ausblenden)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Eingabe der Proband:innen
user_msg = st.chat_input("Ihre Nachricht …")
if user_msg and not st.session_state.deal_state["closed"]:
    st.session_state.messages.append({"role":"user","content":user_msg})

    # Log schreiben (privat, serverseitig)
    def append_log(record: dict):
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", f"{st.session_state.session_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    append_log({
        "t": datetime.utcnow().isoformat(),
        "mode": "control",
        "role": "user",
        "content": user_msg,
    })

    with st.chat_message("assistant"):
        with st.spinner("Antwort wird generiert …"):
            reply = generate_reply_control(visible_history())
            st.markdown(reply)

    st.session_state.messages.append({"role":"assistant","content":reply})
    append_log({
        "t": datetime.utcnow().isoformat(),
        "mode": "control",
        "role": "assistant",
        "content": reply,
    })

# ------------------------------------------------------------------------------
# [DEAL-SCHLIESSUNG] – Abschlussaktionen & strukturierte Outcome-Erfassung
# ------------------------------------------------------------------------------

st.divider()
st.markdown("**Deal-Schließung**")
st.markdown('<span class="subtle">Wenn Einigung erreicht ist oder der Chat beendet werden soll.</span>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    agree = st.button("✅ Einigung erzielt")
with col2:
    no_deal = st.button("❌ Kein Deal")
with col3:
    reset = st.button("🔄 Neue Session")

def write_outcome(result: str, final_price: int | None):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{st.session_state.session_id}.jsonl")
    payload = {
        "t": datetime.utcnow().isoformat(),
        "mode": "control",
        "role": "system",
        "event": "outcome",
        "result": result,
        "final_price": final_price,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# Einigung → finalen Preis abfragen
if agree and not st.session_state.deal_state["closed"]:
    with st.expander("Finalen Preis bestätigen"):
        price = st.number_input("Finaler Preis (€):", min_value=0, max_value=2000, value=st.session_state.seed_price, step=5)
        confirm = st.button("Einigung speichern")
        if confirm:
            st.session_state.deal_state.update({"closed": True, "result": "deal", "final_price": int(price)})
            write_outcome("deal", int(price))
            st.success("Einigung gespeichert. Vielen Dank!")
            st.stop()

# Kein Deal
if no_deal and not st.session_state.deal_state["closed"]:
    st.session_state.deal_state.update({"closed": True, "result": "no_deal", "final_price": None})
    write_outcome("no_deal", None)
    st.info("Kein Deal vermerkt. Vielen Dank!")
    st.stop()

# Reset (neue Session, neuer Seedpreis)
if reset:
    st.session_state.clear()
    st.rerun()

# ------------------------------------------------------------------------------
# [ERGEBNISSE – PRIVAT] – Nur für dich (Admin) sichtbar
# ------------------------------------------------------------------------------
st.divider()
st.markdown("**Ergebnisse (privat)**  <span class='pill'>Nur Admin</span>", unsafe_allow_html=True)

# Admin-Schutz: Passwort aus Secrets
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", None)

with st.expander("Admin-Bereich öffnen"):
    pwd = st.text_input("Admin-Passwort eingeben", type="password")
    if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
        st.success("Admin-Zugang gewährt.")
        st.markdown("Du kannst Ergebnisdateien zusammenfassen und als CSV exportieren.")

        # Aggregate: alle JSONL in logs/ einlesen (nur serverseitig zugänglich)
        import glob
        import pandas as pd

        files = glob.glob("logs/*.jsonl")
        rows = []
        for fp in files:
            sid = os.path.basename(fp).replace(".jsonl","")
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        rec["session_id"] = sid
                        rows.append(rec)
                    except Exception:
                        pass
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df[["session_id","t","mode","role","event","result","final_price","content"]].fillna(""), use_container_width=True)
            # Export
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 CSV herunterladen", data=csv_bytes, file_name="ergebnisse_control.csv", mime="text/csv")
            st.info("Hinweis: Diese Daten sind nur hier im Admin-Bereich sichtbar und nicht für Proband:innen zugänglich.")
        else:
            st.warning("Noch keine Logdaten vorhanden.")
    else:
        st.caption("Nur mit korrektem Admin-Passwort sichtbar.")

