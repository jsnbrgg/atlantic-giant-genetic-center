
# --- LOGIN GATE (multi-user; PBKDF2; Streamlit Secrets) ---
import time, base64, hashlib, hmac
import streamlit as st

# Helper: verify PBKDF2-HMAC (sha256)
def verify_pbkdf2(password: str, stored: str) -> bool:
    """
    stored format: pbkdf2_sha256$<iterations>$<base64salt>$<base64hash>
    """
    try:
        algo, iter_s, salt_b64, hash_b64 = stored.split("$")
        assert algo == "pbkdf2_sha256"
        iters = int(iter_s)
        salt = base64.b64decode(salt_b64.encode())
        actual = base64.b64decode(hash_b64.encode())
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(test, actual)
    except Exception:
        return False

# Read config & users from secrets
AUTH = st.secrets.get("auth", {})
USERS = st.secrets.get("users", {})  # dict: {username: {email, password_hash, role}}

COOKIE_NAME = AUTH.get("cookie_name", "ag-genetics-auth")
COOKIE_TTL_DAYS = int(AUTH.get("cookie_ttl_days", 30))

# Session bootstrap
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None  # dict: {username, email, role}

def logout():
    st.session_state.auth_user = None
    # Force a clean rerun
    st.experimental_set_query_params()
    st.rerun()

def login_view():
    # Title (cleaned)
    st.title("🔐 Atlantic Giant Genetic Center")
    st.caption("Please enter your login details below.")

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        username = st.text_input("Username or Email", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        go = st.button("Sign in", type="primary")

        if go:
            typed = (username or "").strip()

            # Try direct username match first
            user = USERS.get(typed)
            actual_uname = typed

            # If not found, try email match across all users (case-insensitive)
            if not user:
                for uname, uinfo in USERS.items():
                    if isinstance(uinfo, dict) and uinfo.get("email", "").strip().lower() == typed.lower():
                        user = uinfo
                        actual_uname = uname
                        break

            # If still not found, show error
            if not user:
                st.error("Unknown username or email")
                return

            # Verify password against PBKDF2 hash
            ok = verify_pbkdf2(password, user.get("password_hash", ""))
            if not ok:
                st.error("Incorrect password")
                return

            # Success → set session and rerun
            st.session_state.auth_user = {
                "username": actual_uname,
                "email": user.get("email", ""),
                "role": user.get("role", "viewer"),
                "ts": int(time.time()),
            }
            st.success(f"Welcome, {actual_uname}!")
            st.rerun()

    with col2:
        st.info("Tip: Keep your access code private.\n\nIf you forgot your password, contact support.")

def auth_bar():
    u = st.session_state.auth_user
    if not u:
        return
    left, right = st.columns([0.8, 0.2])
    with left:
        st.write(f"**Signed in:** {u['username']} · *{u.get('role','viewer').capitalize()}*")
    with right:
        st.button("Log out", on_click=logout)

# Gate: if not logged in, show login and STOP
if not st.session_state.auth_user:
    login_view()
    st.stop()

# If here → logged in
auth_bar()
# --- END LOGIN GATE ---

# app.py
import os
import re
import sqlite3
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Atlantic Giant Genetic Center", layout="wide")

# ==================== SETTINGS & CONSTANTS ====================
DB_PATH = "pumpkins.db"

HEAVY_SANITY_LIMIT = 25.0
WORLD_RECORD_LIMIT = 3200.0
MIN_FLOOR_QUALIFIER_CEILING = 3000.0
NAME_REPLACEMENTS = {
    "stelts_2731": "2731.5 Stelts 2024",
    "haist_2741": "2741.5 Haist 2025",
    "wolf_2365": "2365 Wolf 2021",
    "paton_2641": "2641.1 Paton 2024",
    "paton_2819": "2819.3 Paton 2025",
    "mendi_2618": "2618 Mendi 2025",
}
GROWER_SYNONYMS = {
    "annabelruben": "mendi",
    "annabelrubin": "mendi",
    "annabel": "mendi",
    "annabelle": "mendi",
    "annabel-ruben": "mendi",
    "annabel_ruben": "mendi",
    "annabelruben2025": "mendi",
}
NOISE_TOKENS = [
    r"dmg", r"damaged", r"damage",
    r"adj", r"uow", r"est", r"est\.", r"clone", r"sibb", r"sib",
]

def denoise_text(raw: str) -> str:
    s = str(raw)
    s = re.sub(r'(?i)\b(\d{3,4})[a-z]\b', r'\1', s)
    for tok in NOISE_TOKENS:
        s = re.sub(rf"(?i)\b{tok}\b", "", s)
        s = re.sub(rf"(?i){tok}", "", s)
    s = re.sub(r"[^\w\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_seed_identity(s: str):
    raw = str(s).strip()
    if not raw or raw.lower() in ["nan", "open", "unknown", ""]:
        return 0.0, "unknown", 0, "unknown_0", "Unknown"
    s_clean = denoise_text(raw)
    years = re.findall(r"\b(20\d\d)\b", s_clean)
    year = int(years[-1]) if years else 0
    weights = re.findall(r"\b(\d{1,4}(?:\.\d+)?)\b", s_clean)
    valid_weights = [float(w) for w in weights if int(float(w)) != year]
    weight = valid_weights[0] if valid_weights else 0.0
    if weight > WORLD_RECORD_LIMIT or weight < 100:
        return 0.0, "unknown", 0, "unknown_0", "Unknown"
    grower_source = s_clean.replace(str(year), "")
    grower = re.sub(r"[^a-zA-Z]", "", grower_source).lower()
    grower = re.sub(r"(dmg|damaged|damage|adj|uow|est|clone|sibb|sib)$", "", grower)
    grower = re.sub(r"(dmg|damaged|damage|adj|uow|est|clone|sibb|sib)", "", grower)
    grower = GROWER_SYNONYMS.get(grower, grower)
    id_base = f"{grower}_{int(weight)}"
    pretty = NAME_REPLACEMENTS.get(
        id_base,
        f"{weight if weight % 1 != 0 else int(weight)} {grower.capitalize()} {year}",
    )
    if int(weight) == 2618 and grower == "mendi" and year == 2025:
        pretty = "2618 Mendi 2025"
    return weight, grower, year, id_base, pretty

@st.cache_data(ttl=3600)
def load_and_analyze():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT Pumpkin_Name, Weight, Mother_Seed, Father_Seed, Percent_Heavy, OTT, Year_Grown FROM pumpkins",
        conn,
    )
    conn.close()
    for col in ["Pumpkin_Name", "Mother_Seed", "Father_Seed"]:
        df[col] = df[col].apply(denoise_text)
    df["W_Num"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["P_Num"] = pd.to_numeric(df["Percent_Heavy"], errors="coerce").fillna(0)
    df_clean = df[(df["W_Num"] < WORLD_RECORD_LIMIT) & (df["P_Num"] <= HEAVY_SANITY_LIMIT)].copy()

    seed_db, progeny_map = {}, {}
    for idx, row in df_clean.iterrows():
        w, _, _, id_key, pretty = get_seed_identity(row["Pumpkin_Name"])
        if id_key == "unknown_0":
            continue
        seed_db[id_key] = {
            "w": w,
            "pretty": pretty,
            "m": row["Mother_Seed"],
            "f": row["Father_Seed"],
            "p": row["P_Num"],
            "ott": row.get("OTT", "N/A"),
            "y": row.get("Year_Grown", "N/A"),
        }
        for p_raw in [row["Mother_Seed"], row["Father_Seed"]]:
            _, _, _, p_id, _ = get_seed_identity(p_raw)
            if p_id != "unknown_0":
                progeny_map.setdefault(p_id, []).append({
                    "w": df_clean.at[idx, "W_Num"], "p": df_clean.at[idx, "P_Num"]
                })

    results = []
    for id_key, data_point in seed_db.items():
        kids = progeny_map.get(id_key, [])
        elite = [k["w"] for k in kids if k["w"] >= 2000]
        p_base = np.mean(elite) if elite else data_point["w"]
        bonus = (
            1.0
            + (np.log1p(len([k for k in kids if k["w"] >= 2700])) * 0.28)
            + (np.log1p(len([k for k in kids if k["w"] >= 2500])) * 0.15)
            + (np.log1p(len(elite)) * 0.12)
        )
        heavy_p = (
            np.percentile([k["p"] for k in kids if k["w"] >= 2000], 75)
            if elite else data_point["p"]
        )
        score = (p_base * bonus) * (1 + (heavy_p / 100))
        max_v = max([k["w"] for k in kids] + [data_point["w"]]) if kids else data_point["w"]
        results.append(
            {
                "NAME": data_point["pretty"],
                "ELITE": len(elite),
                "SUPER": len([k for k in kids if k["w"] >= 2500]),
                "MEGA": len([k for k in kids if k["w"] >= 2700]),
                "GAINS": len([k for k in kids if k["w"] > data_point["w"]]),
                "HEAVY %": f"{heavy_p:+.2f}%",
                "MIN FLOOR": round(p_base * (1 + (heavy_p / 100)), 1),
                "MAX FLOOR": round((max_v * 1.05) * (1 + (heavy_p / 100)), 1),
