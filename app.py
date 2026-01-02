
# --- LOGIN GATE (multi-user; PBKDF2; Streamlit Secrets) ---
import time, base64, hashlib
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
        return hashlib.compare_digest(test, actual)
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
    st.title("🔐 Sign in to Atlantic Giant Genetic Center")
    st.caption("Enter your username or email & password. Need access? Contact the site owner.")

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
DB_PATH = r"C:\\Users\\NEO-1\\Desktop\\Pumpkin Database\\pumpkins.db"
HEAVY_SANITY_LIMIT = 25.0  # tightened: ignore pumpkins > +25% heavy
WORLD_RECORD_LIMIT = 3200.0
MIN_FLOOR_QUALIFIER_CEILING = 3000.0
NAME_REPLACEMENTS = {
    "stelts_2731": "2731.5 Stelts 2024",
    "haist_2741": "2741.5 Haist 2025",
    "wolf_2365": "2365 Wolf 2021",
    "paton_2641": "2641.1 Paton 2024",
    "paton_2819": "2819.3 Paton 2025",
    # Preferred pretty for known canonical IDs (defensive)
    "mendi_2618": "2618 Mendi 2025",
}
# Canonical grower synonyms (aliases -> canonical)
# Ensures "2618 Annabelruben 2025" == "2618 Mendi 2025"
GROWER_SYNONYMS = {
    "annabelruben": "mendi",
    "annabelrubin": "mendi",
    "annabel": "mendi",
    "annabelle": "mendi",
    "annabel-ruben": "mendi",
    "annabel_ruben": "mendi",
    # In case punctuation/space removal produces trailing year in token:
    "annabelruben2025": "mendi",
}
# Tokens to strip anywhere in names (both inside/outside parentheses)
NOISE_TOKENS = [
    r"dmg", r"damaged", r"damage",
    r"adj", r"uow", r"est", r"est\.", r"clone", r"sibb", r"sib",
]

def denoise_text(raw: str) -> str:
    """
    Remove nuisance tokens, collapse whitespace, and normalize number+letter weight suffixes.
    - Strips 2144A/2144B -> 2144 prior to parsing
    - Strips DMG/ADJ/UOW/EST/CLONE/SIBB anywhere (even concatenated)
    """
    s = str(raw)
    # Strip number+single-letter suffix immediately after weight (e.g., 2144A -> 2144)
    s = re.sub(r'(?i)\b(\d{3,4})[a-z]\b', r'\1', s)
    # Remove common nuisance tokens
    for tok in NOISE_TOKENS:
        s = re.sub(rf"(?i)\b{tok}\b", "", s)
        s = re.sub(rf"(?i){tok}", "", s)  # also remove if concatenated (e.g., Patondmg)
    # Remove stray punctuation, keep word chars, spaces, dot, dash
    s = re.sub(r"[^\w\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==================== UTILITIES ====================
def get_seed_identity(s: str):
    """
    Normalize a 'seed' string into (weight, grower, year, id_base, pretty).
    - Strips nuisance tokens anywhere (DMG, ADJ, UOW, EST, CLONE, SIBB).
    - Normalizes number+letter weight suffixes (2144A -> 2144).
    - Canonicalizes grower via synonyms (e.g., Annabelruben -> Mendi).
    - Uses NAME_REPLACEMENTS to map canonical id_base -> pretty.
    """
    raw = str(s).strip()
    if not raw or raw.lower() in ["nan", "open", "unknown", ""]:
        return 0.0, "unknown", 0, "unknown_0", "Unknown"
    s_clean = denoise_text(raw)
    # Extract year and weight
    years = re.findall(r"\b(20\d\d)\b", s_clean)
    year = int(years[-1]) if years else 0
    weights = re.findall(r"\b(\d{1,4}(?:\.\d+)?)\b", s_clean)
    valid_weights = [float(w) for w in weights if int(float(w)) != year]
    weight = valid_weights[0] if valid_weights else 0.0
    if weight > WORLD_RECORD_LIMIT or weight < 100:
        return 0.0, "unknown", 0, "unknown_0", "Unknown"
    # Letters-only grower token (year removed)
    grower_source = s_clean.replace(str(year), "")
    grower = re.sub(r"[^a-zA-Z]", "", grower_source).lower()
    # Strip residual nuisance suffixes (defensive)
    grower = re.sub(r"(dmg|damaged|damage|adj|uow|est|clone|sibb|sib)$", "", grower)
    grower = re.sub(r"(dmg|damaged|damage|adj|uow|est|clone|sibb|sib)", "", grower)
    # Canonicalize via synonyms (cover common variants)
    grower = GROWER_SYNONYMS.get(grower, grower)
    id_base = f"{grower}_{int(weight)}"
    # Preferred pretty mapping (NAME_REPLACEMENTS)
    pretty = NAME_REPLACEMENTS.get(
        id_base,
        f"{weight if weight % 1 != 0 else int(weight)} {grower.capitalize()} {year}",
    )
    # Defensive: force 2618 Mendi 2025 pretty if the identity matches
    if int(weight) == 2618 and grower == "mendi" and year == 2025:
        pretty = "2618 Mendi 2025"
    return weight, grower, year, id_base, pretty

@st.cache_data(ttl=3600)
def load_and_analyze():
    """Load DB, clean and compute metrics used across pages."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT Pumpkin_Name, Weight, Mother_Seed, Father_Seed, Percent_Heavy, OTT, Year_Grown FROM pumpkins",
        conn,
    )
    conn.close()
    # Denoise name fields for consistent matching later
    for col in ["Pumpkin_Name", "Mother_Seed", "Father_Seed"]:
        df[col] = df[col].apply(denoise_text)
    df["W_Num"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["P_Num"] = pd.to_numeric(df["Percent_Heavy"], errors="coerce").fillna(0)
    # Filter unrealistic % Heavy > 25%
    df_clean = df[(df["W_Num"] < WORLD_RECORD_LIMIT) & (df["P_Num"] <= HEAVY_SANITY_LIMIT)].copy()
    seed_db, progeny_map = {}, {}
    # Build seed database + progeny mapping (canonical parents)
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
                progeny_map.setdefault(p_id, []).append({"w": df_clean.at[idx, "W_Num"], "p": df_clean.at[idx, "P_Num"]})
    # Compute base metrics (used by other pages)
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
            if elite
            else data_point["p"]
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
                "_score": score,
                "_id": id_key,
                "_w": data_point["w"],
                "_ott": data_point["ott"],
                "_heavy": data_point["p"],
                "_m": data_point["m"],
                "_f": data_point["f"],
                "_y": data_point["y"],
            }
        )
    res_df = pd.DataFrame(results)
    sorted_names = res_df.sort_values("_w", ascending=False)["NAME"].unique().tolist()
    return res_df, sorted_names, seed_db, df_clean

# ==================== APP INIT ====================
data, all_pumpkins, raw_seed_db, df_raw = load_and_analyze()
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Search & Home"
if "selected_pumpkin" not in st.session_state:
    st.session_state.selected_pumpkin = ""
if "search_key" not in st.session_state:
    st.session_state.search_key = ""

# ==================== GLOBAL CSS ====================
st.markdown(
    """
    <style>
    .dataframe th, .dataframe td { padding: 6px 8px; }
    .dataframe thead th { background: #f5f5f5; }
    /* Tree container and nodes */
    .tree-container { position: relative; border-radius: 8px; background: transparent; overflow: visible; }
    .tree-node { position: absolute; border-radius: 8px; padding: 8px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); background: #ffffff; border: 1px solid #e5e5e5; color: #222; display: flex; flex-direction: column; justify-content: center; }
    .tree-node .label { font-weight: 600; opacity: 0.8; margin-bottom: 4px; }
    .tree-node .name { font-weight: 700; margin-bottom: 4px; }
    .tree-node .line { margin-bottom: 2px; }
    .tree-connector { position: absolute; height: 2px; background: rgba(0,0,0,0.2); }
    /* Search row alignment */
    .search-flex { display: flex; flex-direction: row; align-items: center; gap: 10px; margin-bottom: 10px; }
    .search-flex .clear-btn .stButton > button {
      background: #2b2b2b; color: #ff4242; border: 1px solid #ff4242;
      border-radius: 8px; height: 40px; width: 40px; line-height: 24px; font-size: 20px;
      display: inline-flex; align-items: center; justify-content: center; padding: 0; margin: 0;
    }
    .search-flex .search-box [data-testid="stSelectbox"] { width: 420px; min-width: 320px; max-width: 420px; }
    .search-flex .search-box [data-testid="stSelectbox"] > div:first-child { height: 40px; }
    /* Info card styling */
    .info-card { border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); border-radius: 12px; padding: 14px 16px; margin-top: 10px; }
    .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 24px; align-items: start; }
    .metric { font-weight: 600; margin: 2px 0; }
    .muted { opacity: 0.9; }
    .chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; font-weight: 600; margin: 2px 8px 2px 0; }
    .chip .tag { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 999px; color: #111; }
    .chip.mother { background: rgba(255, 213, 79, 0.20); color: #ffea00; }
    .chip.mother .tag { background: #ffd54f; }
    .chip.father { background: rgba(79, 195, 247, 0.20); color: #80d8ff; }
    .chip.father .tag { background: #4fc3f7; }
    /* Generic table wrapper */
    .tbl-wrap { width: 100%; max-width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); overflow: hidden; }
    /* Top-50 visual layout (shared by Prediction + Heavy Prediction pages) */
    .tbl.top50 thead th {
      position: sticky; top: 0; z-index: 2; text-align: left; padding: 8px 12px;
      background: #1f1f1f; color: #fff; border-bottom: 1px solid rgba(255,255,255,0.18);
      font-weight: 800; letter-spacing: 0.02em;
    }
    .tbl.top50 thead th .hl1 { font-size: 0.95rem; font-weight: 800; }
    .tbl.top50 thead th .hl2 { font-size: 0.80rem; opacity: 0.95; }
    .tbl.top50 thead th .hl3 { font-size: 0.78rem; opacity: 0.85; }
    .tbl.top50 tbody td { text-align: left; padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); }
    .tbl.top50 tbody tr:nth-child(odd) { background: rgba(255,255,255,0.03); }
    .tbl.top50 tbody tr:nth-child(even) { background: rgba(255,255,255,0.06); }
    .tbl.top50 tbody tr:hover { background: rgba(255,255,255,0.11); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== SIDEBAR NAV ====================
st.sidebar.title("🧬 Navigation")
st.session_state.view_mode = st.sidebar.radio(
    "Go to",
