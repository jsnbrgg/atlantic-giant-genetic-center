
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

# Default landing page: Home / Tree (formerly Lineage Tree)
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Home / Tree"
if "selected_pumpkin" not in st.session_state:
    st.session_state.selected_pumpkin = ""
if "search_key" not in st.session_state:
    st.session_state.search_key = ""

# ==================== GLOBAL CSS (unchanged look; horizontal scroll for wide tables) ====================
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

    /* Generic table wrapper — horizontal scroll on phones */
    .tbl-wrap { width: 100%; max-width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.04); overflow-x: auto; overflow-y: hidden; }

    /* Top-50 visual layout (unchanged) */
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

    /* Top nav buttons for all pages */
    .global-topnav .stButton > button {
      height: 40px; border-radius: 8px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== SIDEBAR NAV (Home / Tree first; one-click) ====================
st.sidebar.title("🧬 Navigation")
nav_options = ["Home / Tree", "Progeny Search", "Top 50 Genetic Prediction", "Top 50 Heavy Prediction"]

# Sidebar reflects current view_mode by index, no widget key writes
default_index = nav_options.index(st.session_state.view_mode) if st.session_state.view_mode in nav_options else 0
choice = st.sidebar.radio("Go to", nav_options, index=default_index)

# One-click behavior: update and rerun if changed
if choice != st.session_state.view_mode:
    st.session_state.view_mode = choice
    st.rerun()

# ==================== TOP NAV BAR (buttons across the top, on every page) ====================
def render_top_nav():
    st.markdown('<div class="global-topnav">', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🏠 Home / Tree", use_container_width=True):
            st.session_state.view_mode = "Home / Tree"; st.rerun()
    with b2:
        if st.button("🔍 Progeny Search", use_container_width=True):
            st.session_state.view_mode = "Progeny Search"; st.rerun()
    with b3:
        if st.button("🏆 Top 50 Genetic", use_container_width=True):
            st.session_state.view_mode = "Top 50 Genetic Prediction"; st.rerun()
    with b4:
        if st.button("🛡️ Top 50 Heavy", use_container_width=True):
            st.session_state.view_mode = "Top 50 Heavy Prediction"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# ---------- Helper: generic table (scroll viewport; used by other pages) ----------
def render_pretty_table(df: pd.DataFrame, columns: list[str], height_px: int = 420, table_class: str = ""):
    view = df[columns].copy()
    thead = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in view.columns) + "</tr></thead>"
    rows_html = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in view.columns)
        rows_html.append(f"<tr>{cells}</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
    html = f"""
    <div class="tbl-wrap" style="max-height:{height_px}px;">
      <table class="tbl {table_class}">
        {thead}
        {tbody}
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------- Helper: Three-line header + full height (Top-50 visual layout) ----------
def render_top50_table(df: pd.DataFrame, columns: list[str], height_px: int = 420):
    view = df[columns].copy()
    header_cells = [
        ("RANK", "", ""),
        ("MOTHER SEED NAME", "", ""),
        ("ELITE", "OFFSPRING", ">2000"),
        ("SUPER", "OFFSPRING", ">2500"),
        ("MEGA", "OFFSPRING", ">2700"),
        ("OFFSPRING", "HEAVIER THAN", "MOM"),
        ("HEAVY %", "", ""),
        ("MIN FLOOR", "", ""),
        ("MAX FLOOR", "", ""),
    ]
    def th_cell(title, sub1, sub2):
        return f"<th><div class='hl1'>{title}</div><div class='hl2'>{sub1}</div><div class='hl3'>{sub2}</div></th>"
    thead = "<thead><tr>" + "".join(th_cell(*hc) for hc in header_cells) + "</tr></thead>"
    rows_html = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in view.columns)
        rows_html.append(f"<tr>{cells}</tr>")
    tbody = "<tbody>" + "".join(rows_html) + "</tbody>"
    html = f"""
    <div class="tbl-wrap" style="max-height:none;">
      <table class="tbl top50">
        {thead}
        {tbody}
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==================== PAGE: PROGENY SEARCH (stacked tables; Weight/Year_Grown removed) ====================
if st.session_state.view_mode == "Progeny Search":
    st.title("🔍 Progeny")
    render_top_nav()

    if st.button("✕ Clear Selection"):
        st.session_state.selected_pumpkin = ""; st.rerun()
    selected = st.selectbox(
        "Select Seed to View Progeny (Largest first)",
        options=[""] + all_pumpkins,
        index=(all_pumpkins.index(st.session_state.selected_pumpkin) + 1)
        if st.session_state.selected_pumpkin in all_pumpkins else 0,
    )
    st.session_state.selected_pumpkin = selected
    if st.session_state.selected_pumpkin:
        _, _, _, tid, _ = get_seed_identity(st.session_state.selected_pumpkin)
        p = raw_seed_db.get(tid, {"m": "Unknown", "f": "Unknown"})
        st.subheader(f"Progeny of {st.session_state.selected_pumpkin}")
        st.markdown(f"Mother Seed: {p['m']} \nFather Seed: {p['f']}", unsafe_allow_html=True)
        if st.button("→ View Lineage Tree"):
            st.session_state.view_mode = "Home / Tree"; st.rerun()
        cols_to_show = ["Pumpkin_Name", "Mother_Seed", "Father_Seed"]
        m_kids = df_raw[df_raw["Mother_Seed"] == st.session_state.selected_pumpkin][cols_to_show].sort_values("Pumpkin_Name", ascending=True)
        f_kids = df_raw[df_raw["Father_Seed"] == st.session_state.selected_pumpkin][cols_to_show].sort_values("Pumpkin_Name", ascending=True)
        st.markdown("---")
        st.subheader("Offspring (Used as Mother)")
        st.dataframe(m_kids, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.subheader("Offspring (Used as Pollinator)")
        st.dataframe(f_kids, hide_index=True, use_container_width=True)

# ==================== PAGE: HOME / TREE (formerly Lineage Tree) ====================
elif st.session_state.view_mode == "Home / Tree":
    st.title("🌳 Tree")
    render_top_nav()

    if st.button("✕ Clear Selection"):
        st.session_state.selected_pumpkin = ""; st.rerun()
    selected = st.selectbox(
        "Select Seed for Pedigree (Largest first)",
        options=[""] + all_pumpkins,
        index=(all_pumpkins.index(st.session_state.selected_pumpkin) + 1)
        if st.session_state.selected_pumpkin in all_pumpkins else 0,
    )
    st.session_state.selected_pumpkin = selected

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("### 🛠️ Tree Configuration")
    gens = st.sidebar.slider("Generations", 1, 6, 4)
    st.sidebar.markdown("### 📊 Data to Include")
    inc_w = st.sidebar.checkbox("Weight", value=False)
    inc_heavy = st.sidebar.checkbox("% Heavy", value=True)
    inc_ott = st.sidebar.checkbox("OTT", value=True)
    inc_year = st.sidebar.checkbox("Year Grown", value=False)
    st.sidebar.markdown("### 📐 Layout Controls")
    tw = st.sidebar.slider("Box Width", 100, 300, 190)
    th = st.sidebar.slider("Box Height", 60, 200, 110)
    h_space = st.sidebar.slider("Horizontal Spacing", 20, 140, 60)
    v_start = st.sidebar.slider("Initial Vertical Spread", 100, 600, 240)
    top_pad = st.sidebar.slider("Top Margin", 0, 600, 420)
    label_font_px = st.sidebar.slider("Label/Name Font Size", 10, 32, 16)
    data_font_px = st.sidebar.slider("Data Font Size", 8, 28, 14)

    if st.session_state.selected_pumpkin:
        st.subheader(f"Pedigree: {st.session_state.selected_pumpkin}")
        if st.button("👥 View Progeny"):
            st.session_state.view_mode = "Progeny Search"; st.rerun()

        nodes = []
        def build(name, x, y, step_y, gen, type_label):
            if gen > gens or not name or str(name).strip().lower() == "unknown":
                return
            _, _, _, nid, pretty = get_seed_identity(name)
            d = raw_seed_db.get(nid, {"w": 0, "ott": "N/A", "p": 0, "m": "Unknown", "f": "Unknown", "y": "N/A"})
            nodes.append({"name": pretty, "x": x, "y": y, "label": type_label, "data": d, "gen": gen})
            build(d["m"], x + (tw + h_space), y - step_y, step_y / 2, gen + 1, "Seed")
            build(d["f"], x + (tw + h_space), y + step_y, step_y / 2, gen + 1, "Pollinator")

        build(st.session_state.selected_pumpkin, 20, top_pad, v_start, 1, "Pumpkin")
        if nodes:
            max_right = max(int(n["x"]) + int(tw) for n in nodes)
            max_bottom = max(int(n["y"]) + int(th) for n in nodes)
        else:
            max_right, max_bottom = 0, 0
        pad_x, pad_y = 20, 20
        container_width = max_right + pad_x
        container_height = max_bottom + pad_y
        html = f'<div class="tree-container" style="width:{container_width}px; height:{container_height}px;">'
        def lh(px: int) -> float: return float(f"{max(1.1, min(1.6, px/12)):.2f}")
        for n in nodes:
            base_style = f"left:{int(n['x'])}px; top:{int(n['y'])}px; width:{int(tw)}px; height:{int(th)}px;"
            lines = [
                f'<div class="label" style="font-size:{int(label_font_px)}px; line-height:{lh(label_font_px)};">{n["label"]}</div>',
                f'<div class="name" style="font-size:{int(label_font_px)}px; line-height:{lh(label_font_px)};">{n["name"]}</div>',
            ]
            if inc_w:
                lines.append(f'<div class="line" style="font-size:{int(data_font_px)}px; line-height:{lh(data_font_px)};">Weight: {n["data"]["w"]}</div>')
            if inc_heavy:
                lines.append(f'<div class="line" style="font-size:{int(data_font_px)}px; line-height:{lh(data_font_px)};">% Heavy: {n["data"]["p"]}%</div>')
            if inc_ott:
                lines.append(f'<div class="line" style="font-size:{int(data_font_px)}px; line-height:{lh(data_font_px)};">OTT: {n["data"]["ott"]}</div>')
            if inc_year:
                lines.append(f'<div class="line" style="font-size:{int(data_font_px)}px; line-height:{lh(data_font_px)};">Year: {n["data"]["y"]}</div>')
            content = "\n".join(lines)
            html += f'<div class="tree-node" style="{base_style}">{content}</div>'
            html += f'<div class="tree-connector" style="left:{int(n["x"]) + tw}px; top:{int(n["y"]) + (th // 2)}px; width:{int(h_space)}px;"></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ==================== PAGE: TOP 50 GENETIC PREDICTION ====================
elif st.session_state.view_mode == "Top 50 Genetic Prediction":
    st.title("Genetic Prediction")
    render_top_nav()
    st.caption("This page computes the list independently to keep the Home page fast.")

    df_fast = df_raw.copy()
    for col in ["Pumpkin_Name", "Mother_Seed", "Father_Seed"]:
        df_fast[col] = df_fast[col].apply(denoise_text)
    df_fast["W_Num"] = pd.to_numeric(df_fast["Weight"], errors="coerce")
    df_fast["P_Num"] = pd.to_numeric(df_fast["Percent_Heavy"], errors="coerce").fillna(0)
    df_fast = df_fast[(df_fast["W_Num"] < WORLD_RECORD_LIMIT) & (df_fast["P_Num"] <= HEAVY_SANITY_LIMIT)].copy()

    df_fast["grower"] = df_fast["Pumpkin_Name"].apply(lambda s: get_seed_identity(s)[1])
    grower_pb_map = df_fast.groupby("grower")["W_Num"].max().to_dict()
    kid_pb_set = set()
    for _, r in df_fast.iterrows():
        g = r["grower"]; best = grower_pb_map.get(g, None)
        wv = float(r["W_Num"]) if pd.notnull(r["W_Num"]) else 0.0
        if best is not None and abs(wv - float(best)) < 1e-6:
            kid_pb_set.add(r["Pumpkin_Name"])

    progeny = defaultdict(list)
    for _, row in df_fast.iterrows():
        _, _, _, _, m_pretty = get_seed_identity(row["Mother_Seed"])
        _, _, _, _, f_pretty = get_seed_identity(row["Father_Seed"])
        progeny[m_pretty].append(row)
        progeny[f_pretty].append(row)

    ott_series = pd.to_numeric(df_fast["OTT"], errors="coerce")
    global_ott_med = float(ott_series.dropna().median()) if not ott_series.dropna().empty else 0.0

    def is_pb_for_grower(seed_name: str, w: float) -> bool:
        _, grower, _, _, _ = get_seed_identity(seed_name)
        best = grower_pb_map.get(grower, None)
        return best is not None and abs(float(w) - float(best)) < 1e-6

    def diversity_factor(m_name: str, f_name: str) -> float:
        _, mg, _, mid, _ = get_seed_identity(m_name)
        _, fg, _, fid, _ = get_seed_identity(f_name)
        if mid == fid or mg == fg:
            return 0.95
        if mg != fg and mid != fid and mg != "unknown" and fg != "unknown":
            return 1.03
        return 1.00

    def advanced_score(seed_row: pd.Series) -> float:
        name = seed_row["NAME"]
        w = float(seed_row["_w"]) if pd.notnull(seed_row["_w"]) else 0.0
        heavy_seed = float(seed_row["_heavy"]) if pd.notnull(seed_row["_heavy"]) else 0.0
        ott_raw = seed_row["_ott"]
        try:
            ott = float(ott_raw)
        except Exception:
            ott = np.nan
        own_w_norm = np.clip(min(w, WORLD_RECORD_LIMIT) / 2500.0, 0.6, 1.4)
        ott_norm = np.clip(ott / global_ott_med, 0.85, 1.20) if global_ott_med and not np.isnan(ott) else 1.0
        heavy_norm = 1.0 + max(0.0, heavy_seed) / 100.0 * 0.40
        pb_flag = 1.0 if is_pb_for_grower(name, w) else 0.0

        kids = progeny.get(name, [])
        kcnt = len(kids)
        if kcnt:
            kids_w = np.array([float(pd.to_numeric(r["W_Num"], errors="coerce")) for r in kids], dtype=float)
            kids_p = np.array([float(pd.to_numeric(r["P_Num"], errors="coerce")) for r in kids], dtype=float)
            kids_w_clean = kids_w[~np.isnan(kids_w)]
            c2000 = int(np.sum(kids_w_clean >= 2000))
            c2500 = int(np.sum(kids_w_clean >= 2500))
            c2700 = int(np.sum(kids_w_clean >= 2700))
            heavy_rate = float(np.mean(kids_p > 0.0)) if kids_p.size > 0 else 0.0
            high_heavy_cnt = int(np.sum(kids_p >= 5.0))
            pb_kids = sum(1 for r in kids if r["Pumpkin_Name"] in kid_pb_set)
            pb_kids_rate = pb_kids / kcnt if kcnt > 0 else 0.0
            top_mean = float(np.mean(np.sort(kids_w_clean)[-3:])) if kids_w_clean.size > 0 else 0.0
            heavy_p75 = float(np.percentile(kids_p, 75)) if kids_p.size > 0 else heavy_seed
        else:
            _, _, _, _, m_pretty = get_seed_identity(seed_row["_m"])
            _, _, _, _, f_pretty = get_seed_identity(seed_row["_f"])
            m_kids = progeny.get(m_pretty, []); f_kids = progeny.get(f_pretty, [])
            mf_rows = m_kids + f_kids
            if mf_rows:
                mf_w = np.array([float(pd.to_numeric(r["W_Num"], errors="coerce")) for r in mf_rows], dtype=float)
                mf_p = np.array([float(pd.to_numeric(r["P_Num"], errors="coerce")) for r in mf_rows], dtype=float)
                mf_w_clean = mf_w[~np.isnan(mf_w)]
                c2000 = int(np.sum(mf_w_clean >= 2000))
                c2500 = int(np.sum(mf_w_clean >= 2500))
                c2700 = int(np.sum(mf_w_clean >= 2700))
                heavy_rate = float(np.mean(mf_p > 0.0)) if mf_p.size > 0 else 0.0
                high_heavy_cnt = int(np.sum(mf_p >= 5.0))
                top_mean = float(np.mean(np.sort(mf_w_clean)[-3:])) if mf_w_clean.size > 0 else 0.0
                heavy_p75 = float(np.percentile(mf_p, 75)) if mf_p.size > 0 else heavy_seed
            else:
                c2000 = c2500 = c2700 = 0
                heavy_rate = 0.0
                high_heavy_cnt = 0
                top_mean = 0.0
                heavy_p75 = heavy_seed
            kcnt = 0

        div_factor = diversity_factor(seed_row["_m"], seed_row["_f"]) if seed_row["_m"] and seed_row["_f"] else 1.0
        score_base = own_w_norm * ott_norm * heavy_norm
        offspring_power = 1.0 + 0.18*np.log1p(c2700) + 0.12*np.log1p(c2500) + 0.10*np.log1p(c2000)
        heavy_power = 1.0 + 0.10*heavy_rate + (0.06*(high_heavy_cnt/max(1, kcnt)) if kcnt > 0 else 0.03*heavy_rate)
        pb_power = 1.0 + (0.08*(pb_kids_rate) if kcnt > 0 else 0.0) + (0.05 if pb_flag > 0 else 0.0)
        consistency = 1.0 + 0.06*(heavy_p75/100.0)
        top_mean_factor = 1.0 + min(0.12, (top_mean/2500.0)*0.12)
        novelty = 0.92 if kcnt == 0 else 1.0
        return float(1000.0 * score_base * offspring_power * heavy_power * pb_power * consistency * top_mean_factor * div_factor * novelty)

    advanced = data.copy()
    advanced["GP_SCORE"] = advanced.apply(advanced_score, axis=1)
    top50_adv = advanced.sort_values("GP_SCORE", ascending=False).head(50).copy()
    top50_adv.insert(0, "RANK", [f"#{i+1}" for i in range(len(top50_adv))])
    cols_50 = ["RANK", "NAME", "ELITE", "SUPER", "MEGA", "GAINS", "HEAVY %", "MIN FLOOR", "MAX FLOOR"]
    render_top50_table(top50_adv, cols_50, height_px=520)

# ==================== PAGE: TOP 50 HEAVY PREDICTION ====================
elif st.session_state.view_mode == "Top 50 Heavy Prediction":
    st.title("Heavy Prediction")
    render_top_nav()
    st.caption("Strongest seeds by conservative floor (MIN FLOOR). If fewer than 50 meet the MAX FLOOR threshold, we top up with the next best MIN FLOOR. Table uses the Top-50 visual layout and full height.")

    qualified = data[data["MAX FLOOR"] >= MIN_FLOOR_QUALIFIER_CEILING].copy()
    qualified = qualified.sort_values("MIN FLOOR", ascending=False)
    if len(qualified) < 50:
        remainder = data[~data.index.isin(qualified.index)].copy()
        remainder = remainder.sort_values("MIN FLOOR", ascending=False)
        heavy_pred = pd.concat([qualified, remainder.head(50 - len(qualified))], ignore_index=True)
    else:
        heavy_pred = qualified.head(50).copy()

    heavy_pred.insert(0, "RANK", [f"#{i+1}" for i in range(len(heavy_pred))])
    cols_hp = ["RANK", "NAME", "ELITE", "SUPER", "MEGA", "GAINS", "HEAVY %", "MIN FLOOR", "MAX FLOOR"]
    render_top50_table(heavy_pred, cols_hp, height_px=520)
