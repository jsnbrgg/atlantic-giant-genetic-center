
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
    st.title("ðŸ” Atlantic Giant Genetic Center")
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

            # Success â†’ set session and rerun
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
        st.write(f"**Signed in:** {u['username']} Â· *{u.get('role','viewer').capitalize()}*")
    with right:
        st.button("Log out", on_click=logout)

# Gate: if not logged in, show login and STOP
if not st.session_state.auth_user:
    login_view()
    st.stop()

# If here â†’ logged in
auth_bar()
# --- END LOGIN GATE ---

# app.py
import os
import re
import sqlite3
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
    """Load DB, clean and compute metrics used across pages."""
    import sqlite3
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
    # Compute base metrics (used by pages)
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

# Default landing page: Home / Tree
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

    /* Generic table wrapper â€” horizontal scroll on phones */
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
st.sidebar.title("ðŸ§¬ Navigation")
nav_options = ["Home / Tree", "Progeny Search", "Genetic Potential", "Heavy Potential"]

default_index = nav_options.index(st.session_state.view_mode) if st.session_state.view_mode in nav_options else 0
choice = st.sidebar.radio("Go to", nav_options, index=default_index)
if choice != st.session_state.view_mode:
    st.session_state.view_mode = choice
    st.rerun()

# ==================== TOP NAV BAR (buttons across the top, on every page) ====================
def render_top_nav():
    st.markdown('<div class="global-topnav">', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("ðŸ  Home / Tree", use_container_width=True, key="nav_btn_home"):
            st.session_state.view_mode = "Home / Tree"; st.rerun()
    with b2:
        if st.button("ðŸ” Progeny Search", use_container_width=True, key="nav_btn_prog"):
            st.session_state.view_mode = "Progeny Search"; st.rerun()
    with b3:
        if st.button("ðŸ† Genetic Potential", use_container_width=True, key="nav_btn_gen"):
            st.session_state.view_mode = "Genetic Potential"; st.rerun()
    with b4:
        if st.button("ðŸ›¡ï¸ Heavy Potential", use_container_width=True, key="nav_btn_heavy"):
            st.session_state.view_mode = "Heavy Potential"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# ---------- Helper: generic table (scroll viewport; used by pages) ----------
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

# ==================== PAGE: PROGENY SEARCH (unchanged logic/layout) ====================
if st.session_state.view_mode == "Progeny Search":
    st.title("ðŸ” Progeny")
    render_top_nav()

    if st.button("âœ• Clear Selection"):
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
        st.markdown(f"Mother Seed: {p['m']}  \nFather Seed: {p['f']}", unsafe_allow_html=True)
        if st.button("â†’ View Lineage Tree"):
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
    st.title("ðŸŒ³ Tree")
    render_top_nav()

    if st.button("âœ• Clear Selection"):
        st.session_state.selected_pumpkin = ""; st.rerun()
    selected = st.selectbox(
        "Select Seed for Pedigree (Largest first)",
        options=[""] + all_pumpkins,
        index=(all_pumpkins.index(st.session_state.selected_pumpkin) + 1)
        if st.session_state.selected_pumpkin in all_pumpkins else 0,
    )
    st.session_state.selected_pumpkin = selected

    # --- SIDEBAR CONTROLS (existing) ---
    st.sidebar.markdown("### ðŸ› ï¸ Tree Configuration")
    gens = st.sidebar.slider("Generations", 1, 6, 4)
    st.sidebar.markdown("### ðŸ“Š Data to Include")
    inc_w = st.sidebar.checkbox("Weight", value=False)
    inc_heavy = st.sidebar.checkbox("% Heavy", value=True)
    inc_ott = st.sidebar.checkbox("OTT", value=True)
    inc_year = st.sidebar.checkbox("Year Grown", value=False)
    st.sidebar.markdown("### ðŸ“ Layout Controls")
    tw = st.sidebar.slider("Box Width", 100, 300, 190)
    th = st.sidebar.slider("Box Height", 60, 200, 110)
    h_space = st.sidebar.slider("Horizontal Spacing", 20, 140, 60)
    v_start = st.sidebar.slider("Initial Vertical Spread", 100, 600, 240)
    top_pad = st.sidebar.slider("Top Margin", 0, 600, 420)
    label_font_px = st.sidebar.slider("Label/Name Font Size", 10, 32, 16)
    data_font_px = st.sidebar.slider("Data Font Size", 8, 28, 14)

    if st.session_state.selected_pumpkin:
        st.subheader(f"Pedigree: {st.session_state.selected_pumpkin}")
        if st.button("ðŸ‘¥ View Progeny"):
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

# ==================== PAGE: GENETIC POTENTIAL (with sidebar controls + Reset) ====================
elif st.session_state.view_mode == "Genetic Potential":
    st.title("Genetic Potential")
    render_top_nav()
    st.caption("Tune the model inputs on the sidebar. Defaults match your current configuration.")

    # --- Defaults & reset handling (scoped to Genetic Potential) ---
    gen_defaults = {
        "gen_sr_world_record": int(WORLD_RECORD_LIMIT),
        "gen_sr_heavy_limit": int(HEAVY_SANITY_LIMIT),
        "gen_th_elite": 2000,
        "gen_th_super": 2500,
        "gen_th_mega": 2700,
        "gen_base_weight": 2500,
        "gen_clamp_min": 0.60,
        "gen_clamp_max": 1.40,
        "gen_use_ott_norm": True,
        "gen_ott_min_ratio": 0.85,
        "gen_ott_max_ratio": 1.20,
        "gen_coef_mega": 0.18,
        "gen_coef_super": 0.12,
        "gen_coef_elite": 0.10,
        "gen_use_heavy_power": True,
        "gen_coef_heavy_rate": 0.10,
        "gen_coef_heavy_count": 0.06,
        "gen_coef_heavy_fallback": 0.03,
        "gen_use_pb_power": True,
        "gen_coef_pb_kids": 0.08,
        "gen_bonus_pb_flag": 0.05,
        "gen_use_consistency": True,
        "gen_coef_consistency": 0.06,
        "gen_use_top_mean": True,
        "gen_top_mean_max_gain": 0.12,
        "gen_top_mean_base": 2500,
        "gen_name_filter": "",
        "gen_top_n_choice": 10,  # default to 10
    }
    # initialize once
    if "gen_initialized" not in st.session_state:
        st.session_state.gen_initialized = True
        for k, v in gen_defaults.items():
            st.session_state.setdefault(k, v)

    # === BASE data ===
    df_fast = df_raw.copy()
    for col in ["Pumpkin_Name", "Mother_Seed", "Father_Seed"]:
        df_fast[col] = df_fast[col].apply(denoise_text)
    df_fast["W_Num"] = pd.to_numeric(df_fast["Weight"], errors="coerce")
    df_fast["P_Num"] = pd.to_numeric(df_fast["Percent_Heavy"], errors="coerce").fillna(0)

    # === SIDEBAR widgets (bound to keys) ===
    st.sidebar.markdown("### âš™ï¸ Genetic Potential Options")

    sr_world_record = st.sidebar.slider("World record limit (lbs)", 2500, 4000, st.session_state.gen_sr_world_record, step=50, key="gen_sr_world_record")
    sr_heavy_limit = st.sidebar.slider("Max allowed % Heavy (sanity filter)", 0, 40, st.session_state.gen_sr_heavy_limit, step=1, key="gen_sr_heavy_limit")
    # Apply filters with chosen limits
    df_fast = df_fast[(df_fast["W_Num"] < sr_world_record) & (df_fast["P_Num"] <= sr_heavy_limit)].copy()

    st.sidebar.markdown("#### Offspring thresholds")
    th_elite = st.sidebar.slider("Elite threshold (lbs)", 1500, 3000, st.session_state.gen_th_elite, step=50, key="gen_th_elite")
    th_super = st.sidebar.slider("Super threshold (lbs)", 2000, 3200, st.session_state.gen_th_super, step=50, key="gen_th_super")
    th_mega  = st.sidebar.slider("Mega threshold (lbs)", 2300, 3400, st.session_state.gen_th_mega, step=50, key="gen_th_mega")

    st.sidebar.markdown("#### Weight normalization")
    base_weight = st.sidebar.slider("Normalization base (lbs)", 2000, 3000, st.session_state.gen_base_weight, step=50, key="gen_base_weight")
    clamp_min   = st.sidebar.slider("Min clamp", 0.3, 1.0, st.session_state.gen_clamp_min, 0.01, key="gen_clamp_min")
    clamp_max   = st.sidebar.slider("Max clamp", 1.0, 2.0, st.session_state.gen_clamp_max, 0.01, key="gen_clamp_max")

    st.sidebar.markdown("#### OTT normalization")
    use_ott_norm = st.sidebar.checkbox("Use OTT normalization", value=st.session_state.gen_use_ott_norm, key="gen_use_ott_norm")
    ott_min_ratio = st.sidebar.slider("OTT ratio lower clamp", 0.5, 1.0, st.session_state.gen_ott_min_ratio, 0.01, key="gen_ott_min_ratio")
    ott_max_ratio = st.sidebar.slider("OTT ratio upper clamp", 1.0, 1.8, st.session_state.gen_ott_max_ratio, 0.01, key="gen_ott_max_ratio")

    st.sidebar.markdown("#### Offspring power weights")
    coef_mega  = st.sidebar.slider("Weight for Mega count (log1p * coef)", 0.00, 0.40, st.session_state.gen_coef_mega, 0.01, key="gen_coef_mega")
    coef_super = st.sidebar.slider("Weight for Super count (log1p * coef)", 0.00, 0.30, st.session_state.gen_coef_super, 0.01, key="gen_coef_super")
    coef_elite = st.sidebar.slider("Weight for Elite count (log1p * coef)", 0.00, 0.30, st.session_state.gen_coef_elite, 0.01, key="gen_coef_elite")

    st.sidebar.markdown("#### Heavy power")
    use_heavy_power = st.sidebar.checkbox("Use heavy-rate influence", value=st.session_state.gen_use_heavy_power, key="gen_use_heavy_power")
    coef_heavy_rate   = st.sidebar.slider("Weight for heavy_rate", 0.00, 0.20, st.session_state.gen_coef_heavy_rate, 0.01, key="gen_coef_heavy_rate")
    coef_heavy_count  = st.sidebar.slider("Weight for high heavy count", 0.00, 0.20, st.session_state.gen_coef_heavy_count, 0.01, key="gen_coef_heavy_count")
    coef_heavy_fallback = st.sidebar.slider("Fallback weight (no kids â†’ heavy_rate)", 0.00, 0.20, st.session_state.gen_coef_heavy_fallback, 0.01, key="gen_coef_heavy_fallback")

    st.sidebar.markdown("#### PB power")
    use_pb_power = st.sidebar.checkbox("Use PB influence", value=st.session_state.gen_use_pb_power, key="gen_use_pb_power")
    coef_pb_kids = st.sidebar.slider("Weight for PB kids rate", 0.00, 0.20, st.session_state.gen_coef_pb_kids, 0.01, key="gen_coef_pb_kids")
    bonus_pb_flag = st.sidebar.slider("Bonus if seed is grower PB", 0.00, 0.20, st.session_state.gen_bonus_pb_flag, 0.01, key="gen_bonus_pb_flag")

    st.sidebar.markdown("#### Consistency & top-mean factors")
    use_consistency = st.sidebar.checkbox("Use consistency factor (p75 heavy)", value=st.session_state.gen_use_consistency, key="gen_use_consistency")
    coef_consistency = st.sidebar.slider("Gain per p75 (% / 100)", 0.00, 0.15, st.session_state.gen_coef_consistency, 0.01, key="gen_coef_consistency")
    use_top_mean = st.sidebar.checkbox("Use top-mean factor", value=st.session_state.gen_use_top_mean, key="gen_use_top_mean")
    top_mean_max_gain = st.sidebar.slider("Max gain from top-mean", 0.00, 0.30, st.session_state.gen_top_mean_max_gain, 0.01, key="gen_top_mean_max_gain")
    top_mean_base = st.sidebar.slider("Top-mean base (lbs)", 2000, 3000, st.session_state.gen_top_mean_base, 50, key="gen_top_mean_base")

    st.sidebar.markdown("#### Display")
    name_filter = st.sidebar.text_input("Filter by seed name (contains)", st.session_state.gen_name_filter, key="gen_name_filter")
    top_n_choice = st.sidebar.selectbox("Results to show", [10, 20, 30, 40, 50], index=[10,20,30,40,50].index(st.session_state.gen_top_n_choice), key="gen_top_n_choice")

    # Reset button
    if st.sidebar.button("Reset to defaults", key="gen_reset_btn"):
        for k, v in gen_defaults.items():
            st.session_state[k] = v
        st.rerun()

    # === Derive helper sets (unchanged base) ===
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

    # OTT baseline
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

    # === Advanced score using sidebar controls ===
    def advanced_score(seed_row: pd.Series) -> float:
        name = seed_row["NAME"]  # canonical pretty
        w = float(seed_row["_w"]) if pd.notnull(seed_row["_w"]) else 0.0
        heavy_seed = float(seed_row["_heavy"]) if pd.notnull(seed_row["_heavy"]) else 0.0
        ott_raw = seed_row["_ott"]
        try:
            ott = float(ott_raw)
        except Exception:
            ott = np.nan

        # Base components (own traits)
        own_w_norm = np.clip(min(w, sr_world_record) / float(base_weight), clamp_min, clamp_max)
        if use_ott_norm and global_ott_med and not np.isnan(ott):
            ott_norm = np.clip(ott / global_ott_med, float(st.session_state.gen_ott_min_ratio), float(st.session_state.gen_ott_max_ratio))
        else:
            ott_norm = 1.0
        heavy_norm = 1.0 + max(0.0, heavy_seed) / 100.0 * 0.40  # keep original seed-heavy contribution

        pb_flag = 1.0 if is_pb_for_grower(name, w) else 0.0

        # Offspring features
        kids = progeny.get(name, [])
        kcnt = len(kids)

        if kcnt:
            kids_w = np.array([float(pd.to_numeric(r["W_Num"], errors="coerce")) for r in kids], dtype=float)
            kids_p = np.array([float(pd.to_numeric(r["P_Num"], errors="coerce")) for r in kids], dtype=float)
            kids_w_clean = kids_w[~np.isnan(kids_w)]

            c_elite = int(np.sum(kids_w_clean >= float(th_elite)))
            c_super = int(np.sum(kids_w_clean >= float(th_super)))
            c_mega  = int(np.sum(kids_w_clean >= float(th_mega)))

            heavy_rate = float(np.mean(kids_p > 0.0)) if kids_p.size > 0 else 0.0
            high_heavy_cnt = int(np.sum(kids_p >= 5.0))
            pb_kids = sum(1 for r in kids if r["Pumpkin_Name"] in kid_pb_set)
            pb_kids_rate = pb_kids / kcnt if kcnt > 0 else 0.0
            top_mean = float(np.mean(np.sort(kids_w_clean)[-3:])) if kids_w_clean.size > 0 else 0.0
            heavy_p75 = float(np.percentile(kids_p, 75)) if kids_p.size > 0 else heavy_seed
        else:
            # Parent-based estimation when no kids yet
            _, _, _, _, m_pretty = get_seed_identity(seed_row["_m"])
            _, _, _, _, f_pretty = get_seed_identity(seed_row["_f"])
            m_kids = progeny.get(m_pretty, [])
            f_kids = progeny.get(f_pretty, [])
            mf_rows = m_kids + f_kids
            if mf_rows:
                mf_w = np.array([float(pd.to_numeric(r["W_Num"], errors="coerce")) for r in mf_rows], dtype=float)
                mf_p = np.array([float(pd.to_numeric(r["P_Num"], errors="coerce")) for r in mf_rows], dtype=float)
                mf_w_clean = mf_w[~np.isnan(mf_w)]
                c_elite = int(np.sum(mf_w_clean >= float(th_elite)))
                c_super = int(np.sum(mf_w_clean >= float(th_super)))
                c_mega  = int(np.sum(mf_w_clean >= float(th_mega)))
                heavy_rate = float(np.mean(mf_p > 0.0)) if mf_p.size > 0 else 0.0
                high_heavy_cnt = int(np.sum(mf_p >= 5.0))
                top_mean = float(np.mean(np.sort(mf_w_clean)[-3:])) if mf_w_clean.size > 0 else 0.0
                heavy_p75 = float(np.percentile(mf_p, 75)) if mf_p.size > 0 else heavy_seed
            else:
                c_elite = c_super = c_mega = 0
                heavy_rate = 0.0
                high_heavy_cnt = 0
                top_mean = 0.0
                heavy_p75 = heavy_seed
            kcnt = 0

        # Assemble factors
        score_base = own_w_norm * ott_norm * heavy_norm
        offspring_power = 1.0 + coef_mega*np.log1p(c_mega) + coef_super*np.log1p(c_super) + coef_elite*np.log1p(c_elite)

        if use_heavy_power:
            heavy_power = 1.0 + coef_heavy_rate*heavy_rate + (
                coef_heavy_count*(high_heavy_cnt/max(1, kcnt)) if kcnt > 0 else coef_heavy_fallback*heavy_rate
            )
        else:
            heavy_power = 1.0

        if use_pb_power:
            pb_kids_rate_val = (pb_kids_rate if kcnt > 0 else 0.0)
            pb_power = 1.0 + (coef_pb_kids*pb_kids_rate_val) + (bonus_pb_flag if pb_flag > 0 else 0.0)
        else:
            pb_power = 1.0

        if use_consistency:
            consistency = 1.0 + (st.session_state.gen_coef_consistency)*(heavy_p75/100.0)
        else:
            consistency = 1.0

        if use_top_mean:
            top_mean_factor = 1.0 + min(st.session_state.gen_top_mean_max_gain, (top_mean/float(st.session_state.gen_top_mean_base))*st.session_state.gen_top_mean_max_gain)
        else:
            top_mean_factor = 1.0

        div_factor = diversity_factor(seed_row["_m"], seed_row["_f"]) if (seed_row["_m"] and seed_row["_f"]) else 1.0
        novelty = 0.92 if kcnt == 0 else 1.0

        return float(1000.0 * score_base * offspring_power * heavy_power * pb_power * consistency * top_mean_factor * div_factor * novelty)

    # === Compute advanced score and render ===
    advanced = data.copy()
    advanced["GP_SCORE"] = advanced.apply(advanced_score, axis=1)

    # Optional name filter (contains)
    if name_filter.strip():
        advanced = advanced[advanced["NAME"].str.contains(name_filter.strip(), case=False, na=False)]

    topN = advanced.sort_values("GP_SCORE", ascending=False).head(int(top_n_choice)).copy()
    topN.insert(0, "RANK", [f"#{i+1}" for i in range(len(topN))])

    cols_50 = ["RANK", "NAME", "ELITE", "SUPER", "MEGA", "GAINS", "HEAVY %", "MIN FLOOR", "MAX FLOOR"]
    render_top50_table(topN, cols_50, height_px=520)

# ==================== PAGE: HEAVY POTENTIAL (with sidebar controls + Reset) ====================
elif st.session_state.view_mode == "Heavy Potential":
    st.title("Heavy Potential")
    render_top_nav()
    st.caption("Tune the heavy-page filters in the sidebar. Defaults match your current configuration.")

    # --- Defaults & reset handling (scoped to Heavy Potential) ---
    heavy_defaults = {
        "heavy_name_filter": "",
        "heavy_min_elite": 0,
        "heavy_qualifier": int(MIN_FLOOR_QUALIFIER_CEILING),
        "heavy_sort_by": "MIN FLOOR",      # options: MIN FLOOR / MAX FLOOR
        "heavy_use_topup": True,
        "heavy_top_n": 10,                 # default to 10
    }
    if "heavy_initialized" not in st.session_state:
        st.session_state.heavy_initialized = True
        for k, v in heavy_defaults.items():
            st.session_state.setdefault(k, v)

    # Sidebar controls
    st.sidebar.markdown("### âš™ï¸ Heavy Potential Options")
    name_filter_h = st.sidebar.text_input("Filter by seed name (contains)", st.session_state.heavy_name_filter, key="heavy_name_filter")
    min_elite = st.sidebar.slider("Minimum ELITE count", 0, 10, st.session_state.heavy_min_elite, key="heavy_min_elite")
    qualifier = st.sidebar.slider("Qualifier threshold (MAX FLOOR, lbs)", 2000, 3500, st.session_state.heavy_qualifier, step=50, key="heavy_qualifier")
    sort_by = st.sidebar.radio("Sort by metric", ["MIN FLOOR", "MAX FLOOR"], index=["MIN FLOOR","MAX FLOOR"].index(st.session_state.heavy_sort_by), key="heavy_sort_by")
    use_topup = st.sidebar.checkbox("Top-up results if fewer than Top N meet the threshold", value=st.session_state.heavy_use_topup, key="heavy_use_topup")
    top_n_heavy = st.sidebar.selectbox("Results to show", [10, 20, 30], index=[10,20,30].index(st.session_state.heavy_top_n), key="heavy_top_n")

    # Reset button
    if st.sidebar.button("Reset to defaults", key="heavy_reset_btn"):
        for k, v in heavy_defaults.items():
            st.session_state[k] = v
        st.rerun()

    # === Base data (unchanged) ===
    base_df = data.copy()

    # Apply name filter
    if name_filter_h.strip():
        base_df = base_df[base_df["NAME"].str.contains(name_filter_h.strip(), case=False, na=False)]

    # Apply minimum ELITE filter
    if int(min_elite) > 0:
        base_df = base_df[base_df["ELITE"] >= int(min_elite)]

    # Filter by qualifier (MAX FLOOR)
    qualified = base_df[base_df["MAX FLOOR"] >= int(qualifier)].copy()
    qualified = qualified.sort_values(sort_by, ascending=False)

    # Top-up behavior
    desired_n = int(top_n_heavy)
    if use_topup and len(qualified) < desired_n:
        remainder = base_df[~base_df.index.isin(qualified.index)].copy()
        remainder = remainder.sort_values(sort_by, ascending=False)
        heavy_pred = pd.concat([qualified, remainder.head(desired_n - len(qualified))], ignore_index=True)
        heavy_pred = heavy_pred.sort_values(sort_by, ascending=False).head(desired_n).copy()
    else:
        heavy_pred = qualified.head(desired_n).copy()

    heavy_pred.insert(0, "RANK", [f"#{i+1}" for i in range(len(heavy_pred))])
    cols_hp = ["RANK", "NAME", "ELITE", "SUPER", "MEGA", "GAINS", "HEAVY %", "MIN FLOOR", "MAX FLOOR"]
    render_top50_table(heavy_pred, cols_hp, height_px=520)
