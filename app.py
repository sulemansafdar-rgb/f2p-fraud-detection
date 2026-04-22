"""
F2P Fraud Detection — Risky User Sessions
Streamlit app with live Metabase data, configurable scoring, and RP enrichment.
"""

import streamlit as st
import pandas as pd
import requests
import math
import time
from datetime import date, timedelta

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="F2P Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Metabase connection
# ============================================================
METABASE_URL = st.secrets.get("METABASE_URL", "")
METABASE_API_KEY = st.secrets.get("METABASE_API_KEY", "")
DB_RING = int(st.secrets.get("DB_RING", 166))       # Ring hand-level data
DB_MAIN = int(st.secrets.get("DB_MAIN", 67))        # Main DB (RP data)
CACHE_TTL = int(st.secrets.get("CACHE_TTL", 3600))  # Default 1 hour

if not METABASE_URL or not METABASE_API_KEY:
    st.error("⚠️ Metabase connection not configured. Add METABASE_URL and METABASE_API_KEY to Streamlit secrets.")
    st.code("""
# .streamlit/secrets.toml
METABASE_URL = "https://your-metabase-instance.com"
METABASE_API_KEY = "mb_xxxxxxxxxxxxx"
DB_RING = 166
DB_MAIN = 67
CACHE_TTL = 3600
    """, language="toml")
    st.stop()


def metabase_query(database_id: int, sql: str) -> pd.DataFrame:
    """Execute a native SQL query against Metabase and return a DataFrame."""
    headers = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}
    payload = {
        "database": database_id,
        "type": "native",
        "native": {"query": sql},
    }
    resp = requests.post(
        f"{METABASE_URL}/api/dataset",
        json=payload,
        headers=headers,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    cols = [c["name"] for c in data["cols"]]
    rows = data["rows"]
    return pd.DataFrame(rows, columns=cols)


def paginated_query(database_id: int, sql_template: str, page_size: int = 2000) -> pd.DataFrame:
    """Run a paginated SQL query, fetching all pages."""
    all_dfs = []
    offset = 0
    while True:
        sql = sql_template.format(offset=offset)
        df = metabase_query(database_id, sql)
        if df.empty:
            break
        all_dfs.append(df)
        if len(df) < page_size:
            break
        offset += page_size
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ============================================================
# SQL templates
# ============================================================

def session_sql(date_start: str, date_end: str) -> str:
    """Session scoring query for DB 166 (ring hand-level data)."""
    return f"""WITH game_agg AS (
    SELECT GAME_ID, USER_ID, INTERNAL_REFERENCE_NO,
           MAX(GAME_BIG_BLIND) as big_blind, SUM(WIN - STAKE) as profit_loss,
           TIMESTAMPDIFF(SECOND, MIN(STARTED), MAX(ENDED)) as game_duration,
           MIN(STARTED) as hand_start, MAX(ENDED) as hand_end
    FROM game_history_user
    WHERE TOURNAMENT_TYPE_ID = 6 AND IS_PVT_TABLE = 0
      AND CREATED_DATE >= '{date_start}' AND CREATED_DATE < '{date_end}'
    GROUP BY GAME_ID, USER_ID, INTERNAL_REFERENCE_NO
),
hand_details AS (
    SELECT hu.HAND_HISTORY_ID, hu.USER_ID, hu.IS_SHOWDOWN, hh.GAME_ID,
           COUNT(*) OVER (PARTITION BY hu.HAND_HISTORY_ID) as num_players
    FROM hand_history_user hu
    JOIN hand_history hh ON hu.HAND_HISTORY_ID = hh.HAND_HISTORY_ID
         AND hu.TOURNAMENT_TYPE_ID = hh.TOURNAMENT_TYPE_ID
    WHERE hu.TOURNAMENT_TYPE_ID = 6
      AND hu.CREATED_DATE >= '{date_start}' AND hu.CREATED_DATE < '{date_end}'
      AND hh.CREATED_DATE >= '{date_start}' AND hh.CREATED_DATE < '{date_end}'
),
hand_data AS (
    SELECT ga.USER_ID, ga.INTERNAL_REFERENCE_NO as session_id,
           ga.game_duration, ga.hand_start, ga.hand_end, ga.big_blind,
           hd.num_players, COALESCE(hd.IS_SHOWDOWN, 0) as is_showdown,
           CASE WHEN ga.big_blind > 0 THEN ga.profit_loss / ga.big_blind ELSE 0 END as profit_loss_bb
    FROM game_agg ga
    JOIN hand_details hd ON ga.GAME_ID = hd.GAME_ID AND ga.USER_ID = hd.USER_ID
),
session_agg AS (
    SELECT session_id, USER_ID as user_id,
           COUNT(*) as total_hands,
           SUM(CASE WHEN num_players <= 3 THEN 1 ELSE 0 END) as low_opponent_hands,
           SUM(profit_loss_bb) as total_profit_loss_bb,
           SUM(CASE WHEN num_players <= 2 AND is_showdown = 1 THEN profit_loss_bb ELSE 0 END) as sd_hu_profit_bb,
           SUM(CASE WHEN num_players <= 2 AND is_showdown = 1 THEN 1 ELSE 0 END) as sd_hu_hands,
           SUM(CASE WHEN num_players <= 2 AND is_showdown = 0 THEN profit_loss_bb ELSE 0 END) as nsd_hu_profit_bb,
           SUM(CASE WHEN num_players <= 2 AND is_showdown = 0 THEN 1 ELSE 0 END) as nsd_hu_hands,
           MIN(hand_start) as session_start, MAX(hand_end) as session_end
    FROM hand_data GROUP BY session_id, USER_ID
),
median_prep AS (
    SELECT session_id, USER_ID, game_duration,
           ROW_NUMBER() OVER (PARTITION BY session_id, USER_ID ORDER BY game_duration) as rn,
           COUNT(*) OVER (PARTITION BY session_id, USER_ID) as cnt
    FROM hand_data
),
session_median AS (
    SELECT session_id, USER_ID, AVG(game_duration) as median_hand_time
    FROM median_prep WHERE rn IN (FLOOR((cnt+1)/2), CEIL((cnt+1)/2))
    GROUP BY session_id, USER_ID
)
SELECT sa.session_id, sa.user_id, sa.total_hands, sa.low_opponent_hands,
       ROUND(sa.total_profit_loss_bb, 4) as total_profit_loss_bb,
       sa.sd_hu_hands, ROUND(sa.sd_hu_profit_bb, 4) as sd_hu_profit_bb,
       sa.nsd_hu_hands, ROUND(sa.nsd_hu_profit_bb, 4) as nsd_hu_profit_bb,
       ROUND(sm.median_hand_time, 2) as median_hand_time,
       sa.session_start, sa.session_end
FROM session_agg sa
JOIN session_median sm ON sa.session_id = sm.session_id AND sa.user_id = sm.USER_ID
ORDER BY sa.session_id
LIMIT 2000 OFFSET {{offset}}"""


def rp_earned_sql(date_start: str, date_end: str) -> str:
    """RP earned per user per day — DB 67."""
    return f"""SELECT
  rp.USER_ID as user_id,
  DATE(rp.TRANSACTION_DATE) as txn_date,
  ROUND(SUM(rp.TRANSACTION_AMOUNT), 2) as rp_earned
FROM master_transaction_history_baazirewardpoints rp
INNER JOIN transaction_type tt ON rp.TRANSACTION_TYPE_ID = tt.TRANSACTION_TYPE_ID
WHERE tt.TRANSACTION_DESCRIPTION IN ('LeaderBoard Prize', 'TOURNAMENT_WIN')
  AND rp.TRANSACTION_DATE >= '{date_start}' AND rp.TRANSACTION_DATE < '{date_end}'
GROUP BY rp.USER_ID, DATE(rp.TRANSACTION_DATE)
ORDER BY rp.USER_ID
LIMIT 2000 OFFSET {{offset}}"""


def rp_claimed_sql(date_start: str, date_end: str) -> str:
    """RP claimed per user per day — DB 67."""
    return f"""SELECT
  USER_ID as user_id,
  DATE(TRANSACTION_DATE) as txn_date,
  ROUND(SUM(TRANSACTION_AMOUNT), 2) as rp_claimed
FROM reward_store_transaction_history
WHERE TRANSACTION_STATUS_ID = 352
  AND TRANSACTION_DATE >= '{date_start}' AND TRANSACTION_DATE < '{date_end}'
GROUP BY USER_ID, DATE(TRANSACTION_DATE)
ORDER BY USER_ID
LIMIT 2000 OFFSET {{offset}}"""


def rp_earned_lifetime_sql() -> str:
    """Lifetime RP earned per user — DB 67."""
    return """SELECT
  rp.USER_ID as user_id,
  ROUND(SUM(rp.TRANSACTION_AMOUNT), 2) as rp_earned_lifetime
FROM master_transaction_history_baazirewardpoints rp
INNER JOIN transaction_type tt ON rp.TRANSACTION_TYPE_ID = tt.TRANSACTION_TYPE_ID
WHERE tt.TRANSACTION_DESCRIPTION IN ('LeaderBoard Prize', 'TOURNAMENT_WIN')
GROUP BY rp.USER_ID
ORDER BY rp.USER_ID
LIMIT 2000 OFFSET {offset}"""


def rp_claimed_lifetime_sql() -> str:
    """Lifetime RP claimed per user — DB 67."""
    return """SELECT
  USER_ID as user_id,
  ROUND(SUM(TRANSACTION_AMOUNT), 2) as rp_claimed_lifetime
FROM reward_store_transaction_history
WHERE TRANSACTION_STATUS_ID = 352
GROUP BY USER_ID
ORDER BY USER_ID
LIMIT 2000 OFFSET {offset}"""


# ============================================================
# Data fetching with caching
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching session data from DB...")
def fetch_sessions(date_start: str, date_end: str) -> pd.DataFrame:
    """Fetch raw session data from Metabase (DB 166)."""
    sql = session_sql(date_start, date_end)
    df = paginated_query(DB_RING, sql)
    if not df.empty:
        for col in ["total_hands", "low_opponent_hands", "sd_hu_hands", "nsd_hu_hands"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["total_profit_loss_bb", "sd_hu_profit_bb", "nsd_hu_profit_bb", "median_hand_time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["session_start"] = pd.to_datetime(df["session_start"])
        df["session_end"] = pd.to_datetime(df["session_end"])
        df["session_date"] = df["session_start"].dt.date
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching daily RP data...")
def fetch_rp_daily(date_start: str, date_end: str) -> tuple:
    """Fetch daily RP earned and claimed from Metabase (DB 67)."""
    earned = paginated_query(DB_MAIN, rp_earned_sql(date_start, date_end))
    claimed = paginated_query(DB_MAIN, rp_claimed_sql(date_start, date_end))
    if not earned.empty:
        earned["user_id"] = pd.to_numeric(earned["user_id"], errors="coerce").astype("Int64")
        earned["txn_date"] = pd.to_datetime(earned["txn_date"]).dt.date
        earned["rp_earned"] = pd.to_numeric(earned["rp_earned"], errors="coerce")
    if not claimed.empty:
        claimed["user_id"] = pd.to_numeric(claimed["user_id"], errors="coerce").astype("Int64")
        claimed["txn_date"] = pd.to_datetime(claimed["txn_date"]).dt.date
        claimed["rp_claimed"] = pd.to_numeric(claimed["rp_claimed"], errors="coerce")
    return earned, claimed


@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching lifetime RP data...")
def fetch_rp_lifetime() -> tuple:
    """Fetch lifetime RP earned and claimed from Metabase (DB 67)."""
    earned = paginated_query(DB_MAIN, rp_earned_lifetime_sql())
    claimed = paginated_query(DB_MAIN, rp_claimed_lifetime_sql())
    if not earned.empty:
        earned["user_id"] = pd.to_numeric(earned["user_id"], errors="coerce").astype("Int64")
        earned["rp_earned_lifetime"] = pd.to_numeric(earned["rp_earned_lifetime"], errors="coerce")
    if not claimed.empty:
        claimed["user_id"] = pd.to_numeric(claimed["user_id"], errors="coerce").astype("Int64")
        claimed["rp_claimed_lifetime"] = pd.to_numeric(claimed["rp_claimed_lifetime"], errors="coerce")
    return earned, claimed


def enrich_with_rp(sessions: pd.DataFrame, date_start: str, date_end: str) -> pd.DataFrame:
    """Join RP data onto session data."""
    df = sessions.copy()
    rp_earned_daily, rp_claimed_daily = fetch_rp_daily(date_start, date_end)
    rp_earned_life, rp_claimed_life = fetch_rp_lifetime()

    if not rp_earned_daily.empty:
        df = df.merge(
            rp_earned_daily.rename(columns={"rp_earned": "rp_earned_day"}),
            left_on=["user_id", "session_date"],
            right_on=["user_id", "txn_date"],
            how="left",
        ).drop(columns=["txn_date"], errors="ignore")
    else:
        df["rp_earned_day"] = 0

    if not rp_claimed_daily.empty:
        df = df.merge(
            rp_claimed_daily.rename(columns={"rp_claimed": "rp_claimed_day"}),
            left_on=["user_id", "session_date"],
            right_on=["user_id", "txn_date"],
            how="left",
        ).drop(columns=["txn_date"], errors="ignore")
    else:
        df["rp_claimed_day"] = 0

    if not rp_earned_life.empty:
        df = df.merge(rp_earned_life, on="user_id", how="left")
    else:
        df["rp_earned_lifetime"] = 0

    if not rp_claimed_life.empty:
        df = df.merge(rp_claimed_life, on="user_id", how="left")
    else:
        df["rp_claimed_lifetime"] = 0

    for col in ["rp_earned_day", "rp_claimed_day", "rp_earned_lifetime", "rp_claimed_lifetime"]:
        df[col] = df[col].fillna(0)

    return df


# ============================================================
# Scoring functions (configurable)
# ============================================================

DEFAULT_WR_BUCKETS = [
    (-999999, 0, 0.0),
    (0, 10, 0.1),
    (10, 50, 0.2),
    (50, 100, 0.3),
    (100, 200, 0.4),
    (200, 500, 0.5),
    (500, 1000, 0.6),
    (1000, 1500, 0.7),
    (1500, 2500, 0.8),
    (2500, 5000, 0.9),
    (5000, 999999, 1.0),
]

DEFAULT_WEIGHTS = {
    "win_rate": 0.25,
    "non_showdown_hu": 0.50,
    "showdown_hu": 0.20,
    "time": 0.05,
}

DEFAULT_LEVEL_THRESHOLDS = {"L1": 9.5, "L2": 9.0, "L3": 8.0, "L4": 7.0}


def generate_score_configurable(value, buckets):
    if value is None or pd.isna(value):
        return None
    for low, high, score in buckets:
        if value < 0 and low < 0:
            return score
        if low <= value <= high:
            return score
    return 1.0


def compute_time_score(median_hand_time, beta, k=0.03):
    if median_hand_time is None or pd.isna(median_hand_time):
        return None
    val = math.exp(-k * (median_hand_time - beta))
    return min(max(val, 0.0), 1.0)


def compute_cdb_score(win, non_sd, sd, time_s, weights):
    values = [win, non_sd, sd, time_s]
    weight_list = [weights["win_rate"], weights["non_showdown_hu"], weights["showdown_hu"], weights["time"]]
    total_w, weighted = 0.0, 0.0
    for v, w in zip(values, weight_list):
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            weighted += v * w
            total_w += w
    return (weighted / total_w) if total_w > 0 else None


def assign_level(score_10, thresholds):
    if score_10 is None or pd.isna(score_10):
        return None
    if score_10 >= thresholds["L1"]:
        return "L1"
    if score_10 >= thresholds["L2"]:
        return "L2"
    if score_10 >= thresholds["L3"]:
        return "L3"
    if score_10 >= thresholds["L4"]:
        return "L4"
    return "L5"


def score_dataframe(df, weights, beta, wr_buckets, level_thresholds):
    """Score all sessions with current config parameters."""
    out = df.copy()
    total_hands = out["total_hands"]

    # Win rates
    out["win_rate"] = out.apply(
        lambda r: r["total_profit_loss_bb"] / (r["total_hands"] / 100) if r["total_hands"] > 0 else None, axis=1
    )
    out["win_rate_non_showdown_heads_up"] = out.apply(
        lambda r: r["nsd_hu_profit_bb"] / (r["nsd_hu_hands"] / 100) if r["nsd_hu_hands"] > 0 else None, axis=1
    )
    out["win_rate_showdown_heads_up"] = out.apply(
        lambda r: r["sd_hu_profit_bb"] / (r["sd_hu_hands"] / 100) if r["sd_hu_hands"] > 0 else None, axis=1
    )
    out["low_opponent_hands_percentage"] = out.apply(
        lambda r: r["low_opponent_hands"] / r["total_hands"] if r["total_hands"] > 0 else 0, axis=1
    )

    # Scores
    out["win_rate_score"] = out["win_rate"].apply(lambda v: generate_score_configurable(v, wr_buckets))
    out["non_showdown_heads_up_score"] = out["win_rate_non_showdown_heads_up"].apply(
        lambda v: generate_score_configurable(v, wr_buckets)
    )
    out["showdown_heads_up_score"] = out["win_rate_showdown_heads_up"].apply(
        lambda v: generate_score_configurable(v, wr_buckets)
    )
    out["time_score"] = out["median_hand_time"].apply(lambda t: compute_time_score(t, beta))

    # Composite
    out["_cdb_raw"] = out.apply(
        lambda r: compute_cdb_score(
            r["win_rate_score"], r["non_showdown_heads_up_score"],
            r["showdown_heads_up_score"], r["time_score"], weights
        ), axis=1,
    )
    out["cdb_score"] = out["_cdb_raw"].apply(
        lambda v: round(v * 10, 2) if v is not None and not pd.isna(v) else None
    )
    out["flag"] = out["cdb_score"].apply(lambda v: assign_level(v, level_thresholds))

    out.drop(columns=["_cdb_raw"], inplace=True)
    return out


# ============================================================
# SIDEBAR — Configuration
# ============================================================
st.sidebar.title("⚙️ Configuration")

# --- Date range for data fetch ---
st.sidebar.header("📅 Data Period")
col_from, col_to = st.sidebar.columns(2)
default_from = date.today() - timedelta(days=7)
default_to = date.today()
date_from = col_from.date_input("From", value=default_from, key="date_from")
date_to = col_to.date_input("To", value=default_to, key="date_to")

date_start = date_from.strftime("%Y-%m-%d")
date_end = (date_to + timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive end

if st.sidebar.button("🔄 Refresh data", help="Clear cache and re-fetch from DB"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Cache TTL: {CACHE_TTL // 60} minutes")

# --- Section 1: Score Config ---
st.sidebar.header("1. Score Config")

st.sidebar.subheader("Scoring Weights")
st.sidebar.caption("Weights auto-normalise to sum to 1.0")

w_wr = st.sidebar.slider("Win Rate", 0.0, 1.0, DEFAULT_WEIGHTS["win_rate"], 0.05, key="w_wr")
w_nsd = st.sidebar.slider("Non-Showdown Heads-Up", 0.0, 1.0, DEFAULT_WEIGHTS["non_showdown_hu"], 0.05, key="w_nsd")
w_sd = st.sidebar.slider("Showdown Heads-Up", 0.0, 1.0, DEFAULT_WEIGHTS["showdown_hu"], 0.05, key="w_sd")
w_time = st.sidebar.slider("Time", 0.0, 1.0, DEFAULT_WEIGHTS["time"], 0.05, key="w_time")

raw_sum = w_wr + w_nsd + w_sd + w_time
if raw_sum > 0:
    weights = {
        "win_rate": w_wr / raw_sum,
        "non_showdown_hu": w_nsd / raw_sum,
        "showdown_hu": w_sd / raw_sum,
        "time": w_time / raw_sum,
    }
else:
    weights = DEFAULT_WEIGHTS

st.sidebar.caption(
    f"Raw sum: {raw_sum:.2f} → Normalised: "
    f"WR={weights['win_rate']:.2f}, NSD={weights['non_showdown_hu']:.2f}, "
    f"SD={weights['showdown_hu']:.2f}, T={weights['time']:.2f}"
)

# Fast play threshold
st.sidebar.subheader("Fast Play Threshold")
beta = st.sidebar.slider(
    "Threshold (seconds)", min_value=5, max_value=60, value=15, step=1, key="beta",
    help="Sessions with median hand time below this are considered 'fast play'.",
)

# Win Rate Scoring Buckets
with st.sidebar.expander("Win Rate Scoring Buckets", expanded=False):
    st.caption("Maps win rate (BB/100) to a 0–1 score.")
    bucket_data = pd.DataFrame(
        [{"From (BB/100)": b[0], "To (BB/100)": b[1], "Score": b[2]} for b in DEFAULT_WR_BUCKETS]
    )
    edited_buckets = st.data_editor(bucket_data, num_rows="fixed", use_container_width=True, key="wr_buckets")
    wr_buckets = [(row["From (BB/100)"], row["To (BB/100)"], row["Score"]) for _, row in edited_buckets.iterrows()]

# Level Thresholds
st.sidebar.subheader("Level Thresholds")
st.sidebar.caption("Score on 0–10 scale. L5 = below L4 threshold.")
col_l1, col_l2 = st.sidebar.columns(2)
th_l1 = col_l1.number_input("L1 ≥", value=DEFAULT_LEVEL_THRESHOLDS["L1"], step=0.1, format="%.1f", key="th_l1")
th_l2 = col_l2.number_input("L2 ≥", value=DEFAULT_LEVEL_THRESHOLDS["L2"], step=0.1, format="%.1f", key="th_l2")
col_l3, col_l4 = st.sidebar.columns(2)
th_l3 = col_l3.number_input("L3 ≥", value=DEFAULT_LEVEL_THRESHOLDS["L3"], step=0.1, format="%.1f", key="th_l3")
th_l4 = col_l4.number_input("L4 ≥", value=DEFAULT_LEVEL_THRESHOLDS["L4"], step=0.1, format="%.1f", key="th_l4")
level_thresholds = {"L1": th_l1, "L2": th_l2, "L3": th_l3, "L4": th_l4}

# --- Section 2: Risky Session Filters ---
st.sidebar.header("2. Risky Session Filters")
low_opp_pct_display = st.sidebar.slider(
    "Min Low-Opponent Hands %", min_value=0, max_value=100, value=85, step=5,
    format="%d%%", key="low_opp_pct",
    help="Only show sessions where ≥ this % of hands were played with ≤3 opponents",
)
low_opp_pct = low_opp_pct_display / 100.0

min_hands = st.sidebar.number_input("Min Total Hands", min_value=0, value=0, step=1, key="min_hands")
score_not_null = st.sidebar.checkbox("Score not null", value=True, key="score_not_null")

# --- Section 3: Data Extraction Filters ---
st.sidebar.header("3. Data Extraction Filters")
st.sidebar.caption("These reflect how the raw data was extracted.")
st.sidebar.text("Game Type: Ring (Tournament Type 6)")
st.sidebar.text("Table Type: Public only")
low_opp_count = st.sidebar.number_input(
    "Low-opponent player count (≤)", min_value=2, max_value=6, value=3, step=1, key="low_opp_count",
    help="Changing this requires re-extraction from DB.",
)

st.sidebar.divider()
st.sidebar.caption("v2.0 — Live Metabase · Risky User Sessions")


# ============================================================
# MAIN AREA — Fetch, Score, Display
# ============================================================

st.title("🔍 Risky User Sessions")
st.caption("Chip dumping detection — live from Metabase with configurable scoring")

# --- Fetch data ---
raw_df = fetch_sessions(date_start, date_end)

if raw_df.empty:
    st.warning(f"No session data found for {date_from} → {date_to}. Try a different date range.")
    st.stop()

# Enrich with RP
enriched_df = enrich_with_rp(raw_df, date_start, date_end)

# Dynamic percentile context for beta
all_median_times = enriched_df["median_hand_time"].dropna()
if len(all_median_times) > 0:
    pct_below = (all_median_times < beta).sum() / len(all_median_times) * 100
    st.sidebar.caption(
        f"📊 {pct_below:.1f}% of sessions have median hand time < {beta}s "
        f"(median: {all_median_times.median():.0f}s)"
    )

# Score
scored_df = score_dataframe(enriched_df, weights, beta, wr_buckets, level_thresholds)

# --- Apply risky session filters ---
filtered = scored_df.copy()
filtered = filtered[filtered["low_opponent_hands_percentage"] >= low_opp_pct]
if min_hands > 0:
    filtered = filtered[filtered["total_hands"] >= min_hands]
if score_not_null:
    filtered = filtered[filtered["cdb_score"].notna()]

# --- Top filter bar ---
filter_cols = st.columns([2, 1.5, 1.5, 1.5])

with filter_cols[0]:
    available_dates = sorted(filtered["session_date"].dropna().unique())
    if available_dates:
        date_range = st.date_input(
            "Session Date", value=(min(available_dates), max(available_dates)),
            min_value=min(available_dates), max_value=max(available_dates), key="date_range",
        )
    else:
        date_range = None

with filter_cols[1]:
    level_options = ["All"] + sorted(
        [l for l in filtered["flag"].dropna().unique()],
        key=lambda x: int(x[1]) if x and len(x) == 2 else 99,
    )
    selected_level = st.selectbox("Level", level_options, key="level_filter")

with filter_cols[2]:
    user_search = st.text_input("User ID Search", key="user_search", placeholder="Enter User ID...")

with filter_cols[3]:
    fraud_status = st.selectbox(
        "RP Activity", ["All", "Has RP Activity", "No RP Activity"], key="fraud_status",
    )

# Apply top filters
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered["session_date"] >= date_range[0]) & (filtered["session_date"] <= date_range[1])
    ]
if selected_level != "All":
    filtered = filtered[filtered["flag"] == selected_level]
if user_search:
    filtered = filtered[filtered["user_id"].astype(str).str.contains(user_search, na=False)]
if fraud_status == "Has RP Activity":
    filtered = filtered[(filtered["rp_earned_day"] > 0) | (filtered["rp_claimed_day"] > 0)]
elif fraud_status == "No RP Activity":
    filtered = filtered[(filtered["rp_earned_day"] == 0) & (filtered["rp_claimed_day"] == 0)]

# --- Summary metrics ---
metric_cols = st.columns(6)
metric_cols[0].metric("Total Sessions", f"{len(filtered):,}")
metric_cols[1].metric("L1 Sessions", f"{(filtered['flag'] == 'L1').sum():,}")
metric_cols[2].metric("L2 Sessions", f"{(filtered['flag'] == 'L2').sum():,}")
metric_cols[3].metric("L3 Sessions", f"{(filtered['flag'] == 'L3').sum():,}")
metric_cols[4].metric("Unique Users", f"{filtered['user_id'].nunique():,}")
metric_cols[5].metric("Avg Score", f"{filtered['cdb_score'].mean():.2f}" if len(filtered) > 0 else "—")

st.divider()

# --- Data table ---
display_cols = [
    "user_id", "cdb_score", "session_id", "flag", "win_rate", "median_hand_time",
    "total_hands", "rp_earned_day", "rp_claimed_day", "rp_earned_lifetime",
    "rp_claimed_lifetime", "session_date", "session_start", "session_end",
]

display_names = {
    "user_id": "User ID", "cdb_score": "Chip Dumping Score", "session_id": "Session ID",
    "flag": "Level", "win_rate": "Win Rate (BB/100)", "median_hand_time": "Median Hand Time (s)",
    "total_hands": "Total Hands", "rp_earned_day": "RP Earned (Day)",
    "rp_claimed_day": "RP Claimed (Day)", "rp_earned_lifetime": "RP Earned (Lifetime)",
    "rp_claimed_lifetime": "RP Claimed (Lifetime)", "session_date": "Session Date",
    "session_start": "Session Start", "session_end": "Session End",
}

table_df = (
    filtered[display_cols]
    .sort_values("cdb_score", ascending=False)
    .rename(columns=display_names)
    .reset_index(drop=True)
)


def colour_level(val):
    colours = {
        "L1": "background-color: #ff4444; color: white; font-weight: bold",
        "L2": "background-color: #ff8800; color: white; font-weight: bold",
        "L3": "background-color: #ffcc00; color: black",
        "L4": "background-color: #88cc00; color: black",
        "L5": "background-color: #44aa44; color: white",
    }
    return colours.get(val, "")


format_dict = {
    "Chip Dumping Score": "{:.2f}", "Win Rate (BB/100)": "{:.2f}",
    "Median Hand Time (s)": "{:.1f}", "RP Earned (Day)": "{:,.0f}",
    "RP Claimed (Day)": "{:,.0f}", "RP Earned (Lifetime)": "{:,.0f}",
    "RP Claimed (Lifetime)": "{:,.0f}",
}

styled = table_df.style.format(format_dict, na_rep="—").map(colour_level, subset=["Level"])

st.dataframe(styled, use_container_width=True, height=600, column_config={
    "User ID": st.column_config.NumberColumn(format="%d"),
    "Session ID": st.column_config.TextColumn(),
    "Total Hands": st.column_config.NumberColumn(format="%d"),
})

# --- Download ---
csv_export = table_df.to_csv(index=False)
st.download_button(
    label="📥 Download filtered results as CSV",
    data=csv_export, file_name="risky_sessions_filtered.csv", mime="text/csv",
)

# --- Score distribution ---
with st.expander("📊 Score Distribution", expanded=False):
    if len(filtered) > 0:
        hist_cols = st.columns(2)
        with hist_cols[0]:
            st.subheader("CBD Score Distribution")
            st.bar_chart(filtered["cdb_score"].dropna().value_counts(bins=20).sort_index())
        with hist_cols[1]:
            st.subheader("Level Distribution")
            st.bar_chart(filtered["flag"].value_counts().reindex(["L1", "L2", "L3", "L4", "L5"], fill_value=0))
