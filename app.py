"""
Warkop Intelligence Finder - Lhokseumawe Edition v4.2
Tema Warkop - cozy, warm, coffeehouse style UI with Gradio
"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import pandas as pd

from models.recommender import WarkopRecommender, WarkopRecommenderError

# =============================================================================
# Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "warkops.csv"

APP_TITLE = "Warkop Intelligence Finder"
APP_SUBTITLE = "Lhokseumawe Edition v4.2"

PRESET_PROFILES = {
    "coding": {
        "label": "Coding",
        "subtitle": "Deep Work",
        "weights": {
            "wifi_imp": 0.45,
            "socket_imp": 0.30,
            "quiet_imp": 0.20,
            "value_imp": 0.05,
        },
    },
    "social": {
        "label": "Social",
        "subtitle": "Hangout",
        "weights": {
            "wifi_imp": 0.15,
            "socket_imp": 0.20,
            "quiet_imp": 0.05,
            "value_imp": 0.60,
        },
    },
    "student": {
        "label": "Student",
        "subtitle": "Budget",
        "weights": {
            "wifi_imp": 0.35,
            "socket_imp": 0.25,
            "quiet_imp": 0.10,
            "value_imp": 0.30,
        },
    },
    "premium": {
        "label": "Premium",
        "subtitle": "Quality",
        "weights": {
            "wifi_imp": 0.40,
            "socket_imp": 0.25,
            "quiet_imp": 0.30,
            "value_imp": 0.05,
        },
    },
    "chill": {
        "label": "Chill",
        "subtitle": "Relax",
        "weights": {
            "wifi_imp": 0.10,
            "socket_imp": 0.15,
            "quiet_imp": 0.45,
            "value_imp": 0.30,
        },
    },
}

PRICE_RANGES = ["Cheap", "Medium", "Expensive"]
LEVEL_BARS = {"Low": 1, "Medium": 2, "High": 3}
PRICE_SYMBOLS = {"Cheap": "$","Medium": "$$","Expensive": "$$$",}

REQUIRED_COLUMNS = {
    "name",
    "address",
    "wifi_speed_mbps",
    "socket_availability",
    "noise_level",
    "vibe_category",
    "price_range",
}


# =============================================================================
# Initialize Recommender
# =============================================================================
def initialize_recommender() -> Optional[WarkopRecommender]:
    try:
        if not CSV_PATH.exists():
            logger.error("CSV file not found: %s", CSV_PATH)
            return None

        rec = WarkopRecommender(str(CSV_PATH))
        logger.info("Loaded %s warkops successfully", len(rec))
        return rec
    except Exception as e:
        logger.exception("Failed to initialize recommender: %s", e)
        return None


recommender = initialize_recommender()


def _safe_df() -> pd.DataFrame:
    if recommender is None or not hasattr(recommender, "df") or recommender.df is None:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    return recommender.df.copy()


BASE_DF = _safe_df()

VIBE_TAGS = (
    recommender.get_all_vibe_tags()
    if recommender is not None and hasattr(recommender, "get_all_vibe_tags")
    else []
)

WARKOP_NAMES = (
    sorted(BASE_DF["name"].dropna().astype(str).unique().tolist())
    if "name" in BASE_DF.columns
    else []
)

ADDRESS_LIST = (
    sorted(BASE_DF["address"].dropna().astype(str).unique().tolist())
    if "address" in BASE_DF.columns
    else []
)


# =============================================================================
# Utility Helpers
# =============================================================================
def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _status_msg(msg: str, level: str = "info") -> str:
    level = escape(level)
    return f'<div class="status-bar status-{level}"><span class="status-dot"></span>{msg}</div>'


def _no_recommender_msg() -> str:
    detail = escape(str(CSV_PATH))
    return _status_msg(
        f"Recommender belum berhasil dimuat. Cek file <code>{detail}</code> dan format dataset.",
        "error",
    )


def _empty_initial_msg(msg: str) -> str:
    return f'<div class="empty-state-initial">{escape(msg)}</div>'


def _validate_recommender_ready() -> Optional[Tuple[str, str]]:
    if recommender is None:
        return _no_recommender_msg(), ""

    df = _safe_df()
    if df.empty:
        return _status_msg("Dataset kosong atau gagal dimuat.", "warning"), ""

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return (
            _status_msg(
                f"Kolom dataset tidak lengkap: {escape(', '.join(sorted(missing)))}",
                "error",
            ),
            "",
        )

    return None


# =============================================================================
# Card Renderer
# =============================================================================
def _render_level_bars(level: str) -> str:
    level = _safe_str(level, "Unknown")
    val = LEVEL_BARS.get(level, 0)
    bars = "".join(
        f'<span class="bar {"active" if i < val else ""}"></span>'
        for i in range(3)
    )
    return f'<div class="level-bars" title="{escape(level)}">{bars}</div>'


def _render_wifi_indicator(speed: float) -> str:
    speed = _safe_float(speed, 0.0)

    if speed >= 50:
        cls = "wifi-fast"
    elif speed >= 30:
        cls = "wifi-medium"
    else:
        cls = "wifi-slow"

    return f'<span class="wifi-badge {cls}">{int(speed)} Mbps</span>'


def _generate_summary(row: pd.Series) -> str:
    wifi = _safe_float(row.get("wifi_speed_mbps", 0))
    socket = _safe_str(row.get("socket_availability", "Medium"))
    noise = _safe_str(row.get("noise_level", "Medium"))
    price = _safe_str(row.get("price_range", "Medium"))
    vibe = _safe_str(row.get("vibe_category", ""))

    points = []

    if wifi >= 50:
        points.append("WiFi cepat")
    elif wifi >= 30:
        points.append("WiFi cukup stabil")

    if socket == "High":
        points.append("colokan banyak")

    if noise == "Low":
        points.append("suasana tenang")
    elif noise == "Medium":
        points.append("suasana cukup nyaman")

    if price == "Cheap":
        points.append("ramah di kantong")
    elif price == "Medium":
        points.append("harga masih aman")

    vibe_lower = vibe.lower()
    if "coding" in vibe_lower or "work" in vibe_lower or "study" in vibe_lower:
        points.append("cocok buat kerja atau nugas")
    elif "social" in vibe_lower or "hangout" in vibe_lower:
        points.append("enak buat nongkrong")

    if not points:
        return "Cocok untuk santai dan menikmati suasana."

    return "Cocok untuk " + ", ".join(points[:4]) + "."


def _render_card(row: pd.Series, rank: Optional[int] = None) -> str:
    name = escape(_safe_str(row.get("name", "-")))
    address = escape(_safe_str(row.get("address", "-")))
    vibe_raw = _safe_str(row.get("vibe_category", "-"))
    wifi = _safe_float(row.get("wifi_speed_mbps", 0))
    socket = _safe_str(row.get("socket_availability", "Medium"))
    noise = _safe_str(row.get("noise_level", "Medium"))
    price = _safe_str(row.get("price_range", "Medium"))
    summary = escape(_generate_summary(row))

    score_html = ""
    if "match_score" in row.index and pd.notna(row.get("match_score")):
        score_html = f'<div class="card-score">Cocok {_safe_float(row.get("match_score")):.0f}<small>%</small></div>'
    elif "similarity_score" in row.index and pd.notna(row.get("similarity_score")):
        score_html = f'<div class="card-score similarity">Mirip {_safe_float(row.get("similarity_score")):.0f}<small>%</small></div>'

    rank_html = f'<div class="card-rank">#{rank}</div>' if rank is not None else ""

    vibe_tags = "".join(
        f'<span class="vibe-tag">{escape(tag.strip())}</span>'
        for tag in vibe_raw.split("/")
        if tag.strip()
    )

    price_key = price if price in PRICE_SYMBOLS else "Medium"
    price_class = escape(price.lower().replace(" ", "-"))

    return f"""
    <div class="warkop-card">
        <div class="card-header">
            {rank_html}
            <div class="card-title-section">
                <h3 class="card-title">{name}</h3>
                <div class="card-address">{address}</div>
            </div>
            {score_html}
        </div>
        <div class="card-summary">{summary}</div>
        <div class="card-tags">{vibe_tags or '<span class="vibe-tag">Nyaman</span>'}</div>
        <div class="card-stats">
            <div class="stat-item">
                <span class="stat-icon-wifi"></span>
                <div class="stat-content">
                    <div class="stat-label">Internet</div>
                    {_render_wifi_indicator(wifi)}
                </div>
            </div>
            <div class="stat-item">
                <span class="stat-icon-socket"></span>
                <div class="stat-content">
                    <div class="stat-label">Colokan</div>
                    {_render_level_bars(socket)}
                </div>
            </div>
            <div class="stat-item">
                <span class="stat-icon-noise"></span>
                <div class="stat-content">
                    <div class="stat-label">Suasana</div>
                    {_render_level_bars(noise)}
                </div>
            </div>
            <div class="stat-item">
                <span class="stat-icon-price"></span>
                <div class="stat-content">
                    <div class="stat-label">Harga</div>
                    <span class="price-badge price-{price_class}">{PRICE_SYMBOLS.get(price_key, "?")}</span>
                </div>
            </div>
        </div>
    </div>
    """


def _render_cards(df: Optional[pd.DataFrame], empty_msg: str = "Belum ada hasil") -> str:
    if df is None or df.empty:
        return f"""
        <div class="empty-state">
            <div class="empty-icon"></div>
            <div class="empty-title">{escape(empty_msg)}</div>
            <div class="empty-subtitle">Coba ubah filter atau query Anda</div>
        </div>
        """

    cards = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        cards.append(_render_card(row, rank=idx))

    return f'<div class="cards-grid">{"".join(cards)}</div>'


# =============================================================================
# Feature Handlers
# =============================================================================
def smart_recommend(wifi, socket, quiet, value, price_filter, vibe_filter, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    try:
        filters = {}

        if price_filter and price_filter != "Semua":
            filters["price_range"] = price_filter

        if vibe_filter and vibe_filter != "Semua":
            filters["vibe_category"] = vibe_filter

        result = recommender.recommend_by_weights(
            wifi_imp=float(wifi),
            socket_imp=float(socket),
            quiet_imp=float(quiet),
            value_imp=float(value),
            top_n=int(top_n),
            filters=filters or None,
        )

        if result is None or result.empty:
            return (
                _render_cards(result, "Tidak ada warkop sesuai filter"),
                _status_msg("Coba longgarkan filter Anda.", "warning"),
            )

        return (
            _render_cards(result),
            _status_msg(f"Berhasil menemukan {len(result)} rekomendasi.", "success"),
        )

    except Exception as e:
        logger.exception("smart_recommend error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def use_preset(preset_key: str) -> Tuple[float, float, float, float]:
    profile = PRESET_PROFILES.get(preset_key, PRESET_PROFILES["coding"])
    weights = profile["weights"]
    return (
        weights["wifi_imp"],
        weights["socket_imp"],
        weights["quiet_imp"],
        weights["value_imp"],
    )


def story_search(query, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    query = _safe_str(query, "").strip()
    if not query:
        return _empty_initial_msg("Tulis deskripsi pencarian Anda"), _status_msg("Query kosong.", "warning")

    try:
        result = recommender.recommend_by_story(query, top_n=int(top_n))

        if result is None or result.empty:
            return _render_cards(result, "Tidak ada hasil relevan"), _status_msg("Tidak ada hasil relevan.", "warning")

        short_query = escape(query[:80])
        return (
            _render_cards(result),
            _status_msg(f'{len(result)} warkop ditemukan untuk: "{short_query}"', "success"),
        )

    except Exception as e:
        logger.exception("story_search error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def hybrid_search(query, wifi, socket, quiet, value, ratio, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    try:
        result = recommender.recommend_hybrid(
            user_query=_safe_str(query, ""),
            weights={
                "wifi_imp": float(wifi),
                "socket_imp": float(socket),
                "quiet_imp": float(quiet),
                "value_imp": float(value),
            },
            text_ratio=float(ratio),
            top_n=int(top_n),
        )

        if result is None or result.empty:
            return _render_cards(result, "Tidak ada hasil"), _status_msg("Tidak ada hasil.", "warning")

        return (
            _render_cards(result),
            _status_msg(f"Hasil pencarian pintar menemukan {len(result)} rekomendasi.", "success"),
        )

    except Exception as e:
        logger.exception("hybrid_search error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def vibe_search(vibe_tag, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    vibe_tag = _safe_str(vibe_tag, "").strip()
    if not vibe_tag:
        return _empty_initial_msg("Pilih kategori vibe"), _status_msg("Pilih vibe dulu.", "warning")

    try:
        result = recommender.recommend_by_weights(
            top_n=int(top_n),
            filters={"vibe_category": vibe_tag},
        )

        if result is None or result.empty:
            return (
                _render_cards(result, f"Tidak ada warkop dengan vibe '{vibe_tag}'"),
                _status_msg(f"Tidak ada warkop dengan vibe '{escape(vibe_tag)}'.", "warning"),
            )

        return (
            _render_cards(result),
            _status_msg(f"{len(result)} warkop dengan vibe '{escape(vibe_tag)}'.", "success"),
        )

    except Exception as e:
        logger.exception("vibe_search error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def find_similar(warkop_name, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    warkop_name = _safe_str(warkop_name, "").strip()
    if not warkop_name:
        return _empty_initial_msg("Pilih warkop referensi"), _status_msg("Pilih warkop dulu.", "warning")

    try:
        result = recommender.find_similar(warkop_name, top_n=int(top_n))

        if result is None or result.empty:
            return _render_cards(result, "Tidak ditemukan warkop serupa"), _status_msg("Tidak ada hasil serupa.", "warning")

        return (
            _render_cards(result),
            _status_msg(f"Warkop yang mirip dengan '{escape(warkop_name)}'.", "success"),
        )

    except WarkopRecommenderError as e:
        logger.warning("find_similar warning: %s", e)
        return "", _status_msg(escape(str(e)), "warning")
    except Exception as e:
        logger.exception("find_similar error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def filter_by_address(address, top_n):
    validation = _validate_recommender_ready()
    if validation:
        return validation

    address = _safe_str(address, "").strip()
    if not address:
        return _empty_initial_msg("Pilih kawasan"), _status_msg("Pilih kawasan dulu.", "warning")

    try:
        df = _safe_df()
        result = df[df["address"].astype(str).str.contains(address, case=False, na=False)]

        if result.empty:
            return _render_cards(result, f"Tidak ada di '{address}'"), _status_msg(
                f"Tidak ada warkop di '{escape(address)}'.",
                "warning",
            )

        result = result.head(int(top_n))
        return (
            _render_cards(result),
            _status_msg(f"{len(result)} warkop di '{escape(address)}'.", "success"),
        )

    except Exception as e:
        logger.exception("filter_by_address error")
        return "", _status_msg(f"Error: {escape(str(e))}", "error")


def compare_warkops(warkop_a, warkop_b):
    validation = _validate_recommender_ready()
    if validation:
        return validation[0]

    warkop_a = _safe_str(warkop_a, "").strip()
    warkop_b = _safe_str(warkop_b, "").strip()

    if not warkop_a or not warkop_b:
        return _empty_initial_msg("Pilih dua warkop untuk dibandingkan")

    if warkop_a == warkop_b:
        return _status_msg("Pilih dua warkop yang berbeda.", "warning")

    try:
        df = _safe_df()
        row_a_df = df[df["name"].astype(str) == warkop_a]
        row_b_df = df[df["name"].astype(str) == warkop_b]

        if row_a_df.empty or row_b_df.empty:
            return _status_msg("Salah satu warkop tidak ditemukan di database.", "warning")

        row_a = row_a_df.iloc[0]
        row_b = row_b_df.iloc[0]

        def _compare_row(label, val_a, val_b) -> str:
            return (
                '<div class="cmp-row">'
                f'<div class="cmp-label">{escape(str(label))}</div>'
                f'<div class="cmp-val">{escape(_safe_str(val_a))}</div>'
                f'<div class="cmp-val">{escape(_safe_str(val_b))}</div>'
                "</div>"
            )

        rows = [
            _compare_row("Alamat", row_a.get("address"), row_b.get("address")),
            _compare_row("WiFi", f"{_safe_str(row_a.get('wifi_speed_mbps'))} Mbps", f"{_safe_str(row_b.get('wifi_speed_mbps'))} Mbps"),
            _compare_row("Colokan", row_a.get("socket_availability"), row_b.get("socket_availability")),
            _compare_row("Noise", row_a.get("noise_level"), row_b.get("noise_level")),
            _compare_row("Harga", row_a.get("price_range"), row_b.get("price_range")),
            _compare_row("Vibe", row_a.get("vibe_category"), row_b.get("vibe_category")),
        ]

        return f"""
        <div class="compare-container">
            <div class="cmp-header">
                <div class="cmp-label-h">Atribut</div>
                <div class="cmp-name-h">{escape(warkop_a)}</div>
                <div class="cmp-name-h">{escape(warkop_b)}</div>
            </div>
            <div class="cmp-body">{''.join(rows)}</div>
        </div>
        """

    except Exception as e:
        logger.exception("compare_warkops error")
        return _status_msg(f"Error: {escape(str(e))}", "error")


def get_stats():
    validation = _validate_recommender_ready()
    if validation:
        return validation[0]

    try:
        s = recommender.get_quick_stats()
        top_vibes = s.get("Top Vibe Tags", {}) or {}

        top_vibes_html = "".join(
            f'<div class="vibe-stat-item"><span class="vibe-stat-name">{escape(str(k))}</span><span class="vibe-stat-count">{escape(str(v))}</span></div>'
            for k, v in top_vibes.items()
        )

        return f"""
        <div class="stats-hero">
            <div class="stats-hero-number">{escape(str(s.get('Total Warkop', 0)))}</div>
            <div class="stats-hero-label">Total Warkops Tersedia di Lhokseumawe</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-blue"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">WiFi Tercepat</div>
                    <div class="stat-card-value">{_safe_float(s.get('WiFi Tercepat (Mbps)', 0)):.0f}<small> Mbps</small></div>
                </div>
            </div>
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-blue"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">WiFi Rata-rata</div>
                    <div class="stat-card-value">{_safe_float(s.get('WiFi Rata-rata (Mbps)', 0)):.1f}<small> Mbps</small></div>
                </div>
            </div>
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-green"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">Spot Tenang</div>
                    <div class="stat-card-value">{escape(_safe_str(s.get('Spot Paling Tenang', '-')))}</div>
                </div>
            </div>
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-amber"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">Budget Friendly</div>
                    <div class="stat-card-value">{escape(_safe_str(s.get('Budget Friendly', '-')))}</div>
                </div>
            </div>
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-purple"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">Premium</div>
                    <div class="stat-card-value">{escape(_safe_str(s.get('Premium', '-')))}</div>
                </div>
            </div>
            <div class="stat-card-v2">
                <div class="stat-card-icon stat-teal"></div>
                <div class="stat-card-body">
                    <div class="stat-card-label">Colokan Melimpah</div>
                    <div class="stat-card-value">{escape(_safe_str(s.get('Socket Melimpah', '-')))}</div>
                </div>
            </div>
        </div>

        <div class="vibe-stats-section">
            <h3 class="section-title">Top Vibe Categories</h3>
            <div class="vibe-stats-grid">{top_vibes_html or '<div class="empty-subtitle">Tidak ada data vibe.</div>'}</div>
        </div>
        """

    except Exception as e:
        logger.exception("get_stats error")
        return _status_msg(f"Error: {escape(str(e))}", "error")


def display_all(sort_by, ascending):
    validation = _validate_recommender_ready()
    if validation:
        return validation[0]

    try:
        df = _safe_df()

        if sort_by in df.columns:
            if sort_by in {"name", "address", "vibe_category", "price_range", "noise_level", "socket_availability"}:
                df[sort_by] = df[sort_by].astype(str)
            df = df.sort_values(sort_by, ascending=bool(ascending), na_position="last")

        df = df.drop(columns=[c for c in ["metadata"] if c in df.columns], errors="ignore")
        return _render_cards(df, "Database kosong")

    except Exception as e:
        logger.exception("display_all error")
        return _status_msg(f"Error: {escape(str(e))}", "error")


# =============================================================================
# Custom CSS - Tema Warkop
# =============================================================================
CUSTOM_CSS = """
:root {
    --color-primary: #8B5E3C;
    --color-primary-dark: #6F4528;
    --color-primary-light: #B07A52;
    --color-accent: #D4A574;
    --color-accent-dark: #B98552;

    --bg-main: #2B1D17;
    --bg-soft: #3A281F;
    --bg-card: #4A3328;
    --bg-card-2: #5A4032;
    --bg-hover: #674938;

    --border-color: #6B4A37;
    --border-light: #7A5440;

    --color-text: #F6EBDD;
    --color-text-light: #D8C3B0;
    --color-muted: #B89B83;

    --color-success: #7BA05B;
    --color-warning: #D4A574;
    --color-error: #C97B63;

    --shadow-sm: 0 3px 10px rgba(0, 0, 0, 0.18);
    --shadow-md: 0 8px 20px rgba(0, 0, 0, 0.22);
    --shadow-lg: 0 14px 32px rgba(0, 0, 0, 0.26);

    --radius-sm: 12px;
    --radius-md: 16px;
    --radius-lg: 22px;
}

html, body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(212,165,116,0.08), transparent 20%),
        radial-gradient(circle at bottom right, rgba(139,94,60,0.14), transparent 25%),
        linear-gradient(180deg, #241813 0%, #2B1D17 100%) !important;
    color: var(--color-text) !important;
}

.gradio-container {
    max-width: 1380px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    padding-bottom: 20px;
}

* { box-sizing: border-box; }

label, p, span, div { color: inherit; }

/* HEADER */
.app-header {
    background:
        radial-gradient(circle at top right, rgba(212,165,116,0.16), transparent 24%),
        linear-gradient(135deg, #3A281F 0%, #4A3328 100%);
    color: white;
    padding: 34px 30px;
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
}

.app-header::before {
    content: '';
    position: absolute;
    top: -90px;
    right: -80px;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(255, 220, 180, 0.07) 0%, transparent 70%);
    opacity: 1;
}

.app-header::after {
    content: '';
    position: absolute;
    bottom: -60px;
    left: -40px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(139,94,60,0.18) 0%, transparent 70%);
    opacity: 1;
}

.app-header-content { position: relative; z-index: 1; }

.app-brand {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}

.app-logo {
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #E6C29A 0%, #D4A574 100%);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: 900;
    color: #4A2F22;
    box-shadow: 0 8px 20px rgba(212, 165, 116, 0.24);
}

.app-title {
    font-size: 2em;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.15;
    color: #FFF5EA;
}

.app-subtitle {
    font-size: 0.98em;
    margin-top: 4px;
    max-width: 760px;
    line-height: 1.6;
    color: #EADACA;
}

.app-meta {
    display: flex;
    gap: 14px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.app-meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.88em;
    background: rgba(255,255,255,0.06);
    color: #F6EBDD;
    padding: 8px 13px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(4px);
}

.app-meta-dot {
    width: 7px;
    height: 7px;
    background: var(--color-accent);
    border-radius: 50%;
    display: inline-block;
}

/* TABS */
button[role="tab"] {
    color: var(--color-text-light) !important;
    background: transparent !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 700 !important;
}
button[role="tab"]:hover {
    background: rgba(212,165,116,0.08) !important;
    color: #FFF5EA !important;
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #8B5E3C 0%, #B07A52 100%) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(139,94,60,0.24) !important;
}

/* TITLES */
.panel-title {
    font-size: 1.08em;
    font-weight: 800;
    color: #FFF5EA;
    margin: 0 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
}
.panel-subtitle {
    font-size: 0.92em;
    color: var(--color-text-light);
    margin: -8px 0 16px 0;
    line-height: 1.6;
}

/* BOXES */
.helper-box,
.soft-note,
.empty-state,
.empty-state-initial,
.compare-container,
.about-section,
.stat-card-v2,
.vibe-stats-section,
.warkop-card {
    background: var(--bg-card) !important;
    color: var(--color-text);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
}

.helper-box {
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin: 0 0 16px 0;
    background: linear-gradient(180deg, #52392C 0%, #473126 100%) !important;
}
.helper-title {
    font-size: 0.82em;
    font-weight: 800;
    color: #F2D0A7;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.helper-step {
    font-size: 0.92em;
    color: var(--color-text);
    margin: 7px 0;
    line-height: 1.5;
}

.soft-note {
    border-left: 4px solid var(--color-accent);
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 0.9em;
    margin-bottom: 14px;
    line-height: 1.55;
    background: linear-gradient(180deg, #4A3328 0%, #422D23 100%) !important;
}

/* INPUTS */
input, textarea, select {
    background: var(--bg-soft) !important;
    color: var(--color-text) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}
textarea::placeholder,
input::placeholder {
    color: var(--color-muted) !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--color-accent) !important;
    box-shadow: 0 0 0 1px rgba(212,165,116,0.2) !important;
}

/* PRESET */
.preset-grid { gap: 8px !important; flex-wrap: wrap !important; }
.preset-grid button {
    flex: 1 !important;
    min-width: 110px !important;
    background: var(--bg-card-2) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--color-text) !important;
    padding: 12px 10px !important;
    font-size: 0.86em !important;
    font-weight: 700 !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    white-space: pre-line !important;
}
.preset-grid button:hover {
    border-color: var(--color-accent) !important;
    background: var(--bg-hover) !important;
    color: #FFF5EA !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}

/* STATUS */
.status-bar {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    font-size: 0.92em;
    font-weight: 600;
    margin: 12px 0;
    animation: slideIn 0.3s ease;
    border: 1px solid transparent;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.status-info {
    background: rgba(176,122,82,0.14);
    color: #F2D0A7;
    border-color: rgba(176,122,82,0.18);
}
.status-info .status-dot { background: #D4A574; }

.status-success {
    background: rgba(123,160,91,0.14);
    color: #D8F0C2;
    border-color: rgba(123,160,91,0.18);
}
.status-success .status-dot { background: #7BA05B; }

.status-warning {
    background: rgba(212,165,116,0.14);
    color: #F5D9B5;
    border-color: rgba(212,165,116,0.18);
}
.status-warning .status-dot { background: #D4A574; }

.status-error {
    background: rgba(201,123,99,0.14);
    color: #F3C6B7;
    border-color: rgba(201,123,99,0.18);
}
.status-error .status-dot { background: #C97B63; }

/* CARDS */
.cards-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
    animation: fadeIn 0.35s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.warkop-card {
    border-radius: var(--radius-md);
    padding: 18px 20px;
    transition: all 0.25s ease;
    position: relative;
    background: linear-gradient(180deg, #4A3328 0%, #412D23 100%) !important;
}
.warkop-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: var(--color-accent);
}

.card-header {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 8px;
}
.card-rank {
    background: linear-gradient(135deg, #B07A52 0%, #8B5E3C 100%);
    color: white;
    min-width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.92em;
    flex-shrink: 0;
}
.card-title-section { flex: 1; min-width: 0; }
.card-title {
    margin: 0 0 5px 0;
    font-size: 1.1em;
    font-weight: 800;
    color: #FFF5EA;
    line-height: 1.3;
}
.card-address {
    font-size: 0.86em;
    color: var(--color-text-light);
    display: flex;
    align-items: center;
    gap: 4px;
}
.card-address::before {
    content: '◉';
    color: var(--color-accent);
}
.card-summary {
    font-size: 0.9em;
    color: #EADACA;
    margin: 0 0 12px 0;
    line-height: 1.6;
}
.card-score {
    background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-dark) 100%);
    color: #3A281F;
    padding: 7px 12px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 0.96em;
    flex-shrink: 0;
    text-align: center;
}
.card-score small {
    font-size: 0.72em;
    font-weight: 700;
    margin-left: 1px;
}
.card-score.similarity {
    background: linear-gradient(135deg, #A67C52 0%, #8B5E3C 100%);
    color: white;
}

.card-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 14px;
}
.vibe-tag {
    background: #3A281F;
    color: #F2D0A7;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78em;
    font-weight: 700;
    border: 1px solid var(--border-color);
}

.card-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
}
.stat-item {
    display: flex;
    align-items: center;
    gap: 8px;
}
.stat-content { flex: 1; min-width: 0; }
.stat-label {
    font-size: 0.72em;
    color: var(--color-text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
    margin-bottom: 2px;
}

.stat-icon-wifi, .stat-icon-socket, .stat-icon-noise, .stat-icon-price {
    width: 22px;
    height: 22px;
    border-radius: 7px;
    flex-shrink: 0;
    position: relative;
    background: #3A281F;
    border: 1px solid var(--border-color);
}
.stat-icon-wifi::before {
    content: '';
    position: absolute;
    inset: 5px;
    border: 2px solid var(--color-accent);
    border-bottom: none;
    border-radius: 50% 50% 0 0;
}
.stat-icon-socket::before {
    content: '';
    position: absolute;
    inset: 6px;
    background: var(--color-accent);
    border-radius: 2px;
}
.stat-icon-noise::before {
    content: '';
    position: absolute;
    inset: 5px;
    border-left: 2px solid var(--color-accent);
}
.stat-icon-noise::after {
    content: '';
    position: absolute;
    top: 5px;
    left: 9px;
    width: 6px;
    height: 10px;
    border: 2px solid var(--color-accent);
    border-left: none;
    border-radius: 0 6px 6px 0;
}
.stat-icon-price::before {
    content: '$';
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: var(--color-accent);
    font-size: 0.9em;
}

.wifi-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 7px;
    font-size: 0.78em;
    font-weight: 800;
}
.wifi-fast   { background: rgba(123,160,91,0.16); color: #D8F0C2; }
.wifi-medium { background: rgba(212,165,116,0.16); color: #F5D9B5; }
.wifi-slow   { background: rgba(201,123,99,0.16); color: #F3C6B7; }

.level-bars {
    display: flex;
    gap: 3px;
    align-items: flex-end;
}
.level-bars .bar {
    width: 6px;
    background: #7A5A45;
    border-radius: 2px;
}
.level-bars .bar:nth-child(1) { height: 6px; }
.level-bars .bar:nth-child(2) { height: 10px; }
.level-bars .bar:nth-child(3) { height: 14px; }
.level-bars .bar.active { background: var(--color-accent); }

.price-badge {
    font-weight: 900;
    font-size: 0.95em;
    letter-spacing: -1px;
}
.price-cheap     { color: #D8F0C2; }
.price-medium    { color: #F5D9B5; }
.price-expensive { color: #F3C6B7; }

/* EMPTY */
.empty-state,
.empty-state-initial {
    text-align: center;
    padding: 50px 20px;
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, #4A3328 0%, #412D23 100%) !important;
}
.empty-state-initial {
    color: var(--color-text-light);
    font-size: 0.95em;
}
.empty-icon {
    width: 60px;
    height: 60px;
    background: #3A281F;
    border: 1px solid var(--border-color);
    border-radius: 50%;
    margin: 0 auto 16px;
    position: relative;
}
.empty-icon::before {
    content: '☕';
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}
.empty-title {
    font-size: 1.1em;
    font-weight: 800;
    color: #FFF5EA;
    margin-bottom: 6px;
}
.empty-subtitle {
    font-size: 0.9em;
    color: var(--color-text-light);
}

/* STATS */
.stats-hero {
    background:
        radial-gradient(circle at top right, rgba(212,165,116,0.14), transparent 30%),
        linear-gradient(135deg, #3A281F 0%, #52392C 100%);
    color: white;
    padding: 30px;
    border-radius: var(--radius-md);
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-light);
}
.stats-hero-number {
    font-size: 3.8em;
    font-weight: 900;
    line-height: 1;
    color: #F2D0A7;
    position: relative;
}
.stats-hero-label {
    font-size: 1em;
    opacity: 0.95;
    margin-top: 8px;
    position: relative;
    color: #EADACA;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
}
.stat-card-v2 {
    padding: 18px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.2s ease;
    background: linear-gradient(180deg, #4A3328 0%, #412D23 100%) !important;
}
.stat-card-v2:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.stat-card-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    flex-shrink: 0;
}
.stat-blue   { background: linear-gradient(135deg, #B07A52 0%, #8B5E3C 100%); }
.stat-green  { background: linear-gradient(135deg, #7BA05B 0%, #5E7E47 100%); }
.stat-amber  { background: linear-gradient(135deg, #D4A574 0%, #B98552 100%); }
.stat-purple { background: linear-gradient(135deg, #9C7A63 0%, #7B5A46 100%); }
.stat-teal   { background: linear-gradient(135deg, #7D9D8C 0%, #5F7C6E 100%); }

.stat-card-body { flex: 1; min-width: 0; }
.stat-card-label {
    font-size: 0.78em;
    color: var(--color-text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
    margin-bottom: 4px;
}
.stat-card-value {
    font-size: 1.6em;
    font-weight: 800;
    color: #FFF5EA;
    line-height: 1.1;
}
.stat-card-value small {
    font-size: 0.52em;
    color: var(--color-text-light);
    font-weight: 700;
}

.section-title {
    font-size: 1.12em;
    font-weight: 800;
    color: #FFF5EA;
    margin: 24px 0 16px 0;
}

.vibe-stats-section {
    padding: 20px;
    border-radius: var(--radius-md);
}
.vibe-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
}
.vibe-stat-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: #3A281F;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
}
.vibe-stat-name {
    font-weight: 700;
    color: var(--color-text);
    font-size: 0.9em;
}
.vibe-stat-count {
    background: linear-gradient(135deg, #B07A52 0%, #8B5E3C 100%);
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 0.85em;
}

/* COMPARE */
.compare-container {
    border-radius: var(--radius-md);
    overflow-x: auto;
}
.cmp-header {
    display: grid;
    grid-template-columns: 1fr 1.5fr 1.5fr;
    background: linear-gradient(135deg, #8B5E3C 0%, #B07A52 100%);
    color: white;
    padding: 14px 18px;
    font-weight: 800;
    gap: 16px;
    min-width: 680px;
}
.cmp-name-h { font-size: 1em; }
.cmp-body { padding: 8px 0; min-width: 680px; }
.cmp-row {
    display: grid;
    grid-template-columns: 1fr 1.5fr 1.5fr;
    padding: 12px 18px;
    gap: 16px;
    border-bottom: 1px solid var(--border-color);
    align-items: center;
}
.cmp-row:last-child { border-bottom: none; }
.cmp-row:nth-child(even) { background: rgba(255,255,255,0.02); }
.cmp-label {
    font-weight: 800;
    color: #F2D0A7;
    font-size: 0.88em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.cmp-val {
    color: var(--color-text);
    font-weight: 500;
}

/* BUTTON */
button.primary,
.action-btn button.primary,
button.lg.primary {
    background: linear-gradient(135deg, #8B5E3C 0%, #B07A52 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    padding: 12px 24px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(139, 94, 60, 0.24) !important;
}
button.primary:hover,
.action-btn button.primary:hover,
button.lg.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(139, 94, 60, 0.32) !important;
}

/* FOOTER */
.app-footer {
    text-align: center;
    padding: 26px 20px;
    color: var(--color-text-light);
    font-size: 0.9em;
    border: 1px solid var(--border-color);
    margin-top: 34px;
    background: var(--bg-card);
    border-radius: var(--radius-md);
}
.app-footer strong { color: #FFF5EA; }
.app-footer .footer-divider {
    display: inline-block;
    margin: 0 10px;
    color: var(--color-muted);
}

/* ABOUT */
.about-section {
    border-radius: var(--radius-md);
    padding: 24px;
    margin-bottom: 16px;
    background: linear-gradient(180deg, #4A3328 0%, #412D23 100%) !important;
}
.about-section h3 {
    font-size: 1.08em;
    font-weight: 800;
    color: #FFF5EA;
    margin: 0 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
}
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
}
.about-item {
    display: flex;
    gap: 12px;
    padding: 14px;
    background: #3A281F;
    border-radius: var(--radius-sm);
    align-items: flex-start;
    border: 1px solid var(--border-color);
}
.about-item-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, #B07A52 0%, #8B5E3C 100%);
    flex-shrink: 0;
}
.about-item-content .about-item-title {
    font-weight: 800;
    font-size: 0.92em;
    color: #FFF5EA;
    margin-bottom: 3px;
}
.about-item-content .about-item-desc {
    font-size: 0.84em;
    color: var(--color-text-light);
    line-height: 1.5;
}
.algo-box {
    background: #241813;
    color: #F2D0A7;
    padding: 16px 20px;
    border-radius: var(--radius-sm);
    font-family: 'Courier New', monospace;
    font-size: 0.85em;
    line-height: 1.7;
    margin: 12px 0;
    overflow-x: auto;
    border: 1px solid var(--border-color);
}
.algo-box .algo-comment { color: #B89B83; }
.algo-box .algo-var { color: #E6C29A; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #3A281F; }
::-webkit-scrollbar-thumb { background: #7A5440; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8B5E3C; }

/* RESPONSIVE */
@media (max-width: 900px) {
    .app-title { font-size: 1.55em; }
    .card-stats { grid-template-columns: repeat(2, 1fr); }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .app-header { padding: 24px 18px; }
    .app-logo { width: 46px; height: 46px; font-size: 22px; }
    .app-title { font-size: 1.3em; }
    .app-subtitle { font-size: 0.9em; }
    .card-stats { grid-template-columns: repeat(2, 1fr); }
    .stats-hero-number { font-size: 3em; }
    .stats-grid { grid-template-columns: 1fr 1fr; }
    .vibe-stats-grid { grid-template-columns: 1fr; }
}
"""


# =============================================================================
# About Content Helper
# =============================================================================
def _render_about() -> str:
    features = [
        ("Cari Warkop", "Temukan saran warkop berdasarkan kebutuhan seperti WiFi, harga, colokan, dan suasana."),
        ("Cerita Kebutuhanmu", "Tulis kebutuhanmu dengan bahasa santai, lalu sistem akan mencocokkan hasilnya."),
        ("Cari yang Lebih Pas", "Gabungkan deskripsi bebas dan preferensi agar hasilnya lebih personal."),
        ("Pilih Suasana", "Temukan tempat berdasarkan vibe atau suasana yang kamu inginkan."),
        ("Mirip Warkop Favorit", "Pilih satu warkop favorit, lalu cari alternatif yang mirip."),
        ("Cari per Kawasan", "Jelajahi warkop berdasarkan area atau alamat tertentu."),
        ("Bandingkan Tempat", "Bandingkan dua tempat agar lebih mudah memilih."),
        ("Info Warkop", "Lihat ringkasan statistik seluruh data warkop."),
        ("Daftar Warkop", "Tampilkan semua warkop dan urutkan sesuai kebutuhan."),
    ]

    items_html = "".join(
        f"""
        <div class="about-item">
            <div class="about-item-icon"></div>
            <div class="about-item-content">
                <div class="about-item-title">{escape(title)}</div>
                <div class="about-item-desc">{escape(desc)}</div>
            </div>
        </div>
        """
        for title, desc in features
    )

    tech_rows = [
        ("TF-IDF Vectorizer", "Mengubah teks menjadi representasi numerik agar bisa dibandingkan."),
        ("MinMax Scaler", "Menormalkan fitur numerik ke rentang yang konsisten."),
        ("Cosine Similarity", "Mengukur kemiripan antar teks atau item."),
        ("Content-Based Filtering", "Memberi rekomendasi berdasarkan karakteristik tempat."),
        ("Caching", "Membantu mempercepat pemrosesan query berulang."),
        ("Gradio Blocks", "Membangun antarmuka aplikasi interaktif."),
    ]

    tech_html = "".join(
        f"""
        <div class="cmp-row">
            <div class="cmp-label">{escape(tech)}</div>
            <div class="cmp-val" style="grid-column: span 2;">{escape(func)}</div>
        </div>
        """
        for tech, func in tech_rows
    )

    total = len(_safe_df())

    return f"""
    <div class="about-section">
        <h3>Tentang Aplikasi</h3>
        <p style="color:var(--color-text-light); font-size:0.95em; line-height:1.7; margin:0 0 16px 0;">
            <strong>{escape(APP_TITLE)}</strong> adalah sistem rekomendasi cerdas
            untuk membantu pengguna menemukan warkop terbaik di Lhokseumawe
            sesuai kebutuhan pribadi, baik untuk nugas, nongkrong, meeting, maupun santai.
        </p>
        <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:8px;">
            <div style="background:var(--bg-soft); padding:12px 20px; border-radius:var(--radius-sm); text-align:center; border:1px solid var(--border-color);">
                <div style="font-size:1.8em; font-weight:800; color:var(--color-accent);">{total}</div>
                <div style="font-size:0.8em; color:var(--color-text-light);">Total Warkops</div>
            </div>
            <div style="background:var(--bg-soft); padding:12px 20px; border-radius:var(--radius-sm); text-align:center; border:1px solid var(--border-color);">
                <div style="font-size:1.8em; font-weight:800; color:var(--color-accent);">v4.2</div>
                <div style="font-size:0.8em; color:var(--color-text-light);">Versi</div>
            </div>
            <div style="background:var(--bg-soft); padding:12px 20px; border-radius:var(--radius-sm); text-align:center; border:1px solid var(--border-color);">
                <div style="font-size:1.8em; font-weight:800; color:var(--color-accent);">9+</div>
                <div style="font-size:0.8em; color:var(--color-text-light);">Fitur</div>
            </div>
        </div>
    </div>

    <div class="about-section">
        <h3>Fitur Utama</h3>
        <div class="about-grid">{items_html}</div>
    </div>

    <div class="about-section">
        <h3>Algoritma</h3>
        <div class="algo-box">
<span class="algo-comment"># Numeric Score</span>
<span class="algo-var">score</span> = w_wifi   * wifi_normalized
      + w_socket * socket_normalized
      + w_quiet  * (1 - noise_normalized)
      + w_value  * (1 - price_normalized)

<span class="algo-comment"># Hybrid Score</span>
<span class="algo-var">final</span> = text_ratio * cosine_sim(query, metadata)
      + (1 - text_ratio) * numeric_score

<span class="algo-comment"># Similarity</span>
<span class="algo-var">sim</span> = (cosine_sim_numeric + cosine_sim_text) / 2
        </div>
    </div>

    <div class="about-section">
        <h3>Stack Teknologi</h3>
        <div class="compare-container">
            <div class="cmp-header">
                <div class="cmp-label-h">Teknologi</div>
                <div class="cmp-name-h" style="grid-column: span 2;">Fungsi</div>
            </div>
            <div class="cmp-body">{tech_html}</div>
        </div>
    </div>

    <div class="about-section">
        <h3>Schema Data</h3>
        <div class="compare-container">
            <div class="cmp-header">
                <div class="cmp-label-h">Kolom</div>
                <div class="cmp-name-h">Tipe</div>
                <div class="cmp-name-h">Nilai Valid</div>
            </div>
            <div class="cmp-body">
                <div class="cmp-row"><div class="cmp-label">name</div><div class="cmp-val">string</div><div class="cmp-val">Nama warkop</div></div>
                <div class="cmp-row"><div class="cmp-label">address</div><div class="cmp-val">string</div><div class="cmp-val">Alamat / kawasan</div></div>
                <div class="cmp-row"><div class="cmp-label">wifi_speed_mbps</div><div class="cmp-val">number</div><div class="cmp-val">0 - 100+</div></div>
                <div class="cmp-row"><div class="cmp-label">socket_availability</div><div class="cmp-val">enum</div><div class="cmp-val">Low / Medium / High</div></div>
                <div class="cmp-row"><div class="cmp-label">noise_level</div><div class="cmp-val">enum</div><div class="cmp-val">Low / Medium / High</div></div>
                <div class="cmp-row"><div class="cmp-label">vibe_category</div><div class="cmp-val">string</div><div class="cmp-val">Tag slash-separated</div></div>
                <div class="cmp-row"><div class="cmp-label">price_range</div><div class="cmp-val">enum</div><div class="cmp-val">Cheap / Medium / Expensive</div></div>
            </div>
        </div>
    </div>
    """


# =============================================================================
# Build App
# =============================================================================
def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(
            f"""
            <div class="app-header">
                <div class="app-header-content">
                    <div class="app-brand">
                        <div class="app-logo">☕</div>
                        <div>
                            <div class="app-title">Cari Warkop yang Paling Pas Buat Kamu</div>
                            <div class="app-subtitle">
                                Temukan tempat nyaman untuk ngopi, nugas, nongkrong, meeting, atau santai di Lhokseumawe.
                            </div>
                        </div>
                    </div>
                    <div class="app-meta">
                        <div class="app-meta-item">
                            <span class="app-meta-dot"></span>
                            {len(_safe_df())} warkop tersedia
                        </div>
                        <div class="app-meta-item">
                            <span class="app-meta-dot"></span>
                            Suasana hangat & nyaman
                        </div>
                        <div class="app-meta-item">
                            <span class="app-meta-dot"></span>
                            Cocok untuk mahasiswa, pekerja, dan nongkrong
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        if recommender is None:
            gr.HTML(_no_recommender_msg())

        with gr.Tabs():
            with gr.Tab("Cari Warkop"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Atur Kebutuhan Kamu</div>')
                        gr.HTML('<div class="panel-subtitle">Pilih yang paling penting buat kamu, lalu kami bantu carikan warkop yang cocok.</div>')

                        gr.HTML("""
                        <div class="helper-box">
                            <div class="helper-title">Cara pakai</div>
                            <div class="helper-step">1. Pilih preset cepat atau atur slider sendiri</div>
                            <div class="helper-step">2. Tambahkan filter harga atau suasana kalau perlu</div>
                            <div class="helper-step">3. Klik <b>Cari Sekarang</b></div>
                        </div>
                        """)

                        gr.HTML("<b style='font-size:0.88em;color:var(--color-text-light);'>PRESET CEPAT</b>")
                        with gr.Row(elem_classes="preset-grid"):
                            preset_buttons = {}
                            for key, prof in PRESET_PROFILES.items():
                                preset_buttons[key] = gr.Button(
                                    f"{prof['label']}\n{prof['subtitle']}",
                                    size="sm",
                                )

                        gr.HTML("<hr style='border:1px solid var(--border-color); margin:16px 0;'>")
                        gr.HTML("<b style='font-size:0.88em;color:var(--color-text-light);'>ATUR MANUAL</b>")

                        wifi_s = gr.Slider(0, 1, value=0.45, step=0.05, label="Internet cepat")
                        socket_s = gr.Slider(0, 1, value=0.30, step=0.05, label="Colokan tersedia")
                        quiet_s = gr.Slider(0, 1, value=0.20, step=0.05, label="Suasana tenang")
                        value_s = gr.Slider(0, 1, value=0.05, step=0.05, label="Harga terjangkau")

                        gr.HTML("<hr style='border:1px solid var(--border-color); margin:16px 0;'>")
                        gr.HTML("<b style='font-size:0.88em;color:var(--color-text-light);'>FILTER TAMBAHAN</b>")

                        with gr.Row():
                            price_f = gr.Dropdown(
                                choices=["Semua"] + PRICE_RANGES,
                                value="Semua",
                                label="Rentang harga",
                            )
                            vibe_f = gr.Dropdown(
                                choices=["Semua"] + VIBE_TAGS,
                                value="Semua",
                                label="Suasana",
                            )

                        top_n_s = gr.Slider(3, 23, value=7, step=1, label="Jumlah hasil")
                        rec_btn = gr.Button("Cari Sekarang", variant="primary", size="lg")

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Hasil Rekomendasi</div>')
                        smart_status = gr.HTML(_empty_initial_msg("Atur preferensimu lalu klik tombol cari."))
                        smart_out = gr.HTML()

                for key, btn in preset_buttons.items():
                    btn.click(
                        fn=lambda k=key: use_preset(k),
                        inputs=[],
                        outputs=[wifi_s, socket_s, quiet_s, value_s],
                    )

                rec_btn.click(
                    fn=smart_recommend,
                    inputs=[wifi_s, socket_s, quiet_s, value_s, price_f, vibe_f, top_n_s],
                    outputs=[smart_out, smart_status],
                    show_progress="minimal",
                )

            with gr.Tab("Cerita Kebutuhanmu"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Ceritakan Warkop yang Kamu Cari</div>')
                        gr.HTML("""
                        <div class="panel-subtitle">
                            Tulis dengan bahasa santai, seperti sedang cerita ke teman.
                        </div>
                        """)

                        gr.HTML("""
                        <div class="soft-note">
                            Contoh: "Saya cari tempat tenang buat kerja, internet cepat, dan colokan banyak."
                        </div>
                        """)

                        story_in = gr.Textbox(
                            label="Deskripsi pencarian",
                            placeholder="Contoh: saya cari tempat yang tenang, internet cepat, cocok buat nugas sampai malam...",
                            lines=5,
                        )
                        story_n = gr.Slider(3, 10, value=5, step=1, label="Jumlah hasil")
                        story_btn = gr.Button("Cari dari Cerita", variant="primary", size="lg")

                        gr.HTML("<hr style='border:1px solid var(--border-color); margin:16px 0;'>")
                        gr.HTML("<b style='font-size:0.88em;color:var(--color-text-light);'>CONTOH PENCARIAN</b>")

                        example_queries = [
                            "warkop tenang wifi cepat buat coding",
                            "tempat nongkrong murah kumpul teman",
                            "coffee shop modern yang nyaman",
                            "warkop buat begadang ngerjain tugas",
                            "tempat santai yang colokannya banyak",
                            "warkop premium buat meeting",
                        ]

                        with gr.Column(elem_classes="preset-grid"):
                            for ex in example_queries:
                                ex_btn = gr.Button(ex, size="sm")
                                ex_btn.click(fn=lambda x=ex: x, inputs=[], outputs=story_in)

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Hasil Pencarian</div>')
                        story_status = gr.HTML(_empty_initial_msg("Tulis kebutuhanmu, lalu klik tombol cari."))
                        story_out = gr.HTML()

                story_btn.click(
                    fn=story_search,
                    inputs=[story_in, story_n],
                    outputs=[story_out, story_status],
                    show_progress="minimal",
                )
                story_in.submit(
                    fn=story_search,
                    inputs=[story_in, story_n],
                    outputs=[story_out, story_status],
                    show_progress="minimal",
                )

            with gr.Tab("Cari yang Lebih Pas"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Gabungkan Cerita dan Preferensi</div>')
                        gr.HTML("""
                        <div class="panel-subtitle">
                            Cocok kalau kamu ingin hasil yang lebih personal: deskripsi + prioritas.
                        </div>
                        """)

                        gr.HTML("""
                        <div class="helper-box">
                            <div class="helper-title">Cocok dipakai kalau</div>
                            <div class="helper-step">• Kamu punya gambaran tempat yang diinginkan</div>
                            <div class="helper-step">• Kamu juga ingin mengatur prioritas seperti WiFi, harga, atau suasana</div>
                        </div>
                        """)

                        h_query = gr.Textbox(
                            label="Deskripsi (opsional)",
                            placeholder="Contoh: warkop modern, tenang, cocok buat kerja di area Pase",
                            lines=4,
                        )

                        gr.HTML("<b style='font-size:0.88em;color:var(--color-text-light);'>PRIORITAS KAMU</b>")
                        with gr.Row():
                            h_wifi = gr.Slider(0, 1, value=0.35, step=0.05, label="Internet")
                            h_socket = gr.Slider(0, 1, value=0.25, step=0.05, label="Colokan")
                        with gr.Row():
                            h_quiet = gr.Slider(0, 1, value=0.20, step=0.05, label="Ketenangan")
                            h_value = gr.Slider(0, 1, value=0.20, step=0.05, label="Harga")

                        gr.HTML("<hr style='border:1px solid var(--border-color); margin:16px 0;'>")

                        ratio_s = gr.Slider(
                            0,
                            1,
                            value=0.5,
                            step=0.05,
                            label="Fokus Pencarian",
                            info="Geser ke kiri untuk fokus preferensi, ke kanan untuk fokus deskripsi",
                        )
                        h_top = gr.Slider(3, 15, value=7, step=1, label="Jumlah hasil")
                        h_btn = gr.Button("Cari yang Lebih Pas", variant="primary", size="lg")

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Hasil Pencarian</div>')
                        h_status = gr.HTML(_empty_initial_msg("Isi kebutuhanmu lalu klik tombol cari."))
                        h_out = gr.HTML()

                h_btn.click(
                    fn=hybrid_search,
                    inputs=[h_query, h_wifi, h_socket, h_quiet, h_value, ratio_s, h_top],
                    outputs=[h_out, h_status],
                    show_progress="minimal",
                )

            with gr.Tab("Pilih Suasana"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Pilih Suasana yang Kamu Mau</div>')
                        gr.HTML("""
                        <div class="panel-subtitle">
                            Cocok kalau kamu sudah tahu vibe tempat yang diinginkan.
                        </div>
                        """)
                        vibe_dd = gr.Dropdown(choices=VIBE_TAGS, label="Pilih suasana")
                        vibe_n = gr.Slider(3, 20, value=8, step=1, label="Jumlah hasil")
                        vibe_btn = gr.Button("Tampilkan Hasil", variant="primary", size="lg")

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Hasil</div>')
                        vibe_status = gr.HTML(_empty_initial_msg("Pilih suasana lalu klik tombol tampilkan."))
                        vibe_out = gr.HTML()

                vibe_btn.click(
                    fn=vibe_search,
                    inputs=[vibe_dd, vibe_n],
                    outputs=[vibe_out, vibe_status],
                    show_progress="minimal",
                )

            with gr.Tab("Mirip Warkop Favorit"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Cari Warkop yang Mirip</div>')
                        gr.HTML("""
                        <div class="panel-subtitle">
                            Sudah punya warkop favorit? Pilih di bawah, lalu kami carikan yang mirip.
                        </div>
                        """)
                        sim_dd = gr.Dropdown(choices=WARKOP_NAMES, label="Pilih warkop referensi")
                        sim_n = gr.Slider(3, 10, value=5, step=1, label="Jumlah hasil")
                        sim_btn = gr.Button("Cari yang Mirip", variant="primary", size="lg")

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Warkop Serupa</div>')
                        sim_status = gr.HTML(_empty_initial_msg("Pilih satu warkop referensi terlebih dahulu."))
                        sim_out = gr.HTML()

                sim_btn.click(
                    fn=find_similar,
                    inputs=[sim_dd, sim_n],
                    outputs=[sim_out, sim_status],
                    show_progress="minimal",
                )

            with gr.Tab("Cari per Kawasan"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        gr.HTML('<div class="panel-title">Cari Berdasarkan Lokasi</div>')
                        gr.HTML("""
                        <div class="panel-subtitle">
                            Lihat warkop berdasarkan kawasan atau alamat tertentu.
                        </div>
                        """)
                        addr_dd = gr.Dropdown(choices=ADDRESS_LIST, label="Pilih kawasan")
                        addr_n = gr.Slider(3, 20, value=10, step=1, label="Jumlah hasil")
                        addr_btn = gr.Button("Cari Lokasi", variant="primary", size="lg")

                    with gr.Column(scale=3, min_width=420):
                        gr.HTML('<div class="panel-title">Hasil</div>')
                        addr_status = gr.HTML(_empty_initial_msg("Pilih kawasan yang ingin kamu lihat."))
                        addr_out = gr.HTML()

                addr_btn.click(
                    fn=filter_by_address,
                    inputs=[addr_dd, addr_n],
                    outputs=[addr_out, addr_status],
                    show_progress="minimal",
                )

            with gr.Tab("Bandingkan Tempat"):
                gr.HTML('<div class="panel-title">Bandingkan Dua Warkop</div>')
                gr.HTML("""
                <div class="panel-subtitle">
                    Cocok untuk membantu memilih tempat sebelum berangkat.
                </div>
                """)
                with gr.Row():
                    cmp_a = gr.Dropdown(choices=WARKOP_NAMES, label="Warkop pertama")
                    cmp_b = gr.Dropdown(choices=WARKOP_NAMES, label="Warkop kedua")

                cmp_btn = gr.Button("Bandingkan Sekarang", variant="primary", size="lg")
                cmp_out = gr.HTML(_empty_initial_msg("Pilih dua warkop yang ingin dibandingkan."))

                cmp_btn.click(
                    fn=compare_warkops,
                    inputs=[cmp_a, cmp_b],
                    outputs=cmp_out,
                )

            with gr.Tab("Info Warkop"):
                gr.HTML('<div class="panel-title">Ringkasan Data Warkop</div>')
                gr.HTML("""
                <div class="panel-subtitle">
                    Lihat gambaran umum seluruh data warkop yang tersedia.
                </div>
                """)
                stats_btn = gr.Button("Muat Statistik", variant="primary", size="lg")
                stats_out = gr.HTML(_empty_initial_msg("Klik tombol untuk menampilkan statistik."))

                stats_btn.click(
                    fn=get_stats,
                    inputs=[],
                    outputs=stats_out,
                    show_progress="minimal",
                )

            with gr.Tab("Daftar Warkop"):
                gr.HTML('<div class="panel-title">Lihat Semua Warkop</div>')
                gr.HTML("""
                <div class="panel-subtitle">
                    Jelajahi semua warkop dan urutkan sesuai kebutuhanmu.
                </div>
                """)
                with gr.Row():
                    sort_dd = gr.Dropdown(
                        choices=[
                            "name",
                            "wifi_speed_mbps",
                            "price_range",
                            "noise_level",
                            "socket_availability",
                            "vibe_category",
                            "address",
                        ],
                        value="name",
                        label="Urutkan berdasarkan",
                        scale=2,
                    )
                    asc_cb = gr.Checkbox(value=True, label="Urutan naik", scale=1)

                all_btn = gr.Button("Lihat Semua Warkop", variant="primary", size="lg")
                all_out = gr.HTML(_empty_initial_msg("Klik tombol untuk menampilkan semua data."))

                all_btn.click(
                    fn=display_all,
                    inputs=[sort_dd, asc_cb],
                    outputs=all_out,
                    show_progress="minimal",
                )

            with gr.Tab("Tentang"):
                gr.HTML(_render_about())

        gr.HTML(
            """
            <div class="app-footer">
                <strong>Warkop Intelligence Finder</strong>
                <span class="footer-divider">|</span>
                Dibuat untuk membantu menemukan tempat yang sesuai kebutuhan pengguna
                <span class="footer-divider">|</span>
                <strong>Politeknik Negeri Lhokseumawe</strong>
            </div>
            """
        )

    return app


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    app = build_app()
    app.queue(
        max_size=20,
        default_concurrency_limit=4,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )