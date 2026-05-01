"""
Warkop Intelligence Finder - Lhokseumawe Edition v3.0
Gradio-based UI for ML-powered warkop recommendations
"""

import gradio as gr
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

from models.recommender import WarkopRecommender, WarkopRecommenderError

# =============================================================================
# Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "warkops.csv"

PRESET_PROFILES = {
    "coding": {
        "label": "Coding & Deep Work",
        "weights": {"wifi_imp": 0.45, "socket_imp": 0.30, "quiet_imp": 0.20, "value_imp": 0.05},
        "desc": "WiFi stabil, banyak colokan, suasana tenang"
    },
    "social": {
        "label": "Social & Hangout",
        "weights": {"wifi_imp": 0.15, "socket_imp": 0.20, "quiet_imp": 0.05, "value_imp": 0.60},
        "desc": "Suasana ramai, harga terjangkau, cocok berkumpul"
    },
    "student": {
        "label": "Student Mode",
        "weights": {"wifi_imp": 0.35, "socket_imp": 0.25, "quiet_imp": 0.10, "value_imp": 0.30},
        "desc": "Balance antara WiFi, colokan, dan budget mahasiswa"
    },
    "premium": {
        "label": "Premium Experience",
        "weights": {"wifi_imp": 0.40, "socket_imp": 0.25, "quiet_imp": 0.30, "value_imp": 0.05},
        "desc": "Kualitas terbaik, harga bukan masalah"
    },
    "chill": {
        "label": "Chill & Quiet",
        "weights": {"wifi_imp": 0.10, "socket_imp": 0.15, "quiet_imp": 0.45, "value_imp": 0.30},
        "desc": "Tempat tenang untuk santai dan bersantai"
    },
}


# =============================================================================
# Initialize Recommender
# =============================================================================
def initialize_recommender() -> Optional[WarkopRecommender]:
    """Safely initialize recommender with error handling."""
    try:
        rec = WarkopRecommender(str(CSV_PATH))
        logger.info(f"Loaded {len(rec)} warkops successfully")
        return rec
    except Exception as e:
        logger.exception(f"Failed to init recommender: {e}")
        return None


recommender = initialize_recommender()

# Dropdown options
VIBE_TAGS = recommender.get_all_vibe_tags() if recommender else []
PRICE_RANGES = ['Cheap', 'Medium', 'Expensive']
NOISE_LEVELS = ['Low', 'Medium', 'High']
SOCKET_LEVELS = ['Low', 'Medium', 'High']
WARKOP_NAMES = sorted(recommender.df['name'].tolist()) if recommender else []
ADDRESS_LIST = sorted(recommender.df['address'].unique().tolist()) if recommender else []


# =============================================================================
# Helper Functions
# =============================================================================
def _check_recommender() -> Optional[str]:
    """Return error message if recommender not loaded."""
    if recommender is None:
        return ("### System Error\n\n"
                "Recommender belum berhasil dimuat.\n\n"
                "Pastikan file `data/warkops.csv` tersedia dan formatnya valid.")
    return None


def _format_results(df: pd.DataFrame, title: str = "") -> str:
    """Format DataFrame into clean markdown."""
    if df is None or df.empty:
        return "### Tidak ada hasil ditemukan\n\nCoba ubah filter atau preferensi Anda."

    priority_cols = [
        'name', 'address', 'wifi_speed_mbps', 'socket_availability',
        'noise_level', 'price_range', 'vibe_category',
        'match_score', 'similarity_score'
    ]
    display_cols = [c for c in priority_cols if c in df.columns]

    rename_map = {
        'name': 'Nama Warkop',
        'address': 'Alamat',
        'wifi_speed_mbps': 'WiFi (Mbps)',
        'socket_availability': 'Colokan',
        'noise_level': 'Noise',
        'price_range': 'Harga',
        'vibe_category': 'Vibe',
        'match_score': 'Score (%)',
        'similarity_score': 'Similarity (%)',
    }

    display_df = df[display_cols].rename(columns=rename_map).reset_index(drop=True)
    display_df.index = display_df.index + 1

    header = f"### {title}\n\n" if title else ""
    return header + display_df.to_markdown(index=True)


# =============================================================================
# Tab Handlers
# =============================================================================
def smart_recommend(wifi, socket, quiet, value, price_filter, vibe_filter, top_n):
    """Custom recommendation with sliders + filters."""
    if err := _check_recommender():
        return err
    try:
        filters = {}
        if price_filter and price_filter != "Semua":
            filters['price_range'] = price_filter
        if vibe_filter and vibe_filter != "Semua":
            filters['vibe_category'] = vibe_filter

        result = recommender.recommend_by_weights(
            wifi_imp=wifi, socket_imp=socket,
            quiet_imp=quiet, value_imp=value,
            top_n=int(top_n),
            filters=filters or None
        )
        return _format_results(result, f"Top {len(result)} Rekomendasi Personal")
    except Exception as e:
        logger.exception("Error in smart_recommend")
        return f"### Error\n\n```\n{e}\n```"


def use_preset(preset_key: str) -> Tuple[float, float, float, float]:
    """Apply preset weights to sliders."""
    w = PRESET_PROFILES[preset_key]["weights"]
    return w["wifi_imp"], w["socket_imp"], w["quiet_imp"], w["value_imp"]


def story_search(query, top_n):
    """NLP-based semantic search."""
    if err := _check_recommender():
        return err
    if not query or not query.strip():
        return "### Tulis cerita atau deskripsi pencarian Anda terlebih dahulu"
    try:
        result = recommender.recommend_by_story(query, top_n=int(top_n))
        return _format_results(result, f"Hasil pencarian: \"{query}\"")
    except Exception as e:
        logger.exception("Error in story_search")
        return f"### Error\n\n```\n{e}\n```"


def hybrid_search(query, wifi, socket, quiet, value, ratio, top_n):
    """Hybrid recommendation combining text + numeric."""
    if err := _check_recommender():
        return err
    try:
        result = recommender.recommend_hybrid(
            user_query=query or "",
            weights={"wifi_imp": wifi, "socket_imp": socket,
                     "quiet_imp": quiet, "value_imp": value},
            text_ratio=float(ratio),
            top_n=int(top_n)
        )
        return _format_results(result, "Hybrid AI Recommendation")
    except Exception as e:
        logger.exception("Error in hybrid_search")
        return f"### Error\n\n```\n{e}\n```"


def vibe_search(vibe_tag, top_n):
    """Filter by vibe tag."""
    if err := _check_recommender():
        return err
    if not vibe_tag:
        return "### Pilih kategori vibe terlebih dahulu"
    try:
        result = recommender.recommend_by_weights(
            top_n=int(top_n),
            filters={'vibe_category': vibe_tag}
        )
        return _format_results(result, f"Warkop dengan Vibe: {vibe_tag}")
    except Exception as e:
        return f"### Error\n\n{e}"


def find_similar(warkop_name, top_n):
    """Find similar warkops."""
    if err := _check_recommender():
        return err
    if not warkop_name:
        return "### Pilih warkop referensi terlebih dahulu"
    try:
        result = recommender.find_similar(warkop_name, top_n=int(top_n))
        return _format_results(result, f"Warkop Mirip dengan: {warkop_name}")
    except WarkopRecommenderError as e:
        return f"### {e}"
    except Exception as e:
        logger.exception("Error in find_similar")
        return f"### Error\n\n{e}"


def filter_by_address(address, top_n):
    """Filter warkops by address/area."""
    if err := _check_recommender():
        return err
    if not address:
        return "### Pilih alamat/kawasan terlebih dahulu"
    df = recommender.df.copy()
    result = df[df['address'].str.contains(address, case=False, na=False)]
    if result.empty:
        return f"### Tidak ada warkop di kawasan '{address}'"
    result = result.head(int(top_n))
    return _format_results(result, f"Warkop di Kawasan: {address}")


def get_stats():
    """Generate statistics dashboard."""
    if err := _check_recommender():
        return err
    try:
        s = recommender.get_quick_stats()
        top_vibes = "\n".join(
            [f"- **{k}**: `{v}` warkop" for k, v in s['Top Vibe Tags'].items()]
        )

        return f"""
## Database Statistics - Warkop Lhokseumawe

### Overview

| Metric | Value |
|--------|-------|
| Total Warkops | `{s['Total Warkop']}` |
| Unique Vibe Tags | `{s['Total Unique Vibes']}` |

### WiFi Performance

| Metric | Value |
|--------|-------|
| Tercepat | `{s['WiFi Tercepat (Mbps)']:.0f} Mbps` |
| Rata-rata | `{s['WiFi Rata-rata (Mbps)']:.1f} Mbps` |
| Terlambat | `{s['WiFi Terlambat (Mbps)']:.0f} Mbps` |

### Special Features

| Feature | Count |
|---------|-------|
| Spot Tenang (Low Noise) | `{s['Spot Paling Tenang']}` |
| Spot Ramai (High Noise) | `{s['Spot Bising']}` |
| Budget Friendly (Cheap) | `{s['Budget Friendly']}` |
| Premium (Expensive) | `{s['Premium']}` |
| Colokan Melimpah (High) | `{s['Socket Melimpah']}` |
| Colokan Sedikit (Low) | `{s['Socket Sedikit']}` |

### Top Vibe Tags

{top_vibes}

---
*Powered by Content-Based Filtering + TF-IDF Vectorization*
"""
    except Exception as e:
        logger.exception("Error in get_stats")
        return f"### Error\n\n{e}"


def display_all(sort_by, ascending):
    """Display complete database with sorting."""
    if err := _check_recommender():
        return err
    try:
        df = recommender.df.copy()
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        df = df.drop(columns=[c for c in ['metadata'] if c in df.columns])
        return _format_results(df, f"Database Lengkap (sorted by `{sort_by}`)")
    except Exception as e:
        return f"### Error\n\n{e}"


def compare_warkops(warkop_a, warkop_b):
    """Compare two warkops side-by-side."""
    if err := _check_recommender():
        return err
    if not warkop_a or not warkop_b:
        return "### Pilih dua warkop untuk dibandingkan"
    if warkop_a == warkop_b:
        return "### Pilih dua warkop yang berbeda"

    df = recommender.df
    row_a = df[df['name'] == warkop_a].iloc[0]
    row_b = df[df['name'] == warkop_b].iloc[0]

    cols = ['address', 'wifi_speed_mbps', 'socket_availability',
            'noise_level', 'price_range', 'vibe_category']
    labels = ['Alamat', 'WiFi (Mbps)', 'Colokan', 'Noise', 'Harga', 'Vibe']

    table = f"| Atribut | {warkop_a} | {warkop_b} |\n|---------|---|---|\n"
    for col, label in zip(cols, labels):
        table += f"| **{label}** | {row_a[col]} | {row_b[col]} |\n"

    return f"### Perbandingan Warkop\n\n{table}"


# =============================================================================
# Custom CSS
# =============================================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }

.header-section {
    background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%);
    color: white;
    padding: 45px 30px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(62, 39, 35, 0.3);
    text-align: center;
}
.header-section h1 {
    margin: 0 0 12px 0;
    font-size: 2.4em;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.header-section .tagline {
    font-size: 1.05em;
    opacity: 0.95;
    margin: 6px 0;
}
.header-section .badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85em;
    margin: 8px 4px 0 4px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.info-box {
    background: linear-gradient(135deg, #FFF8F0 0%, #FFEEDD 100%);
    border-left: 4px solid #5D4037;
    padding: 16px 22px;
    margin: 18px 0;
    border-radius: 8px;
    font-size: 0.95em;
    color: #333;
}
.info-box strong { color: #5D4037; }

.footer-section {
    text-align: center;
    padding: 30px 20px;
    color: #666;
    font-size: 0.9em;
    border-top: 2px solid #EEE;
    margin-top: 40px;
}
.footer-section strong { color: #5D4037; }
"""


# =============================================================================
# Build Gradio App
# =============================================================================
def build_app() -> gr.Blocks:
    """Build the complete Gradio application."""

    with gr.Blocks(
        title="Warkop Intelligence Finder - Lhokseumawe",
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange"),
        css=CUSTOM_CSS
    ) as app:

        # ===== Header =====
        gr.HTML("""
        <div class="header-section">
            <h1>Warkop Intelligence Finder</h1>
            <p class="tagline">Temukan warkop terbaik di Lhokseumawe sesuai kebutuhan Anda</p>
            <p class="tagline" style="font-size: 0.95em;">Powered by Hybrid Machine Learning - Content-Based + Semantic Search</p>
            <span class="badge">AI-Driven</span>
            <span class="badge">Data-Backed</span>
            <span class="badge">Local Lhokseumawe</span>
            <span class="badge">Real-time</span>
        </div>
        """)

        # ===== System Status Warning =====
        if recommender is None:
            gr.HTML("""
            <div style="background:#FEE; border-left:4px solid #C00; padding:15px; border-radius:8px; margin-bottom:20px;">
                <strong>System Warning:</strong> Recommender gagal dimuat.
                Pastikan file <code>data/warkops.csv</code> tersedia dan valid.
            </div>
            """)

        # ===== Tabs =====
        with gr.Tabs():

            # ----------------- Tab 1: Smart Recommendation -----------------
            with gr.Tab("Smart Recommendation"):
                gr.Markdown("### Rekomendasi Personal dengan Preset & Filter")
                gr.HTML("""
                <div class="info-box">
                    <strong>Tips:</strong> Klik salah satu preset untuk auto-fill slider,
                    atau atur manual sesuai kebutuhan. Tambahkan filter untuk hasil lebih spesifik.
                </div>
                """)

                gr.Markdown("**Quick Presets:**")
                with gr.Row():
                    preset_buttons = {}
                    for key, prof in PRESET_PROFILES.items():
                        preset_buttons[key] = gr.Button(prof["label"], size="sm")

                gr.Markdown("**Atur Preferensi (otomatis dinormalisasi):**")
                with gr.Row():
                    with gr.Column():
                        wifi_s = gr.Slider(0, 1, value=0.45, step=0.05,
                                           label="WiFi Speed",
                                           info="Kecepatan internet")
                        socket_s = gr.Slider(0, 1, value=0.30, step=0.05,
                                             label="Socket Availability",
                                             info="Ketersediaan colokan")
                    with gr.Column():
                        quiet_s = gr.Slider(0, 1, value=0.20, step=0.05,
                                            label="Quietness",
                                            info="Tingkat ketenangan")
                        value_s = gr.Slider(0, 1, value=0.05, step=0.05,
                                            label="Value for Money",
                                            info="Harga terjangkau")

                gr.Markdown("**Optional Filters:**")
                with gr.Row():
                    price_f = gr.Dropdown(
                        choices=["Semua"] + PRICE_RANGES, value="Semua",
                        label="Filter Harga"
                    )
                    vibe_f = gr.Dropdown(
                        choices=["Semua"] + VIBE_TAGS, value="Semua",
                        label="Filter Vibe"
                    )
                    top_n_s = gr.Slider(3, 20, value=7, step=1, label="Jumlah Hasil")

                rec_btn = gr.Button("Get Recommendations", variant="primary", size="lg")
                rec_out = gr.Markdown()

                # Wire preset buttons
                for key, btn in preset_buttons.items():
                    btn.click(
                        lambda k=key: use_preset(k),
                        outputs=[wifi_s, socket_s, quiet_s, value_s]
                    )

                rec_btn.click(
                    smart_recommend,
                    inputs=[wifi_s, socket_s, quiet_s, value_s,
                            price_f, vibe_f, top_n_s],
                    outputs=rec_out
                )

            # ----------------- Tab 2: Story Search -----------------
            with gr.Tab("Story Search"):
                gr.Markdown("### Cari dengan Bahasa Natural")
                gr.HTML("""
                <div class="info-box">
                    <strong>NLP-Powered:</strong> Tuliskan deskripsi warkop ideal Anda dalam bahasa sehari-hari.
                    AI akan match menggunakan TF-IDF & Cosine Similarity.
                </div>
                """)

                story_in = gr.Textbox(
                    label="Ceritakan warkop yang Anda cari",
                    placeholder="Contoh: warkop tenang di Bukit Rata buat ngerjain tugas dengan wifi cepat",
                    lines=3
                )
                story_n = gr.Slider(3, 10, value=5, step=1, label="Jumlah Hasil")
                story_btn = gr.Button("Search by Story", variant="primary", size="lg")
                story_out = gr.Markdown()

                gr.Examples(
                    examples=[
                        "warkop tenang buat coding dengan wifi cepat",
                        "tempat nongkrong murah meriah buat kumpul teman",
                        "specialty coffee modern dan instagramable",
                        "warkop legendary di bukit rata",
                        "tempat 24 jam buat begadang ngerjain skripsi",
                        "premium deep work space dengan colokan banyak",
                        "warkop sporty buat nobar bareng teman",
                        "kafe creative buat freelancer di merdeka",
                    ],
                    inputs=story_in,
                    label="Contoh Query"
                )

                story_btn.click(story_search, inputs=[story_in, story_n], outputs=story_out)

            # ----------------- Tab 3: Hybrid AI -----------------
            with gr.Tab("Hybrid AI"):
                gr.Markdown("### Kombinasi Story Search + Personal Preferences")
                gr.HTML("""
                <div class="info-box">
                    <strong>Most Powerful:</strong> Gabungkan deskripsi natural dengan preferensi numerik.
                    Atur slider <em>Text Ratio</em> untuk balance antara semantic search & rule-based scoring.
                </div>
                """)

                h_query = gr.Textbox(
                    label="Deskripsi (opsional)",
                    placeholder="Contoh: warkop modern quiet di pase",
                    lines=2
                )

                with gr.Row():
                    h_wifi = gr.Slider(0, 1, value=0.35, step=0.05, label="WiFi")
                    h_socket = gr.Slider(0, 1, value=0.25, step=0.05, label="Socket")
                    h_quiet = gr.Slider(0, 1, value=0.20, step=0.05, label="Quiet")
                    h_value = gr.Slider(0, 1, value=0.20, step=0.05, label="Value")

                with gr.Row():
                    ratio_s = gr.Slider(
                        0, 1, value=0.5, step=0.05,
                        label="Text vs Numeric Ratio",
                        info="0 = pakai slider saja, 1 = pakai deskripsi saja"
                    )
                    h_top = gr.Slider(3, 15, value=7, step=1, label="Jumlah Hasil")

                h_btn = gr.Button("Run Hybrid AI", variant="primary", size="lg")
                h_out = gr.Markdown()

                h_btn.click(
                    hybrid_search,
                    inputs=[h_query, h_wifi, h_socket, h_quiet, h_value, ratio_s, h_top],
                    outputs=h_out
                )

            # ----------------- Tab 4: By Vibe -----------------
            with gr.Tab("By Vibe"):
                gr.Markdown("### Filter Berdasarkan Suasana / Vibe Tag")
                gr.HTML("""
                <div class="info-box">
                    Vibe akan match secara substring. Misalnya pilih <strong>"Social"</strong>
                    akan menampilkan semua warkop dengan tag mengandung kata Social.
                </div>
                """)

                with gr.Row():
                    vibe_dd = gr.Dropdown(
                        choices=VIBE_TAGS,
                        label="Pilih Kategori Vibe",
                        scale=2
                    )
                    vibe_n = gr.Slider(3, 20, value=8, step=1, label="Jumlah Hasil", scale=1)

                vibe_btn = gr.Button("Search by Vibe", variant="primary", size="lg")
                vibe_out = gr.Markdown()
                vibe_btn.click(vibe_search, inputs=[vibe_dd, vibe_n], outputs=vibe_out)

            # ----------------- Tab 5: Find Similar -----------------
            with gr.Tab("Find Similar"):
                gr.Markdown("### Cari Warkop dengan Karakteristik Serupa")
                gr.HTML("""
                <div class="info-box">
                    Pilih warkop favorit Anda, lalu temukan warkop lain dengan vibe & fitur serupa
                    berdasarkan kombinasi numeric + text similarity.
                </div>
                """)

                with gr.Row():
                    sim_dd = gr.Dropdown(
                        choices=WARKOP_NAMES,
                        label="Pilih Warkop Referensi",
                        scale=2
                    )
                    sim_n = gr.Slider(3, 10, value=5, step=1, label="Jumlah Hasil", scale=1)

                sim_btn = gr.Button("Find Similar", variant="primary", size="lg")
                sim_out = gr.Markdown()
                sim_btn.click(find_similar, inputs=[sim_dd, sim_n], outputs=sim_out)

            # ----------------- Tab 6: By Location -----------------
            with gr.Tab("By Location"):
                gr.Markdown("### Filter Berdasarkan Kawasan / Alamat")
                gr.HTML("""
                <div class="info-box">
                    Pilih kawasan untuk melihat semua warkop di area tersebut.
                    Pencarian dilakukan dengan substring match (case-insensitive).
                </div>
                """)

                with gr.Row():
                    addr_dd = gr.Dropdown(
                        choices=ADDRESS_LIST,
                        label="Pilih Kawasan / Alamat",
                        scale=2
                    )
                    addr_n = gr.Slider(3, 20, value=10, step=1, label="Jumlah Hasil", scale=1)

                addr_btn = gr.Button("Search by Location", variant="primary", size="lg")
                addr_out = gr.Markdown()
                addr_btn.click(filter_by_address, inputs=[addr_dd, addr_n], outputs=addr_out)

            # ----------------- Tab 7: Compare Warkops -----------------
            with gr.Tab("Compare"):
                gr.Markdown("### Bandingkan Dua Warkop Side-by-Side")
                gr.HTML("""
                <div class="info-box">
                    Pilih dua warkop berbeda untuk membandingkan atribut mereka secara langsung.
                </div>
                """)

                with gr.Row():
                    cmp_a = gr.Dropdown(choices=WARKOP_NAMES, label="Warkop A")
                    cmp_b = gr.Dropdown(choices=WARKOP_NAMES, label="Warkop B")

                cmp_btn = gr.Button("Compare", variant="primary", size="lg")
                cmp_out = gr.Markdown()
                cmp_btn.click(compare_warkops, inputs=[cmp_a, cmp_b], outputs=cmp_out)

            # ----------------- Tab 8: Statistics -----------------
            with gr.Tab("Statistics"):
                gr.Markdown("### Database Insights & Analytics")
                stats_btn = gr.Button("Refresh Statistics", variant="primary", size="lg")
                stats_output = gr.Markdown()
                stats_btn.click(get_stats, outputs=stats_output)

                # Auto-load on app start
                app.load(get_stats, outputs=stats_output)

            # ----------------- Tab 9: All Warkops -----------------
            with gr.Tab("All Warkops"):
                gr.Markdown("### Database Lengkap")
                gr.HTML("""
                <div class="info-box">
                    Lihat seluruh database warkop dengan opsi sorting berdasarkan kolom tertentu.
                </div>
                """)

                with gr.Row():
                    sort_dd = gr.Dropdown(
                        choices=['name', 'wifi_speed_mbps', 'price_range',
                                 'noise_level', 'socket_availability',
                                 'vibe_category', 'address'],
                        value='name',
                        label="Urutkan berdasarkan"
                    )
                    asc_cb = gr.Checkbox(value=True, label="Ascending (A-Z, kecil-besar)")

                all_btn = gr.Button("Load Database", variant="primary", size="lg")
                all_out = gr.Markdown()
                all_btn.click(display_all, inputs=[sort_dd, asc_cb], outputs=all_out)

            # ----------------- Tab 10: About -----------------
            with gr.Tab("About"):
                gr.Markdown("""
                ## Tentang Warkop Intelligence Finder

                Aplikasi ini adalah **sistem rekomendasi cerdas** untuk membantu Anda menemukan
                warkop terbaik di Lhokseumawe sesuai kebutuhan personal Anda.

                ### Teknologi yang Digunakan

                | Teknologi | Fungsi |
                |-----------|--------|
                | **Content-Based Filtering** | Rekomendasi berbasis fitur numerik |
                | **TF-IDF Vectorization** | Konversi teks ke vektor numerik |
                | **Cosine Similarity** | Mengukur kemiripan antar item |
                | **MinMax Scaler** | Normalisasi fitur numerik |
                | **Hybrid Scoring** | Kombinasi numeric + semantic |
                | **Gradio Framework** | Interactive UI |

                ### Fitur Utama

                1. **Smart Recommendation** - Rekomendasi via slider preferensi
                2. **Story Search** - Cari dengan bahasa natural (NLP)
                3. **Hybrid AI** - Kombinasi text + numeric scoring
                4. **By Vibe** - Filter berdasarkan suasana
                5. **Find Similar** - Temukan warkop mirip
                6. **By Location** - Filter berdasarkan kawasan
                7. **Compare** - Bandingkan dua warkop
                8. **Statistics** - Dashboard analitik database
                9. **All Warkops** - Browse seluruh database

                ### Schema Data

                | Kolom | Tipe | Deskripsi |
                |-------|------|-----------|
                | `name` | string | Nama warkop |
                | `address` | string | Alamat / kawasan |
                | `wifi_speed_mbps` | int | Kecepatan WiFi (Mbps) |
                | `socket_availability` | enum | Low / Medium / High |
                | `noise_level` | enum | Low / Medium / High |
                | `vibe_category` | string | Tag vibe (slash-separated) |
                | `price_range` | enum | Cheap / Medium / Expensive |

                ### Algoritma Scoring

                **Numeric Score:**
                ```
                score = w_wifi * wifi_norm
                      + w_socket * socket_norm
                      + w_quiet * (1 - noise_norm)
                      + w_value * (1 - price_norm)
                ```

                **Hybrid Score:**
                ```
                final = text_ratio * cosine_sim(query, metadata)
                      + (1 - text_ratio) * numeric_score
                ```

                ### Pengembang

                Dibuat untuk **Politeknik Negeri Lhokseumawe**
                Program Studi: Teknik Informatika
                """)

        # ===== Footer =====
        gr.HTML("""
        <div class="footer-section">
            <p><strong>Warkop Intelligence Finder v3.0</strong></p>
            <p>Powered by Hybrid Machine Learning | TF-IDF + Cosine Similarity + Content-Based Filtering</p>
            <p style="font-size: 0.85em; color: #999;">
                Built for <strong>Politeknik Negeri Lhokseumawe</strong> - Informatics Engineering
            </p>
        </div>
        """)

    return app


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )