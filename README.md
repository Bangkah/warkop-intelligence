# Warkop Intelligence Finder (WIE)

> Sistem rekomendasi cerdas berbasis Machine Learning untuk menemukan warkop terbaik di Lhokseumawe sesuai kebutuhan personal Anda.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44%2B-orange.svg)](https://gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## Daftar Isi

- [Tentang Project](#tentang-project)
- [Fitur Utama](#fitur-utama)
- [Tech Stack](#tech-stack)
- [Algoritma & Metodologi](#algoritma--metodologi)
- [Struktur Project](#struktur-project)
- [Schema Data](#schema-data)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Preset Profiles](#preset-profiles)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Roadmap](#roadmap)
- [Kontribusi](#kontribusi)
- [Troubleshooting](#troubleshooting)
- [Lisensi](#lisensi)
- [Pengembang](#pengembang)

---

## Tentang Project

**Warkop Intelligence Finder (WIE)** adalah aplikasi web interaktif yang menggunakan kombinasi **Content-Based Filtering** dan **Semantic Search** untuk merekomendasikan warkop berdasarkan preferensi pengguna. Project ini dikembangkan sebagai bagian dari studi Teknik Informatika di **Politeknik Negeri Lhokseumawe**.

### Latar Belakang

Lhokseumawe memiliki budaya warkop yang sangat kuat. Dari mahasiswa yang mencari tempat belajar dengan WiFi cepat, hingga pekerja yang butuh suasana tenang untuk fokus, setiap orang memiliki kebutuhan berbeda. Aplikasi ini dirancang untuk memecahkan masalah tersebut dengan pendekatan **data-driven recommendation**.

### Tujuan

- Memberikan rekomendasi warkop yang **personal** dan **akurat**
- Mengkombinasikan **multiple criteria** dalam satu sistem scoring
- Menyediakan **multiple search modes** untuk fleksibilitas pengguna
- Mendokumentasikan ekosistem warkop lokal Lhokseumawe

---

## Fitur Utama

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | **Smart Recommendation** | Rekomendasi via slider preferensi dengan 5 preset profile siap pakai |
| 2 | **Story Search** | Pencarian dengan bahasa natural (NLP-powered) menggunakan TF-IDF |
| 3 | **Hybrid AI** | Kombinasi text + numeric scoring dengan ratio yang dapat diatur |
| 4 | **By Vibe** | Filter berdasarkan suasana / vibe tag (Social, Coding, Premium, dll) |
| 5 | **Find Similar** | Temukan warkop dengan karakteristik serupa berdasarkan satu referensi |
| 6 | **By Location** | Filter berdasarkan kawasan / alamat |
| 7 | **Compare** | Bandingkan dua warkop side-by-side |
| 8 | **Statistics** | Dashboard analitik database dengan auto-load |
| 9 | **All Warkops** | Browse seluruh database dengan opsi sorting |
| 10 | **About** | Dokumentasi lengkap dalam aplikasi |

### Highlight Fitur

- **Preset Profiles** untuk quick recommendation (Coding, Social, Student, Premium, Chill)
- **Filter System** terintegrasi di multiple tab (price, vibe)
- **Caching** dengan `@lru_cache` untuk performa optimal
- **Error Handling** robust dengan logging
- **Auto-normalization** weights agar tidak perlu sum = 1
- **Substring matching** untuk vibe tags yang fleksibel

---

## Tech Stack

### Core Dependencies

| Layer | Teknologi | Versi |
|-------|-----------|-------|
| **Frontend UI** | Gradio | >= 4.44.0 |
| **Backend** | Python | >= 3.9 |
| **ML Framework** | scikit-learn | >= 1.3.0 |
| **Data Processing** | Pandas | >= 2.0.0 |
| **Numerical Computing** | NumPy | >= 1.24.0 |
| **Table Formatting** | Tabulate | >= 0.9.0 |

### ML Components

- **TfidfVectorizer** - Konversi teks ke vektor numerik
- **MinMaxScaler** - Normalisasi fitur numerik ke range [0, 1]
- **Cosine Similarity** - Mengukur kemiripan antar item
- **Content-Based Filtering** - Rekomendasi berbasis fitur item

---

## Algoritma & Metodologi

### 1. Content-Based Filtering (Numeric Scoring)

Setiap warkop direpresentasikan sebagai vektor 4-dimensi:

```
v = [wifi, sockets, quiet, value]
```

Dimana:
- `wifi` = `wifi_speed_mbps / max_wifi_speed`
- `sockets` = `socket_level / 3.0` (Low=1, Medium=2, High=3)
- `quiet` = `1 - (noise_level / 3.0)` (inverted)
- `value` = `1 - (price_level / 3.0)` (inverted)

**Numeric Score:**

```
score = w_wifi   * wifi_normalized
      + w_socket * socket_normalized
      + w_quiet  * (1 - noise_normalized)
      + w_value  * (1 - price_normalized)
```

Dimana `w_*` adalah weight yang diberikan user (otomatis dinormalisasi sehingga sum = 1).

### 2. Semantic Search (TF-IDF + Cosine Similarity)

Setiap warkop digabungkan menjadi metadata text:

```python
metadata = name + address + vibe_category + noise_level + price_range + socket_level
```

Kemudian:

1. **TF-IDF Vectorization** mengkonversi text ke sparse matrix
2. **Cosine Similarity** dihitung antara query dan setiap warkop
3. Top-N warkop dengan similarity tertinggi diberikan sebagai rekomendasi

**Formula Cosine Similarity:**

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

### 3. Hybrid Scoring

Menggabungkan kedua pendekatan dengan ratio yang dapat diatur:

```
final_score = text_ratio       * cosine_similarity(query, metadata)
            + (1 - text_ratio) * numeric_score
```

- `text_ratio = 0` → Pure numeric (slider-based)
- `text_ratio = 1` → Pure semantic (story-based)
- `text_ratio = 0.5` → Balanced hybrid

### 4. Find Similar

Untuk mencari warkop mirip, digunakan **average similarity** dari kedua space:

```
similarity = (cos_sim_numeric + cos_sim_text) / 2
```

---

## Struktur Project

```
warkop-intelligence/
│
├── app.py                      # Gradio UI application (entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
├── README.md.hf                # README untuk Hugging Face Spaces
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container orchestration
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker ignore rules
│
├── data/
│   └── warkops.csv             # Warkop database (20 entries)
│
├── models/
│   ├── __init__.py             # Package initializer
│   └── recommender.py          # WarkopRecommender engine
│
└── docs/                       # Optional documentation
    ├── MIGRATION_NOTES.md
    ├── PROJECT_SUMMARY.md
    └── SETUP_GUIDE.md
```

---

## Schema Data

File `data/warkops.csv` menggunakan struktur berikut:

| Kolom | Tipe | Nilai Valid | Deskripsi |
|-------|------|-------------|-----------|
| `name` | string | - | Nama warkop |
| `address` | string | - | Alamat / kawasan |
| `wifi_speed_mbps` | int | 0 - 100+ | Kecepatan WiFi (Mbps) |
| `socket_availability` | enum | Low / Medium / High | Ketersediaan colokan |
| `noise_level` | enum | Low / Medium / High | Tingkat keramaian |
| `vibe_category` | string | slash-separated | Tag vibe (e.g. "Social/Student") |
| `price_range` | enum | Cheap / Medium / Expensive | Range harga |

### Contoh Data

```csv
name,address,wifi_speed_mbps,socket_availability,noise_level,vibe_category,price_range
Warkop Syarif Delima,Bukit Rata,20,Medium,High,Social/Student,Cheap
Station Coffee Premium,Jl. Merdeka Barat,55,High,Medium,Deep Work/Premium,Expensive
KOPIKA. CO,Jl. Merdeka Timur,45,High,Low,Creative/Work,Medium
```

### Statistik Dataset Saat Ini

- **Total Warkops:** 20
- **WiFi Range:** 15 - 55 Mbps
- **Vibe Tags:** 25+ unique tags
- **Coverage Area:** Bukit Rata, Banda Sakti, Pase, Merdeka, dll

---

## Instalasi

### Prasyarat

- Python 3.9 atau lebih tinggi
- pip (Python package manager)
- Git (optional, untuk clone)

### Langkah-langkah

#### 1. Clone Repository

```bash
git clone https://github.com/Bangkah/warkop-intelligence.git
cd warkop-intelligence
```

Atau download ZIP dari GitHub dan extract.

#### 2. Setup Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verifikasi Instalasi

```bash
python -c "import gradio, pandas, sklearn; print('All packages installed successfully')"
```

#### 5. Run Aplikasi

```bash
python app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:7860`

---

## Cara Penggunaan

### 1. Smart Recommendation

Tab utama untuk rekomendasi personal dengan kontrol penuh.

**Langkah:**
1. Klik salah satu **Quick Preset** untuk auto-fill slider
2. Atau atur manual 4 slider preferensi (WiFi, Socket, Quiet, Value)
3. Tambahkan filter opsional (Harga, Vibe)
4. Atur jumlah hasil yang diinginkan (3-20)
5. Klik **Get Recommendations**

**Tip:** Weights akan otomatis dinormalisasi, jadi Anda tidak perlu memastikan total = 1.

### 2. Story Search

Pencarian dengan bahasa natural Indonesia.

**Contoh Query:**

```
"warkop tenang buat coding dengan wifi cepat"
"tempat nongkrong murah meriah buat kumpul teman"
"specialty coffee modern dan instagramable"
"warkop legendary di bukit rata"
"tempat 24 jam buat begadang ngerjain skripsi"
```

**Tip:** Gunakan kata kunci yang mendeskripsikan vibe, lokasi, atau kebutuhan spesifik.

### 3. Hybrid AI

Kombinasi terkuat - text + numeric scoring.

**Cara Kerja Text Ratio:**
- `0.0` → Pure numeric (hanya slider)
- `0.3` → Numeric dominant
- `0.5` → Balanced (default)
- `0.7` → Text dominant
- `1.0` → Pure semantic (hanya deskripsi)

### 4. By Vibe

Filter berdasarkan tag vibe spesifik. Pilih dari dropdown yang berisi semua vibe unique dari dataset.

### 5. Find Similar

Pilih warkop favorit Anda → temukan warkop lain dengan vibe & fitur serupa berdasarkan kombinasi numeric + text similarity.

### 6. By Location

Filter berdasarkan kawasan / alamat. Berguna jika Anda ingin warkop di area spesifik (misal: Bukit Rata, Banda Sakti).

### 7. Compare

Bandingkan dua warkop side-by-side untuk membantu keputusan akhir.

### 8. Statistics

Dashboard analitik dengan info:
- Total warkops
- WiFi performance (min, avg, max)
- Distribusi noise, price, socket
- Top vibe tags

### 9. All Warkops

Browse seluruh database dengan opsi sorting berdasarkan kolom apapun.

---

## Preset Profiles

Aplikasi menyediakan 5 preset rekomendasi siap pakai:

| Preset | WiFi | Socket | Quiet | Value | Use Case |
|--------|------|--------|-------|-------|----------|
| **Coding & Deep Work** | 0.45 | 0.30 | 0.20 | 0.05 | Developer, freelancer, programmer |
| **Social & Hangout** | 0.15 | 0.20 | 0.05 | 0.60 | Kumpul teman, gathering |
| **Student Mode** | 0.35 | 0.25 | 0.10 | 0.30 | Mahasiswa belajar / tugas |
| **Premium Experience** | 0.40 | 0.25 | 0.30 | 0.05 | Kualitas terbaik, harga bukan masalah |
| **Chill & Quiet** | 0.10 | 0.15 | 0.45 | 0.30 | Santai, baca buku, istirahat |

---

## API Reference

### Python API Examples

#### Basic Usage

```python
from models.recommender import WarkopRecommender

# Initialize
rec = WarkopRecommender("data/warkops.csv")

# Print info
print(rec)
print(f"Total warkops: {len(rec)}")
```

#### Weight-Based Recommendation

```python
result = rec.recommend_by_weights(
    wifi_imp=0.5,
    socket_imp=0.3,
    quiet_imp=0.1,
    value_imp=0.1,
    top_n=5,
    filters={'price_range': 'Cheap'}  # optional
)
print(result)
```

#### Story Search

```python
result = rec.recommend_by_story(
    user_query="warkop tenang dengan wifi cepat buat coding",
    top_n=5
)
```

#### Hybrid Recommendation

```python
result = rec.recommend_hybrid(
    user_query="warkop modern di pase",
    weights={
        "wifi_imp": 0.4,
        "socket_imp": 0.3,
        "quiet_imp": 0.2,
        "value_imp": 0.1
    },
    text_ratio=0.6,  # 60% text, 40% numeric
    top_n=7
)
```

#### Find Similar

```python
result = rec.find_similar("Station Coffee Premium", top_n=3)
```

#### Use Preset

```python
# Available presets: 'coding', 'social', 'student', 'premium', 'chill'
result = rec.recommend_by_preset("coding", top_n=5)
```

#### Get Statistics

```python
stats = rec.get_quick_stats()
print(stats)
# Output:
# {
#     'Total Warkop': 20,
#     'WiFi Tercepat (Mbps)': 55.0,
#     'WiFi Rata-rata (Mbps)': 33.75,
#     'Spot Paling Tenang': 4,
#     ...
# }
```

#### Get Vibe Tags

```python
tags = rec.get_all_vibe_tags()
print(tags)
# ['Chill', 'Classic', 'Creative', 'Deep Work', 'Espresso', ...]
```

### Method Signatures

```python
# Weight-based
recommend_by_weights(
    wifi_imp: float = 0.4,
    socket_imp: float = 0.3,
    quiet_imp: float = 0.2,
    value_imp: float = 0.1,
    top_n: int = 5,
    filters: Optional[Dict] = None
) -> pd.DataFrame

# Story-based
recommend_by_story(
    user_query: str,
    top_n: int = 3,
    min_similarity: float = 0.0,
    filters: Optional[Dict] = None
) -> pd.DataFrame

# Hybrid
recommend_hybrid(
    user_query: str = "",
    weights: Optional[Dict] = None,
    text_ratio: float = 0.5,
    top_n: int = 5,
    filters: Optional[Dict] = None
) -> pd.DataFrame

# Find similar
find_similar(
    warkop_name: str,
    top_n: int = 3
) -> pd.DataFrame

# Preset
recommend_by_preset(
    preset: str,  # 'coding', 'social', 'student', 'premium', 'chill'
    top_n: int = 5,
    filters: Optional[Dict] = None
) -> pd.DataFrame
```

---

## Roadmap

### Selesai
- [x] Content-based filtering (numeric)
- [x] TF-IDF semantic search (text)
- [x] Hybrid recommendation (text + numeric)
- [x] Multi-tab Gradio UI (10 tabs)
- [x] Preset profiles
- [x] Filter system (price, vibe, location)
- [x] Find similar warkops
- [x] Compare warkops
- [x] Statistics dashboard
- [x] Caching dengan @lru_cache
- [x] Deployment guide (HF Spaces, Docker, VPS)

### Dalam Pengembangan
- [ ] User authentication & profile
- [ ] Review & rating system
- [ ] Map visualization (Leaflet/Folium)
- [ ] Sentence-Transformers embeddings (BERT-based)
- [ ] Real-time data update via API
- [ ] Mobile-responsive optimization

### Future Plans
- [ ] Collaborative filtering (user-based)
- [ ] Image gallery per warkop
- [ ] Opening hours & busy times
- [ ] Multi-language support (English, Aceh)
- [ ] Mobile app (React Native / Flutter)
- [ ] Admin dashboard untuk update database
- [ ] Webhook untuk new warkop submission

---

## Kontribusi

Kontribusi sangat dipersilakan! Berikut cara berkontribusi:

### Reporting Bugs

1. Cek [Issues](../../issues) untuk memastikan bug belum dilaporkan
2. Buat issue baru dengan template berikut:
   - Deskripsi bug
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots (jika ada)
   - Environment (OS, Python version, dll)

### Suggesting Features

1. Buka [Discussions](../../discussions)
2. Pilih kategori "Ideas"
3. Jelaskan use case dan expected behavior

### Pull Requests

1. **Fork** project ini
2. Buat **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes dengan message yang descriptive (`git commit -m 'feat: add amazing feature'`)
4. **Push** ke branch (`git push origin feature/AmazingFeature`)
5. Buka **Pull Request** dengan deskripsi lengkap

### Commit Convention

Project ini menggunakan [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - Fitur baru
- `fix:` - Bug fix
- `docs:` - Update dokumentasi
- `style:` - Formatting (tidak mengubah logic)
- `refactor:` - Refactoring code
- `test:` - Menambah/update tests
- `chore:` - Maintenance tasks

### Menambahkan Data Warkop Baru

Untuk submit warkop baru ke database, edit `data/warkops.csv` dengan format yang sesuai schema, kemudian buat PR.

---

## Troubleshooting

### Port 7860 Sudah Digunakan

**Linux / macOS:**
```bash
lsof -ti:7860 | xargs kill -9
```

**Windows PowerShell:**
```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 7860).OwningProcess | Stop-Process
```

### ModuleNotFoundError

```bash
# Pastikan venv aktif
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### CSV File Not Found

```bash
# Verifikasi file ada
ls data/warkops.csv

# Cek working directory
python -c "import os; print(os.getcwd())"
```

### Gradio Theme Error

Update Gradio ke versi terbaru:

```bash
pip install --upgrade gradio
```

### Permission Denied (Windows)

Jika `Activate.ps1` gagal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Slow Performance

- Gunakan `@lru_cache` (sudah implemented)
- Reduce `tfidf_max_features` di config
- Limit `top_n` ke nilai yang reasonable (3-15)

---

## FAQ

**Q: Apakah saya bisa menambahkan warkop dari kota lain?**
A: Ya, cukup edit `data/warkops.csv`. Pastikan schema tetap sama. Untuk multi-kota, pertimbangkan menambah kolom `city`.

**Q: Apakah algoritma bisa diganti dengan deep learning?**
A: Ya, di roadmap kami akan menambahkan Sentence-Transformers untuk semantic search yang lebih powerful.

**Q: Bisakah aplikasi ini digunakan untuk recommend cafe / restoran?**
A: Tentu! Cukup ganti dataset dan sesuaikan kolom. Algoritma scoring tetap sama.

**Q: Apakah ada API endpoint untuk integrasi dengan aplikasi lain?**
A: Saat ini belum ada REST API. Anda bisa import `WarkopRecommender` langsung di Python project Anda.

**Q: Bagaimana cara menambahkan filter baru?**
A: Edit method `_build_filter_mask` di `recommender.py` dan tambahkan UI di `app.py`.

---

## Lisensi

Project ini dilisensikan di bawah [MIT License](LICENSE).


## Pengembang

Project ini dikembangkan dengan dedikasi untuk:

**Politeknik Negeri Lhokseumawe**
Program Studi: Teknik Informatika

### Kontak

- **GitHub:** [@Bangkah](https://github.com/Bangkah)
- **Email:** mdhyaulatha@gmail.com
- **LinkedIn:** [Muhammad Dhiyaul Atha](https://www.linkedin.com/in/muhammad-dhyaul-atha/)

---

## Acknowledgments

Terima kasih kepada:

- **Komunitas warkop Lhokseumawe** atas inspirasi data
- **scikit-learn team** untuk ML toolkit yang powerful
- **Gradio team** untuk UI framework yang user-friendly
- **Hugging Face** untuk hosting platform gratis
- **Politeknik Negeri Lhokseumawe** untuk dukungan akademis
- **Open Source community** untuk semua libraries yang digunakan

---

## Citation

Jika Anda menggunakan project ini dalam riset atau publikasi, mohon mencantumkan:

```bibtex
@software{warkop_intelligence_2026,
  author = Muhammad Dhiyaul Atha,
  title = {Warkop Intelligence Finder: ML-Based Warkop Recommender for Lhokseumawe},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Bangkah/warkop-intelligence}
}
```

---

## Star History

Jika project ini bermanfaat, mohon berikan star di GitHub!

---

<div align="center">

**Made with dedication for Lhokseumawe Coffee Culture**

[Back to Top](#warkop-intelligence-finder-wie)

</div>