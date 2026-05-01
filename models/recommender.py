"""
Warkop Recommender Engine v3.0 - Lhokseumawe Edition
Content-Based Filtering (Numeric) + Semantic Search (TF-IDF)

Schema:
    name, address, wifi_speed_mbps, socket_availability,
    noise_level, vibe_category, price_range
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, Dict, List
from dataclasses import dataclass, field
from functools import lru_cache

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class RecommenderConfig:
    """Configuration for the recommender engine."""
    socket_levels: Dict[str, int] = field(default_factory=lambda: {
        'Low': 1, 'Medium': 2, 'High': 3
    })
    noise_levels: Dict[str, int] = field(default_factory=lambda: {
        'Low': 1, 'Medium': 2, 'High': 3
    })
    price_levels: Dict[str, int] = field(default_factory=lambda: {
        'Cheap': 1, 'Medium': 2, 'Expensive': 3
    })

    stopwords: List[str] = field(default_factory=lambda: [
        'saya', 'mau', 'yang', 'dan', 'di', 'ke', 'buat', 'cari', 'dengan',
        'untuk', 'dari', 'ini', 'itu', 'atau', 'juga', 'aku', 'gue', 'ada',
        'bisa', 'lagi', 'sama', 'pada', 'akan', 'sudah', 'belum', 'jl',
        'tempat', 'warkop', 'kupi', 'kopi', 'coffee'
    ])

    required_columns: List[str] = field(default_factory=lambda: [
        'name', 'address', 'wifi_speed_mbps', 'socket_availability',
        'noise_level', 'vibe_category', 'price_range'
    ])

    tfidf_ngram_range: tuple = (1, 2)
    tfidf_max_features: int = 3000
    tfidf_min_df: int = 1


class WarkopRecommenderError(Exception):
    """Custom exception for recommender errors."""
    pass


# =============================================================================
# Main Recommender Class
# =============================================================================
class WarkopRecommender:
    """
    Hybrid recommender for warkops in Lhokseumawe.

    Features:
    - Weight-based content filtering (WiFi, sockets, quiet, value)
    - Semantic search via TF-IDF (vibe + name + address)
    - Hybrid mode (combines both)
    - Filtering by price/vibe/noise
    - Find similar warkops
    """

    DEFAULT_WEIGHTS = {
        'wifi_imp': 0.4,
        'socket_imp': 0.3,
        'quiet_imp': 0.2,
        'value_imp': 0.1
    }

    PRESET_PROFILES = {
        'coding':  {'wifi_imp': 0.45, 'socket_imp': 0.30, 'quiet_imp': 0.20, 'value_imp': 0.05},
        'social':  {'wifi_imp': 0.15, 'socket_imp': 0.20, 'quiet_imp': 0.05, 'value_imp': 0.60},
        'student': {'wifi_imp': 0.35, 'socket_imp': 0.25, 'quiet_imp': 0.10, 'value_imp': 0.30},
        'premium': {'wifi_imp': 0.40, 'socket_imp': 0.25, 'quiet_imp': 0.30, 'value_imp': 0.05},
        'chill':   {'wifi_imp': 0.10, 'socket_imp': 0.15, 'quiet_imp': 0.45, 'value_imp': 0.30},
    }

    def __init__(
        self,
        csv_path: Union[str, Path, pd.DataFrame],
        config: Optional[RecommenderConfig] = None
    ):
        self.config = config or RecommenderConfig()
        self.df = self._load_data(csv_path)
        self._validate_data()

        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(
            stop_words=self.config.stopwords,
            ngram_range=self.config.tfidf_ngram_range,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
        )

        self.feature_matrix: Optional[np.ndarray] = None
        self.text_matrix = None
        self._feature_names: List[str] = []

        self._clean_data()
        self._prepare_numeric_features()
        self._prepare_text_features()

        logger.info(f"Recommender initialized with {len(self.df)} warkops")

    # -------------------------------------------------------------------------
    # Data Loading & Validation
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_data(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = Path(source)
        if not path.exists():
            raise WarkopRecommenderError(f"File not found: {path}")
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise WarkopRecommenderError(f"Failed to read CSV: {e}") from e

    def _validate_data(self) -> None:
        missing = set(self.config.required_columns) - set(self.df.columns)
        if missing:
            raise WarkopRecommenderError(f"Missing required columns: {missing}")
        if self.df.empty:
            raise WarkopRecommenderError("Dataset is empty")

    def _clean_data(self) -> None:
        """Standardize categorical & numeric columns."""
        categorical_cols = ['socket_availability', 'noise_level', 'price_range']
        for col in categorical_cols:
            self.df[col] = (
                self.df[col].astype(str).str.strip().str.title()
                .replace('Nan', 'Medium').fillna('Medium')
            )

        self.df['wifi_speed_mbps'] = pd.to_numeric(
            self.df['wifi_speed_mbps'], errors='coerce'
        ).fillna(0)

        for col in ['name', 'address', 'vibe_category']:
            self.df[col] = self.df[col].fillna('').astype(str).str.strip()

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    def _prepare_numeric_features(self) -> None:
        features = pd.DataFrame(index=self.df.index)

        max_wifi = self.df['wifi_speed_mbps'].max()
        features['wifi'] = (
            self.df['wifi_speed_mbps'] / max_wifi if max_wifi > 0 else 0
        )
        features['sockets'] = (
            self.df['socket_availability'].map(self.config.socket_levels).fillna(2) / 3.0
        )
        noise_num = self.df['noise_level'].map(self.config.noise_levels).fillna(2)
        features['quiet'] = 1 - (noise_num / 3.0)
        price_num = self.df['price_range'].map(self.config.price_levels).fillna(2)
        features['value'] = 1 - (price_num / 3.0)

        self._feature_names = features.columns.tolist()
        self.feature_matrix = self.scaler.fit_transform(features)

    def _prepare_text_features(self) -> None:
        """Build text metadata combining vibe, name, address, and categorical info."""
        vibe_clean = self.df['vibe_category'].str.replace('/', ' ').str.lower()

        self.df['metadata'] = (
            self.df['name'].str.lower() + ' ' +
            self.df['address'].str.lower() + ' ' +
            vibe_clean + ' ' +
            self.df['noise_level'].str.lower() + ' noise ' +
            self.df['price_range'].str.lower() + ' price ' +
            self.df['socket_availability'].str.lower() + ' socket'
        )

        self.text_matrix = self.tfidf.fit_transform(self.df['metadata'])

    # -------------------------------------------------------------------------
    # Recommendation: Weight-Based
    # -------------------------------------------------------------------------
    def recommend_by_weights(
        self,
        wifi_imp: float = 0.4,
        socket_imp: float = 0.3,
        quiet_imp: float = 0.2,
        value_imp: float = 0.1,
        top_n: int = 5,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> pd.DataFrame:
        """Recommendation based on weighted preferences."""
        weights = np.array([wifi_imp, socket_imp, quiet_imp, value_imp], dtype=float)

        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        total = weights.sum()
        if total == 0:
            raise ValueError("At least one weight must be positive")
        weights = weights / total

        scores = self.feature_matrix @ weights
        mask = self._build_filter_mask(filters)
        masked_scores = np.where(mask, scores, -np.inf)

        n_available = int(mask.sum())
        if n_available == 0:
            logger.warning("No warkops match the filters")
            return pd.DataFrame()

        top_n = min(top_n, n_available)
        top_indices = np.argsort(masked_scores)[::-1][:top_n]

        results = self.df.iloc[top_indices].copy()
        results['match_score'] = (scores[top_indices] * 100).round(2)
        return self._clean_output(results)

    # -------------------------------------------------------------------------
    # Recommendation: Story / Semantic Search
    # -------------------------------------------------------------------------
    def recommend_by_story(
        self,
        user_query: str,
        top_n: int = 3,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> pd.DataFrame:
        """Semantic search via natural language query."""
        if not user_query or not user_query.strip():
            return self._clean_output(self.df.head(top_n).copy())

        similarities = self._compute_query_similarity(user_query.lower().strip())
        mask = self._build_filter_mask(filters)
        masked_sims = np.where(mask, similarities, -1)

        top_indices = np.argsort(masked_sims)[::-1][:top_n]

        if masked_sims[top_indices[0]] <= min_similarity:
            logger.info("Low similarity, falling back to weighted defaults")
            return self.recommend_by_weights(top_n=top_n, filters=filters)

        results = self.df.iloc[top_indices].copy()
        results['match_score'] = (similarities[top_indices] * 100).round(2)
        return self._clean_output(results)

    # -------------------------------------------------------------------------
    # Recommendation: Hybrid
    # -------------------------------------------------------------------------
    def recommend_hybrid(
        self,
        user_query: str = "",
        weights: Optional[Dict[str, float]] = None,
        text_ratio: float = 0.5,
        top_n: int = 5,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> pd.DataFrame:
        """Hybrid: combines numeric weights + semantic similarity."""
        if not 0 <= text_ratio <= 1:
            raise ValueError("text_ratio must be between 0 and 1")

        w = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        weight_arr = np.array([w['wifi_imp'], w['socket_imp'], w['quiet_imp'], w['value_imp']])
        weight_arr = weight_arr / weight_arr.sum()

        numeric_scores = self.feature_matrix @ weight_arr
        text_scores = (
            self._compute_query_similarity(user_query.lower().strip())
            if user_query.strip() else np.zeros(len(self.df))
        )

        effective_ratio = text_ratio if user_query.strip() else 0.0
        combined = effective_ratio * text_scores + (1 - effective_ratio) * numeric_scores

        mask = self._build_filter_mask(filters)
        combined_masked = np.where(mask, combined, -np.inf)

        n_available = int(mask.sum())
        if n_available == 0:
            return pd.DataFrame()
        top_n = min(top_n, n_available)
        top_indices = np.argsort(combined_masked)[::-1][:top_n]

        results = self.df.iloc[top_indices].copy()
        results['match_score'] = (combined[top_indices] * 100).round(2)
        return self._clean_output(results)

    # -------------------------------------------------------------------------
    # Find Similar
    # -------------------------------------------------------------------------
    def find_similar(self, warkop_name: str, top_n: int = 3) -> pd.DataFrame:
        """Find warkops similar to a given one."""
        matches = self.df[self.df['name'].str.contains(warkop_name, case=False, na=False, regex=False)]
        if matches.empty:
            raise WarkopRecommenderError(f"Warkop '{warkop_name}' not found")

        idx = matches.index[0]
        sim_numeric = cosine_similarity(
            self.feature_matrix[idx].reshape(1, -1), self.feature_matrix
        ).flatten()
        sim_text = cosine_similarity(self.text_matrix[idx], self.text_matrix).flatten()
        combined = (sim_numeric + sim_text) / 2
        combined[idx] = -1  # exclude self

        top_indices = np.argsort(combined)[::-1][:top_n]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = (combined[top_indices] * 100).round(2)
        return self._clean_output(results)

    # -------------------------------------------------------------------------
    # Preset Recommendations
    # -------------------------------------------------------------------------
    def recommend_by_preset(self, preset: str, top_n: int = 5,
                            filters: Optional[Dict] = None) -> pd.DataFrame:
        """Quick recommendation using preset profile."""
        if preset not in self.PRESET_PROFILES:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESET_PROFILES)}")
        weights = self.PRESET_PROFILES[preset]
        return self.recommend_by_weights(**weights, top_n=top_n, filters=filters)

    # -------------------------------------------------------------------------
    # Filter Helpers
    # -------------------------------------------------------------------------
    def _build_filter_mask(
        self, filters: Optional[Dict[str, Union[str, List[str]]]]
    ) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        if not filters:
            return mask
        for col, value in filters.items():
            if col not in self.df.columns:
                logger.warning(f"Filter column '{col}' not found, skipping")
                continue
            values = [value] if isinstance(value, str) else list(value)

            # Special handling for vibe_category (substring match)
            if col == 'vibe_category':
                col_mask = self.df[col].str.lower().apply(
                    lambda x: any(v.lower() in x for v in values)
                ).values
            else:
                col_mask = self.df[col].isin(values).values
            mask &= col_mask
        return mask

    @staticmethod
    def _clean_output(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in ['metadata'] if c in df.columns]
        return df.drop(columns=cols_to_drop).reset_index(drop=True)

    @lru_cache(maxsize=128)
    def _compute_query_similarity(self, query: str) -> np.ndarray:
        query_vec = self.tfidf.transform([query])
        return cosine_similarity(query_vec, self.text_matrix).flatten()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    def get_quick_stats(self) -> dict:
        """Return summary statistics for dashboard."""
        all_vibes = []
        for v in self.df['vibe_category'].dropna():
            all_vibes.extend([t.strip() for t in v.split('/')])
        vibe_counts = pd.Series(all_vibes).value_counts().to_dict()

        return {
            'Total Warkop': int(len(self.df)),
            'WiFi Tercepat (Mbps)': float(self.df['wifi_speed_mbps'].max()),
            'WiFi Rata-rata (Mbps)': round(float(self.df['wifi_speed_mbps'].mean()), 2),
            'WiFi Terlambat (Mbps)': float(self.df['wifi_speed_mbps'].min()),
            'Spot Paling Tenang': int((self.df['noise_level'] == 'Low').sum()),
            'Spot Bising': int((self.df['noise_level'] == 'High').sum()),
            'Budget Friendly': int((self.df['price_range'] == 'Cheap').sum()),
            'Premium': int((self.df['price_range'] == 'Expensive').sum()),
            'Socket Melimpah': int((self.df['socket_availability'] == 'High').sum()),
            'Socket Sedikit': int((self.df['socket_availability'] == 'Low').sum()),
            'Top Vibe Tags': dict(list(vibe_counts.items())[:10]),
            'Total Unique Vibes': len(vibe_counts),
        }

    def get_all_vibe_tags(self) -> List[str]:
        """Get all unique vibe tags (split by /)."""
        all_vibes = set()
        for v in self.df['vibe_category'].dropna():
            for tag in v.split('/'):
                all_vibes.add(tag.strip())
        return sorted(all_vibes)

    def __repr__(self) -> str:
        return f"<WarkopRecommender(n={len(self.df)}, features={self._feature_names})>"

    def __len__(self) -> int:
        return len(self.df)