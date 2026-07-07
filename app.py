from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
DATASET_FOLDER = "ml-32m"
DEFAULT_DATA_ROOT = Path(__file__).parent / "data"


st.set_page_config(
    page_title="Clustering MovieLens 32M",
    page_icon="M",
    layout="wide",
)


def find_dataset_dir(folder: str) -> str:
    root = Path(folder).expanduser()
    candidates = [root, root / DATASET_FOLDER]

    for candidate in candidates:
        if (candidate / "movies.csv").exists() and (candidate / "ratings.csv").exists():
            return str(candidate)

    raise FileNotFoundError(
        "Folder harus berisi movies.csv dan ratings.csv, atau subfolder ml-32m."
    )


@st.cache_data(show_spinner="Menyiapkan dataset MovieLens 32M...")
def ensure_dataset(data_root: str) -> str:
    root = Path(data_root)
    dataset_dir = root / DATASET_FOLDER
    movies_path = dataset_dir / "movies.csv"
    ratings_path = dataset_dir / "ratings.csv"

    if movies_path.exists() and ratings_path.exists():
        return str(dataset_dir)

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-32m.zip"

    if not zip_path.exists():
        urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("Dataset MovieLens 32M gagal diekstrak.")

    return str(dataset_dir)


@st.cache_data(show_spinner="Mengagregasi rating dan membangun fitur...")
def load_feature_matrix(dataset_dir: str, chunksize: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    dataset_path = Path(dataset_dir)
    movies_path = dataset_path / "movies.csv"
    ratings_path = dataset_path / "ratings.csv"

    movies = pd.read_csv(movies_path, dtype={"movieId": "int32"})
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
        chunksize=chunksize,
    ):
        grouped = (
            chunk.groupby("movieId", observed=True)["rating"]
            .agg(rating_sum="sum", rating_count="count")
            .reset_index()
        )
        partials.append(grouped)

    rating_stats = (
        pd.concat(partials, ignore_index=True)
        .groupby("movieId", observed=True)
        .agg(rating_sum=("rating_sum", "sum"), rating_count=("rating_count", "sum"))
        .reset_index()
    )
    rating_stats["avg_rating"] = (
        rating_stats["rating_sum"] / rating_stats["rating_count"]
    )

    movies = movies.merge(
        rating_stats[["movieId", "avg_rating", "rating_count"]],
        on="movieId",
        how="inner",
    )

    genre_features = movies["genres"].str.get_dummies("|")
    numeric_features = movies[["avg_rating", "rating_count"]].copy()
    numeric_features["rating_count"] = np.log1p(numeric_features["rating_count"])

    features = pd.concat([genre_features, numeric_features], axis=1)
    scaled = StandardScaler().fit_transform(features)

    return movies, scaled, features.columns.tolist()


@st.cache_data(show_spinner="Melatih model clustering...")
def prepare_clustered_movies(
    dataset_dir: str,
    chunksize: int,
    cluster_count: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[float, float]]:
    movies, scaled, feature_columns = load_feature_matrix(dataset_dir, chunksize)

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        batch_size=4096,
        n_init=10,
    )

    clustered = movies.copy()
    clustered["cluster"] = model.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=random_state)
    pca_result = pca.fit_transform(scaled)
    clustered["PCA1"] = pca_result[:, 0]
    clustered["PCA2"] = pca_result[:, 1]

    cluster_summary = (
        clustered.groupby("cluster")
        .agg(
            avg_rating=("avg_rating", "mean"),
            rating_count=("rating_count", "mean"),
            jumlah_film=("movieId", "count"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    explained = tuple(float(x) for x in pca.explained_variance_ratio_)
    return clustered, cluster_summary, feature_columns, explained


@st.cache_data(show_spinner="Menghitung elbow dan silhouette...")
def compute_diagnostics(
    dataset_dir: str,
    chunksize: int,
    max_k: int,
    random_state: int,
    sample_size: int,
) -> pd.DataFrame:
    _, scaled, _ = load_feature_matrix(dataset_dir, chunksize)
    sample_size = min(sample_size, len(scaled))
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(scaled), sample_size, replace=False)

    rows = []
    for k in range(2, max_k + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=4096,
            n_init=10,
        )
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled[sample_idx], labels[sample_idx])
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": score})

    return pd.DataFrame(rows)


def format_movie_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df[["title", "genres", "avg_rating", "rating_count", "cluster"]].copy()
    formatted["avg_rating"] = formatted["avg_rating"].round(3)
    formatted["rating_count"] = formatted["rating_count"].astype("int64")
    return formatted


def find_movies(movies: pd.DataFrame, query: str, limit: int = 25) -> pd.DataFrame:
    if not query.strip():
        return movies.sort_values("rating_count", ascending=False).head(limit)

    mask = movies["title"].str.contains(query.strip(), case=False, na=False, regex=False)
    return movies.loc[mask].sort_values("rating_count", ascending=False).head(limit)


def recommend_from_cluster(
    movies: pd.DataFrame,
    movie_id: int,
    recommendation_count: int,
    min_rating_count: int,
) -> tuple[pd.Series, pd.DataFrame]:
    target = movies.loc[movies["movieId"].eq(movie_id)].iloc[0]
    recommendations = movies.loc[
        movies["cluster"].eq(target["cluster"]) & movies["movieId"].ne(movie_id)
    ].copy()
    recommendations = recommendations.loc[
        recommendations["rating_count"].ge(min_rating_count)
    ]

    recommendations = recommendations.sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    ).head(recommendation_count)

    return target, recommendations


st.title("Clustering MovieLens 32M")

with st.sidebar:
    st.header("Pengaturan")
    data_source = st.radio(
        "Sumber data",
        ["Download otomatis", "Folder lokal"],
        index=0,
    )

    local_folder = ""
    if data_source == "Folder lokal":
        local_folder = st.text_input(
            "Path folder",
            value=str(DEFAULT_DATA_ROOT / DATASET_FOLDER),
        )

    cluster_count = st.slider("Jumlah cluster", 2, 20, 10)
    recommendation_count = st.slider("Jumlah rekomendasi", 5, 30, 10)
    min_rating_count = st.number_input(
        "Minimal jumlah rating",
        min_value=1,
        max_value=1_000_000,
        value=100,
        step=50,
    )
    chunksize = st.select_slider(
        "Ukuran chunk ratings",
        options=[500_000, 1_000_000, 2_000_000, 5_000_000],
        value=2_000_000,
        format_func=lambda value: f"{value:,}",
    )
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    if "run_pipeline" not in st.session_state:
        st.session_state.run_pipeline = False

    if st.button("Proses Dataset", type="primary", use_container_width=True):
        st.session_state.run_pipeline = True


if not st.session_state.run_pipeline:
    st.info("Dataset belum diproses.")
    st.stop()


try:
    if data_source == "Download otomatis":
        dataset_dir = ensure_dataset(str(DEFAULT_DATA_ROOT))
    else:
        dataset_dir = find_dataset_dir(local_folder)

    movies, cluster_summary, feature_columns, explained_variance = prepare_clustered_movies(
        dataset_dir=dataset_dir,
        chunksize=chunksize,
        cluster_count=cluster_count,
        random_state=int(random_state),
    )
except Exception as exc:
    st.error(f"Pipeline gagal: {exc}")
    st.stop()


overview, recommendations_tab, clusters_tab, diagnostics_tab = st.tabs(
    ["Overview", "Rekomendasi", "Cluster", "Elbow & Silhouette"]
)

with overview:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Film", f"{len(movies):,}")
    metric_cols[1].metric("Cluster", f"{cluster_count}")
    metric_cols[2].metric("Fitur", f"{len(feature_columns)}")
    metric_cols[3].metric(
        "PCA variance",
        f"{sum(explained_variance):.2%}",
    )

    st.subheader("Ringkasan Cluster")
    st.dataframe(
        cluster_summary.round({"avg_rating": 3, "rating_count": 0}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Visualisasi PCA")
    scatter_limit = st.slider(
        "Jumlah titik PCA",
        min_value=1_000,
        max_value=min(20_000, len(movies)),
        value=min(10_000, len(movies)),
        step=1_000,
    )
    sample = movies.sample(scatter_limit, random_state=int(random_state)).copy()
    sample["cluster_label"] = "Cluster " + sample["cluster"].astype(str)
    st.scatter_chart(
        sample,
        x="PCA1",
        y="PCA2",
        color="cluster_label",
        height=520,
        use_container_width=True,
    )

    st.subheader("Preview Data")
    st.dataframe(
        format_movie_table(movies.sort_values("rating_count", ascending=False).head(100)),
        use_container_width=True,
        hide_index=True,
    )

with recommendations_tab:
    query = st.text_input("Cari judul film", value="Toy Story")
    matches = find_movies(movies, query)

    if matches.empty:
        st.warning("Film tidak ditemukan.")
    else:
        movie_lookup = {
            int(row.movieId): f"{row.title} | {row.genres}"
            for row in matches.itertuples(index=False)
        }
        selected_movie_id = st.selectbox(
            "Pilih film",
            options=list(movie_lookup.keys()),
            format_func=movie_lookup.get,
        )
        target, recs = recommend_from_cluster(
            movies,
            selected_movie_id,
            recommendation_count,
            int(min_rating_count),
        )

        st.metric("Cluster film terpilih", f"Cluster {int(target['cluster'])}")
        st.dataframe(
            format_movie_table(recs),
            use_container_width=True,
            hide_index=True,
        )

with clusters_tab:
    selected_cluster = st.selectbox(
        "Pilih cluster",
        options=sorted(movies["cluster"].unique()),
        format_func=lambda value: f"Cluster {value}",
    )
    cluster_movies = movies.loc[movies["cluster"].eq(selected_cluster)]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Jumlah film", f"{len(cluster_movies):,}")
    col_b.metric("Rata-rata rating", f"{cluster_movies['avg_rating'].mean():.3f}")
    col_c.metric("Rata-rata jumlah rating", f"{cluster_movies['rating_count'].mean():,.0f}")

    st.subheader("Film Terbaik")
    best_movies = cluster_movies.sort_values(
        ["avg_rating", "rating_count"], ascending=[False, False]
    ).head(50)
    st.dataframe(
        format_movie_table(best_movies),
        use_container_width=True,
        hide_index=True,
    )

with diagnostics_tab:
    max_k = st.slider("Maksimum k", 3, 25, 20)
    silhouette_sample_size = st.slider(
        "Sample silhouette",
        min_value=500,
        max_value=min(10_000, len(movies)),
        value=min(5_000, len(movies)),
        step=500,
    )

    if st.button("Hitung Metrik", use_container_width=True):
        diagnostics = compute_diagnostics(
            dataset_dir=dataset_dir,
            chunksize=chunksize,
            max_k=max_k,
            random_state=int(random_state),
            sample_size=silhouette_sample_size,
        )

        chart_data = diagnostics.set_index("k")
        st.subheader("Elbow Method")
        st.line_chart(chart_data[["inertia"]], height=320, use_container_width=True)

        st.subheader("Silhouette Score")
        st.line_chart(chart_data[["silhouette"]], height=320, use_container_width=True)

        st.dataframe(diagnostics.round(4), use_container_width=True, hide_index=True)


csv = movies.to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh movies_clustered.csv",
    data=csv,
    file_name=f"movies_clustered_k{cluster_count}.csv",
    mime="text/csv",
    use_container_width=True,
)
