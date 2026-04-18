import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Rich UI Imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings("ignore")
console = Console()

# CONFIGURATION AND PARAMETERS
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TAGS_PATH = DATA_DIR / "tags.csv"
LINKS_PATH = DATA_DIR / "links.csv"

TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TEST_USER_ID = 1
TOP_N_RECS = 10

# SETUP FOLDERS
RESULTS_DIR.mkdir(exist_ok=True)
if not DATA_DIR.exists():
    console.print(
        f"[bold red]Error:[/bold red] Data directory '{DATA_DIR}' not found. Please ensure your CSV files are inside it."
    )


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    tags = pd.read_csv(TAGS_PATH)
    links = pd.read_csv(LINKS_PATH)

    table = Table(title="Dataset Summary", title_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Record Count", justify="right", style="green")

    table.add_row("Movies", f"{len(movies):,}")
    table.add_row("Ratings", f"{len(ratings):,}")
    table.add_row("Tags", f"{len(tags):,}")

    console.print(table)
    return movies, ratings, tags, links


def build_item_profiles(movies: pd.DataFrame, tags: pd.DataFrame):
    movies = movies.copy()
    # Extract year
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)

    # Scale year
    scaler = MinMaxScaler()
    movies["year_norm"] = scaler.fit_transform(
        movies[["year"]].fillna(movies["year"].median())
    )

    # One-Hot Encoding Genres
    genres_clean = movies["genres"].replace("(no genres listed)", "")
    genre_dummies = genres_clean.str.get_dummies(sep="|")

    # Vectorize Tags
    tag_docs = (
        tags.groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(ts.astype(str).str.lower()))
        .reset_index()
        .rename(columns={"tag": "tag_text"})
    )
    movies = movies.merge(tag_docs, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(movies["tag_text"]).toarray()
    tfidf_cols = [f"tag_{t}" for t in tfidf.get_feature_names_out()]

    # Combine all features
    year_arr = movies[["year_norm"]].values
    genre_arr = genre_dummies.values
    feature_matrix = np.hstack([year_arr, genre_arr, tfidf_matrix])
    feature_names = ["year_norm"] + list(genre_dummies.columns) + tfidf_cols

    console.print(
        f"\n[bold yellow][Step 1][/bold yellow] Item profile matrix created: [blue]{feature_matrix.shape}[/blue]"
    )
    return feature_matrix, movies.set_index("movieId"), feature_names


def build_user_profile(
    user_id: int,
    ratings: pd.DataFrame,
    feature_matrix: np.ndarray,
    movie_df: pd.DataFrame,
):
    user_ratings = ratings[ratings["userId"] == user_id].copy()
    if user_ratings.empty:
        raise ValueError(f"userId {user_id} not found in ratings.")

    valid_ids = set(movie_df.index)
    user_ratings = user_ratings[user_ratings["movieId"].isin(valid_ids)]

    movie_ids_ordered = movie_df.index.tolist()
    id_to_row = {mid: i for i, mid in enumerate(movie_ids_ordered)}

    row_indices = [id_to_row[mid] for mid in user_ratings["movieId"]]
    rating_vals = user_ratings["rating"].values

    watched_features = feature_matrix[row_indices]
    user_profile = rating_vals[np.newaxis, :] @ watched_features
    user_profile /= rating_vals.sum()

    watched_ids = set(user_ratings["movieId"].tolist())
    console.print(
        f"[bold yellow][Step 2][/bold yellow] User {user_id} preferences modeled using {len(watched_ids)} ratings."
    )

    return user_profile, watched_ids


def recommend(
    user_profile: np.ndarray,
    watched_ids: set,
    feature_matrix: np.ndarray,
    movie_df: pd.DataFrame,
    top_n: int = TOP_N_RECS,
):
    movie_ids_ordered = movie_df.index.tolist()

    candidate_mask = np.array([mid not in watched_ids for mid in movie_ids_ordered])
    candidate_ids = np.array(movie_ids_ordered)[candidate_mask]
    candidate_features = feature_matrix[candidate_mask]

    similarities = cosine_similarity(user_profile, candidate_features)[0]

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_ids = candidate_ids[top_indices]
    top_scores = similarities[top_indices]

    results = (
        movie_df.loc[top_ids, ["title", "genres"]]
        .copy()
        .assign(similarity=top_scores)
        .reset_index(drop=True)
    )
    results.index += 1

    # Print with Rich Table
    rec_table = Table(
        title=f"Top {top_n} Recommendations for User {TEST_USER_ID}",
        title_style="bold green",
    )
    rec_table.add_column("Rank", justify="center")
    rec_table.add_column("Title", style="italic")
    rec_table.add_column("Genres", style="dim")
    rec_table.add_column("Match Score", justify="right", style="bold yellow")

    for i, row in results.iterrows():
        rec_table.add_row(
            str(i), row["title"], row["genres"], f"{row['similarity']:.4f}"
        )

    console.print("\n", rec_table)
    return results


def main():
    try:
        # 1. Load Data
        movies, ratings, tags, links = load_data()

        # 2. Process
        feature_matrix, movie_df, _ = build_item_profiles(movies, tags)
        user_profile, watched_ids = build_user_profile(
            TEST_USER_ID, ratings, feature_matrix, movie_df
        )

        # 3. Recommend
        top_results = recommend(user_profile, watched_ids, feature_matrix, movie_df)

        # 4. Save Results
        output_file = RESULTS_DIR / f"recommendations_user_{TEST_USER_ID}.csv"
        top_results.to_csv(output_file)
        console.print(
            f"\n[bold green]Success![/bold green] Results saved to: [underline]{output_file}[/underline]"
        )

        # 5. Evaluation Notes
        console.print(
            Panel(
                "Evaluation Strategy:\n• Hit Rate@K\n• Precision@K\n• Recall@K",
                title="Next Steps",
                subtitle="Step 4",
                border_style="cyan",
            )
        )

    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
