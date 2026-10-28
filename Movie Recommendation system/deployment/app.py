from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1120; color: #eef2ff; }
    [data-testid="stSidebar"] { background: #111a2d; }
    .hero {
        padding: 1.7rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #202b54, #121a2c 75%);
        border: 1px solid #293653;
    }
    .hero h1 { margin: 0; color: #f4f6ff; }
    .hero p { color: #bdc8e2; margin-bottom: 0; }
    .metric-card {
        padding: 1rem; border: 1px solid #293653; border-radius: 14px;
        background: #121c30; min-height: 100px;
    }
    .movie-card {
        padding: 0.9rem; border: 1px solid #293653; border-radius: 12px;
        background: #121c30; margin-bottom: 0.7rem;
    }
    .movie-title { font-weight: 700; color: #f4f6ff; }
    .muted { color: #aab7d3; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("ml-latest-small")


@st.cache_data
def load_data():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    links_path = DATA_DIR / "links.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            "Dataset files not found: movies.csv and ratings.csv. Make sure the "
            "'ml-latest-small' folder is next to app.py."
        )

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    links_df = pd.read_csv(links_path) if links_path.exists() else pd.DataFrame()
    return movies_df, ratings_df, links_df


@st.cache_resource
def build_models():
    movies_df, ratings_df, _ = load_data()

    # Content-based model: genres -> TF-IDF -> cosine similarity
    genre_text = movies_df["genres"].fillna("").str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer(stop_words="english")
    genre_matrix = vectorizer.fit_transform(genre_text)
    content_similarity = cosine_similarity(genre_matrix, dense_output=True)

    # Same reproducible split and baseline collaborative filtering as the notebook
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=0.2, random_state=42
    )
    train_matrix = train_ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    train_filled = train_matrix.fillna(0)
    similarity = cosine_similarity(train_filled)
    similarity_df = pd.DataFrame(
        similarity, index=train_matrix.index, columns=train_matrix.index
    )
    return content_similarity, train_ratings, test_ratings, train_matrix, similarity_df


def recommend_similar_movies(movie_title, top_n=5):
    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        return pd.DataFrame(columns=["title", "genres"])
    movie_idx = matches.index[0]
    scores = list(enumerate(content_similarity[movie_idx]))
    scores.sort(key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, score in scores if idx != movie_idx][:top_n]
    result = movies.iloc[top_indices][["title", "genres"]].copy()
    result["Similarity"] = [content_similarity[movie_idx][idx] for idx in top_indices]
    return result.reset_index(drop=True)


def recommend_movies_from_train(user_id, n_recommendations=10, n_neighbors=50, min_support=3):
    columns = ["movieId", "title", "genres", "Predicted Rating", "Similar Users Count"]
    if user_id not in train_user_item_matrix.index:
        return pd.DataFrame(columns=columns)

    similar_users = (
        train_user_similarity_df.loc[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
        .head(n_neighbors)
    )
    neighbor_ratings = train_user_item_matrix.loc[similar_users.index]
    weighted_ratings = neighbor_ratings.fillna(0).mul(similar_users, axis=0).sum(axis=0)
    rating_weights = neighbor_ratings.notna().mul(similar_users, axis=0).sum(axis=0)
    predicted = weighted_ratings.div(rating_weights.replace(0, np.nan))
    rating_counts = neighbor_ratings.notna().sum(axis=0)

    already_rated = train_user_item_matrix.loc[user_id].dropna().index
    predicted = predicted.drop(index=already_rated, errors="ignore")
    candidates = pd.DataFrame({
        "movieId": predicted.index,
        "Predicted Rating": predicted.values,
        "Similar Users Count": rating_counts.reindex(predicted.index).values,
    })
    candidates = candidates[
        (candidates["Similar Users Count"] >= min_support) & candidates["Predicted Rating"].notna()
    ]
    candidates = candidates.sort_values(
        ["Predicted Rating", "Similar Users Count"], ascending=[False, False]
    ).head(n_recommendations)
    return candidates.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")[columns].reset_index(drop=True)


def show_movie_cards(frame, show_prediction=False):
    if frame.empty:
        st.info("No suitable results found. Please try another selection.")
        return
    cols = st.columns(min(5, len(frame)))
    for idx, (_, row) in enumerate(frame.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">{row.get("genres", "Genres unavailable")}</div>', unsafe_allow_html=True)
            if show_prediction and pd.notna(row.get("Predicted Rating")):
                st.caption(f'Predicted rating: {row["Predicted Rating"]:.2f} / 5')
                st.caption(f'Similar users: {int(row["Similar Users Count"])}')
            elif pd.notna(row.get("Similarity")):
                st.caption(f'Similarity: {row["Similarity"]:.2f}')
            st.markdown("</div>", unsafe_allow_html=True)


try:
    movies, ratings, links = load_data()
    content_similarity, train_ratings, test_ratings, train_user_item_matrix, train_user_similarity_df = build_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.title("🎬 Movie Recommender")
sections = ["Home", "Movie Recommendation", "User Recommendation", "Project Insights"]

# Apply navigation requests before the radio widget is instantiated.
if "page_choice" not in st.session_state:
    st.session_state["page_choice"] = "Home"
if "next_page" in st.session_state:
    st.session_state["page_choice"] = st.session_state.pop("next_page")

page = st.sidebar.radio(
    "Navigate",
    sections,
    key="page_choice",
)
st.sidebar.caption("MovieLens · Content-Based + Collaborative Filtering")

if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <h1>🎬 Movie Recommendation System</h1>
          <p>Discover movies you'll love — powered by data, not guesswork.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Explore the app")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🍿 Movie Recommendation")
        st.write("Choose a movie to discover similar titles using content-based filtering.")
        if st.button("Find similar movies", use_container_width=True):
            st.session_state["next_page"] = "Movie Recommendation"
            st.rerun()
    with c2:
        st.markdown("### 👤 User Recommendation")
        st.write("Select a User ID to get recommendations based on similar users.")
        if st.button("Get recommendations", use_container_width=True):
            st.session_state["next_page"] = "User Recommendation"
            st.rerun()
    st.info("Choose a section from the sidebar to get started.")

elif page == "Movie Recommendation":
    st.markdown("## 🍿 Movie Recommendation")
    st.write("Choose a movie to find similar titles based on genres.")
    selected_movie = st.selectbox("Choose a movie", sorted(movies["title"].dropna().unique()))
    number = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Find Similar Movies", type="primary"):
        result = recommend_similar_movies(selected_movie, number)
        show_movie_cards(result)

elif page == "User Recommendation":
    st.markdown("## 👤 Personalized Recommendations")
    st.write("Choose a User ID available in the training data.")
    valid_users = sorted(train_user_item_matrix.index.astype(int).tolist())
    user_id = st.selectbox("User ID", valid_users, index=0)
    number = st.slider("Number of recommendations", 5, 20, 10)
    neighbors = st.slider("Number of similar users", 10, 100, 50, step=10)
    if st.button("Get Recommendations", type="primary"):
        result = recommend_movies_from_train(user_id, number, neighbors)
        show_movie_cards(result, show_prediction=True)

elif page == "Project Insights":
    st.markdown("## 📊 Project Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{movies['movieId'].nunique():,}")
    c2.metric("Users", f"{ratings['userId'].nunique():,}")
    c3.metric("Ratings", f"{len(ratings):,}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Average rating", f"{ratings['rating'].mean():.2f} / 5")
    c5.metric("Precision@10", "2.30%", help="Previously evaluated in the project notebook.")
    c6.metric("Recall@10", "3.11%", help="Previously evaluated in the project notebook.")
    st.markdown("### Rating distribution")
    counts = ratings["rating"].value_counts().sort_index()
    st.bar_chart(counts)
    st.caption("Precision@10 and Recall@10 are shown from the previously recorded notebook evaluation; they are not recalculated when the app starts.")
