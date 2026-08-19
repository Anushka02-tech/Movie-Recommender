import streamlit as st
import numpy as np
import pandas as pd
import requests
import re

st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_API_KEY = st.secrets.get("tmdb_api_key", None)

ARTICLE_SUFFIX_RE = re.compile(r"^(.*),\s*(The|A|An)$")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

def clean_title_for_search(raw_title):

    title = raw_title.strip()

    year_match = YEAR_RE.search(title)
    year = year_match.group(1) if year_match else None
    if year_match:
        title = title[:year_match.start()].strip()

    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()

    article_match = ARTICLE_SUFFIX_RE.match(title)
    if article_match:
        title = f"{article_match.group(2)} {article_match.group(1)}".strip()

    return title, year

@st.cache_data(show_spinner=False)
def fetch_poster_url(title):
    if not TMDB_API_KEY:
        return None

    cleaned_title, year = clean_title_for_search(title)

    def _search(query, year=None):
        params = {"api_key": TMDB_API_KEY, "query": query}
        if year:
            params["year"] = year
        resp = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json().get("results", [])

    try:
        results = _search(cleaned_title, year)
        if not results and year:
            results = _search(cleaned_title)
        if not results:
            results = _search(title)

        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w200{results[0]['poster_path']}"
    except requests.RequestException:
        pass
    return None

@st.cache_data
def load_data():
    movies = pd.read_csv('movies_25m.csv')[['movie_id', 'title']]
    ratings = pd.read_parquet('ratings_25m_filtered.parquet')
    movie_ids = pd.read_csv('movie_ids.csv')['movie_id'].values
    display_names = pd.read_csv('user_display_names.csv')
    genre_matrix = pd.read_csv('genre_matrix.csv', index_col='movie_id')
    X = np.load('X_trained.npy')
    Theta = np.load('Theta_trained.npy')
    Y_mean = np.load('Y_mean.npy')
    R = np.load('R.npy')
    return movies, ratings, movie_ids, display_names, genre_matrix, X, Theta, Y_mean, R

movies, ratings, movie_ids, display_names, genre_matrix, X, Theta, Y_mean, R = load_data()
all_genres = sorted(genre_matrix.columns)

predictions = X @ Theta.T + Y_mean.reshape(-1, 1)
all_user_ids = sorted(ratings['user_id'].unique())
user_id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}

name_to_id = dict(zip(display_names['display_name'], display_names['user_id']))
id_to_name = dict(zip(display_names['user_id'], display_names['display_name']))
all_display_names = sorted(name_to_id.keys())

MIN_RATINGS_THRESHOLD = 5

def popularity_recommend(n=5, min_ratings=20):
    stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= min_ratings]
    top_n = stats.sort_values('mean', ascending=False).head(n)
    result = top_n.merge(movies, on='movie_id')
    return result[['title', 'mean']].rename(columns={'mean': 'predicted_rating'})

def genre_recommend(selected_genres, n=5, min_ratings=50):
    if not selected_genres:
        return popularity_recommend(n=n)

    matches = genre_matrix[selected_genres].sum(axis=1) > 0
    matching_movie_ids = genre_matrix.index[matches]

    stats = ratings[ratings['movie_id'].isin(matching_movie_ids)].groupby('movie_id')['rating'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= min_ratings]

    if len(stats) == 0:
        stats = ratings[ratings['movie_id'].isin(matching_movie_ids)].groupby('movie_id')['rating'].agg(['mean', 'count'])

    top_n = stats.sort_values('mean', ascending=False).head(n)
    result = top_n.merge(movies, on='movie_id')
    return result[['title', 'mean']].rename(columns={'mean': 'predicted_rating'})

def recommend(user_idx, n=5, min_ratings=20):
    user_ratings = predictions[:, user_idx]
    already_rated = R[:, user_idx] == 1
    too_obscure = R.sum(axis=1) < min_ratings

    scores = user_ratings.copy()
    scores[already_rated] = -np.inf
    scores[too_obscure] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]
    top_movie_ids = movie_ids[top_indices]
    top_scores = scores[top_indices]

    result = pd.DataFrame({'movie_id': top_movie_ids, 'predicted_rating': top_scores})
    result = result.merge(movies, on='movie_id')
    return result[['title', 'predicted_rating']]

def get_user_history(user_id, n=5):
    user_ratings = ratings[ratings['user_id'] == user_id]
    top_rated = user_ratings.sort_values('rating', ascending=False).head(n)
    result = top_rated.merge(movies, on='movie_id')
    return result[['title', 'rating']]

st.title("Movie Recommender System")
st.caption("Display names are randomly generated for demo purposes, the underlying MovieLens data is fully anonymized.")

with st.sidebar:
    st.header("Settings")

    user_type = st.radio(
        "Who are we recommending for?",
        options=["Existing user", "New user (cold-start demo)"]
    )

    if user_type == "Existing user":
        selected_name = st.selectbox(
            "Select a user to get recommendations for:",
            options=all_display_names,
            index=0
        )
        selected_user = name_to_id[selected_name]
        selected_genres = []
    else:
        selected_user = None
        selected_name = None
        st.info("Simulating a brand-new user with zero rating history. "
                "Tell us what you like, and we'll personalize recommendations using genre preferences instead.")
        selected_genres = st.multiselect(
            "Pick a few genres you enjoy:",
            options=all_genres,
            default=[]
        )

    num_recs = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)

    get_recs_clicked = st.button("Get Recommendations", type="primary", use_container_width=True)

if "has_run" not in st.session_state:
    st.session_state.has_run = False

if get_recs_clicked:
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.info("Set your preferences in the sidebar, then hit **Get Recommendations** to see results here.")

if get_recs_clicked:
    with st.spinner("Finding recommendations for you..."):
        if selected_user is not None:
            num_ratings = (ratings['user_id'] == selected_user).sum()
        else:
            num_ratings = 0  

        use_personalized = selected_user is not None and num_ratings >= MIN_RATINGS_THRESHOLD

        if use_personalized:
            method_label = "personalized (collaborative filtering)"
            user_idx = user_id_to_idx[selected_user]
            recs = recommend(user_idx, n=num_recs)
        elif selected_user is None and selected_genres:
            method_label = f"genre-based (matching {', '.join(selected_genres)})"
            recs = genre_recommend(selected_genres, n=num_recs)
        else:
            method_label = "popularity-based (cold-start fallback, no genres selected)"
            recs = popularity_recommend(n=num_recs)

        recs = recs.copy()
        recs["poster_url"] = recs["title"].apply(fetch_poster_url)

    st.toast("Recommendations ready!")

    with st.expander(
        f"Based on: {selected_name}'s ratings" if selected_name else "Based on: your stated preferences",
        expanded=False
    ):
        if selected_user is None:
            if selected_genres:
                st.write("_No rating history — matching on selected genres:_")
                for g in selected_genres:
                    st.write(f"{g}")
            else:
                st.write("_No rating history and no genres selected — showing general popularity._")
        else:
            history = get_user_history(selected_user, n=5)
            if len(history) == 0:
                st.write("No rating history found for this user.")
            else:
                for _, row in history.iterrows():
                    st.write(f"{row['rating']:.0f}  —  {row['title']}")

    st.subheader(f"Recommended for {selected_name}" if selected_name else "Recommended for you")
    st.caption(f"Method: {method_label}")

    POSTERS_PER_ROW = 5
    rows = [recs.iloc[i:i + POSTERS_PER_ROW] for i in range(0, len(recs), POSTERS_PER_ROW)]

    for row_chunk in rows:
        cols = st.columns(POSTERS_PER_ROW)
        for col, (_, movie) in zip(cols, row_chunk.iterrows()):
            with col:
                if isinstance(movie["poster_url"], str) and movie["poster_url"]:
                    st.image(movie["poster_url"], use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#eee;border-radius:6px;height:180px;"
                        "display:flex;align-items:center;justify-content:center;"
                        "color:#999;font-size:0.8rem;text-align:center;padding:8px;'>"
                        "No poster found</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(f"**{movie['title']}**")
                st.caption(f"predicted {movie['predicted_rating']:.2f}")

st.divider()
st.caption(
    "Model: collaborative filtering with 15 latent features, trained via gradient descent "
    "with L2 regularization on 6.46 million ratings (filtered from MovieLens 25M to the "
    "8,000 most active users and 4,000 most-rated movies). "
    "Test RMSE: 0.7227 (vs. 0.9220 for a popularity baseline — a 21.6% improvement)."
)