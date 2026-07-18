"""
UPGRADE - ml-25m Data Pipeline
=================================
ml-25m has ~25 million ratings, ~162,000 users, ~62,000 movies.
Building a dense (movies x users) matrix from the FULL dataset would need
roughly 80GB of memory -- not feasible on a laptop.

Solution: filter down to the most active users and most-rated movies.
This is standard practice in real recommender systems (focus compute on
"warm" users with enough history; handle everyone else via cold-start
fallback, which you already built). We'll keep this filtered-but-large
subset, and document the filtering clearly in the report.

Output format matches your existing scripts (Y.npy, R.npy, movie_ids.csv),
so day3to5 / day8to9 / day10 / app.py only need their file paths updated.
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------
# Tunable thresholds -- adjust these if you want a bigger/smaller matrix
# -----------------------------------------------------------
MOVIE_MIN_RATINGS = 100    # a movie needs at least this many ratings to be kept
USER_MIN_RATINGS = 50      # a user needs at least this many ratings to be kept
MAX_USERS = 8000           # hard cap -- keeps memory usage safe
MAX_MOVIES = 4000          # hard cap -- keeps memory usage safe

# -----------------------------------------------------------
# STEP 1: Load raw data (ml-25m uses commas + headers, unlike ml-1m/ml-100k)
# -----------------------------------------------------------

print("Loading ratings.csv (this is a big file, may take a minute)...")
ratings = pd.read_csv('ml-25m/ratings.csv')  # columns: userId, movieId, rating, timestamp
movies = pd.read_csv('ml-25m/movies.csv')    # columns: movieId, title, genres

ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
movies = movies.rename(columns={'movieId': 'movie_id'})

print(f"Raw: {len(ratings):,} ratings, {ratings['user_id'].nunique():,} users, "
      f"{ratings['movie_id'].nunique():,} movies")

# -----------------------------------------------------------
# STEP 2: Filter down to a tractable, still-substantial subset
# -----------------------------------------------------------

# Pass 1: keep movies with enough ratings
movie_counts = ratings.groupby('movie_id').size()
keep_movies = movie_counts[movie_counts >= MOVIE_MIN_RATINGS].index
ratings = ratings[ratings['movie_id'].isin(keep_movies)]

# Pass 2: keep users with enough ratings (within the filtered movie set)
user_counts = ratings.groupby('user_id').size()
keep_users = user_counts[user_counts >= USER_MIN_RATINGS].index
ratings = ratings[ratings['user_id'].isin(keep_users)]

# Pass 3: if still too big, cap to the MOST ACTIVE users/movies
user_counts = ratings.groupby('user_id').size().sort_values(ascending=False)
if len(user_counts) > MAX_USERS:
    top_users = user_counts.head(MAX_USERS).index
    ratings = ratings[ratings['user_id'].isin(top_users)]

movie_counts = ratings.groupby('movie_id').size().sort_values(ascending=False)
if len(movie_counts) > MAX_MOVIES:
    top_movies = movie_counts.head(MAX_MOVIES).index
    ratings = ratings[ratings['movie_id'].isin(top_movies)]

print(f"\nFiltered: {len(ratings):,} ratings, {ratings['user_id'].nunique():,} users, "
      f"{ratings['movie_id'].nunique():,} movies")
print(f"(Filtered from the full 25M-rating dataset to the most active users "
      f"and most-rated movies, to keep the matrix a manageable size.)")

# -----------------------------------------------------------
# STEP 3: Parse genres (for the hybrid model in Days 3-4)
# -----------------------------------------------------------

kept_movie_ids = ratings['movie_id'].unique()
movies = movies[movies['movie_id'].isin(kept_movie_ids)].copy()
movies['genre_list'] = movies['genres'].str.split('|')

all_genres = sorted(set(
    g for genres in movies['genre_list'] for g in genres if g != '(no genres listed)'
))
print(f"\nFound {len(all_genres)} unique genres")

genre_matrix = pd.DataFrame(0, index=movies['movie_id'], columns=all_genres)
for _, row in movies.iterrows():
    for g in row['genre_list']:
        if g in all_genres:
            genre_matrix.loc[row['movie_id'], g] = 1
genre_matrix.to_csv('genre_matrix.csv')
print("Saved genre_matrix.csv")

# -----------------------------------------------------------
# STEP 4: Build the rating matrix (vectorized -- important at this scale)
# -----------------------------------------------------------

all_movie_ids = sorted(ratings['movie_id'].unique())
all_user_ids = sorted(ratings['user_id'].unique())
movie_id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}
user_id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}

num_movies = len(all_movie_ids)
num_users = len(all_user_ids)
est_mb = num_movies * num_users * 8 / 1e6
print(f"\nBuilding {num_movies} x {num_users} matrix (~{est_mb:.0f} MB per matrix)")

movie_indices = ratings['movie_id'].map(movie_id_to_idx).values
user_indices = ratings['user_id'].map(user_id_to_idx).values

Y = np.zeros((num_movies, num_users))
R = np.zeros((num_movies, num_users))
Y[movie_indices, user_indices] = ratings['rating'].values
R[movie_indices, user_indices] = 1

sparsity = 1 - (len(ratings) / (num_movies * num_users))
print(f"Sparsity: {sparsity:.2%}")

# -----------------------------------------------------------
# STEP 5: Save (same filenames as before, so other scripts still work
# once you point them at 'ml-25m' instead of 'ml-100k'/'ml-1m')
# -----------------------------------------------------------

np.save('Y.npy', Y)
np.save('R.npy', R)
pd.Series(all_movie_ids, name='movie_id').to_csv('movie_ids.csv', index=False)
pd.Series(all_user_ids, name='user_id').to_csv('user_ids.csv', index=False)
movies[['movie_id', 'title', 'genres']].to_csv('movies_25m.csv', index=False)
ratings.to_csv('ratings_25m_filtered.csv', index=False)

print("\nSaved Y.npy, R.npy, movie_ids.csv, user_ids.csv, movies_25m.csv, ratings_25m_filtered.csv")
