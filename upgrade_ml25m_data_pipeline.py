import pandas as pd
import numpy as np

MOVIE_MIN_RATINGS = 100    
USER_MIN_RATINGS = 50     
MAX_USERS = 8000           
MAX_MOVIES = 4000          

print("Loading ratings.csv (this is a big file, may take a minute)...")
ratings = pd.read_csv('ml-25m/ratings.csv')  # columns: userId, movieId, rating, timestamp
movies = pd.read_csv('ml-25m/movies.csv')    # columns: movieId, title, genres

ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
movies = movies.rename(columns={'movieId': 'movie_id'})

print(f"Raw: {len(ratings):,} ratings, {ratings['user_id'].nunique():,} users, "
      f"{ratings['movie_id'].nunique():,} movies")

movie_counts = ratings.groupby('movie_id').size()
keep_movies = movie_counts[movie_counts >= MOVIE_MIN_RATINGS].index
ratings = ratings[ratings['movie_id'].isin(keep_movies)]

user_counts = ratings.groupby('user_id').size()
keep_users = user_counts[user_counts >= USER_MIN_RATINGS].index
ratings = ratings[ratings['user_id'].isin(keep_users)]

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

np.save('Y.npy', Y)
np.save('R.npy', R)
pd.Series(all_movie_ids, name='movie_id').to_csv('movie_ids.csv', index=False)
pd.Series(all_user_ids, name='user_id').to_csv('user_ids.csv', index=False)
movies[['movie_id', 'title', 'genres']].to_csv('movies_25m.csv', index=False)
ratings.to_csv('ratings_25m_filtered.csv', index=False)

print("\nSaved Y.npy, R.npy, movie_ids.csv, user_ids.csv, movies_25m.csv, ratings_25m_filtered.csv")
