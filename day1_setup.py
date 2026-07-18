"""
DAY 1: Movie Recommender System - Setup & Data Exploration
============================================================

STEP 0: GET THE DATA (do this first, manually)
------------------------------------------------
1. Go to: https://grouplens.org/datasets/movielens/
2. Download "ml-100k.zip" (the small 100K dataset, ~5MB)
3. Unzip it into the same folder as this script, so you have:
     ml-100k/u.data
     ml-100k/u.item
     ml-100k/u.user

Install dependencies first:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# STEP 1: Load the data
# -----------------------------------------------------------

# u.data: user_id, movie_id, rating, timestamp (tab-separated, no header)
ratings = pd.read_csv(
    'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# u.item: movie_id, title, release_date, ... (pipe-separated, latin-1 encoding)
movies = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1],
    names=['movie_id', 'title']
)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print(ratings.head())
print(movies.head())

# -----------------------------------------------------------
# STEP 2: Quick exploration (keep this brief, 1-2 hrs max)
# -----------------------------------------------------------

# Distribution of ratings (1-5 stars)
plt.figure()
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution.png')
plt.close()

n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()
n_ratings = len(ratings)

print(f"\nUsers: {n_users}, Movies: {n_movies}, Ratings: {n_ratings}")

# Sparsity: what % of the user-movie matrix is actually filled in?
possible_ratings = n_users * n_movies
sparsity = 1 - (n_ratings / possible_ratings)
print(f"Sparsity: {sparsity:.2%} of the matrix is EMPTY (this is normal and expected)")

# Top 10 most-rated movies (worth mentioning in your writeup)
most_rated = (
    ratings.groupby('movie_id').size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name='num_ratings')
    .merge(movies, on='movie_id')
)
print("\nTop 10 most-rated movies:")
print(most_rated[['title', 'num_ratings']])

# -----------------------------------------------------------
# STEP 3: Build the user-movie rating matrix
# (mirrors the Y matrix from Andrew Ng's course:
#  rows = movies, columns = users)
# -----------------------------------------------------------

# Pivot: rows=movie_id, cols=user_id, values=rating
R_df = ratings.pivot(index='movie_id', columns='user_id', values='rating')

# Y: the rating matrix, with 0 where there's no rating
Y = R_df.fillna(0).values

# R: binary mask, 1 if the movie WAS rated by that user, 0 otherwise
R = (~R_df.isna()).astype(int).values

print(f"\nY shape: {Y.shape}  (movies x users)")
print(f"R shape: {R.shape}  (1 = rated, 0 = not rated)")

# -----------------------------------------------------------
# STEP 4: Non-personalized baseline recommender
# (this is your "before" comparison point for later)
# -----------------------------------------------------------

def baseline_recommend(n=5):
    """Recommend the n highest-average-rated movies (min 20 ratings to avoid noise)."""
    stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= 20]  # filter out movies with too few ratings
    top_n = stats.sort_values('mean', ascending=False).head(n)
    result = top_n.merge(movies, on='movie_id')
    return result[['title', 'mean', 'count']]

print("\nBaseline recommendations (top-rated movies overall):")
print(baseline_recommend(5))

# -----------------------------------------------------------
# Save processed data for Day 2+ (collaborative filtering)
# -----------------------------------------------------------
np.save('Y.npy', Y)
np.save('R.npy', R)
R_df.index.to_series().to_csv('movie_ids.csv', index=False)
print("\nSaved Y.npy, R.npy, movie_ids.csv for tomorrow's collaborative filtering step.")
