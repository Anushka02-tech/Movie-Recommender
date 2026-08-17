import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ratings = pd.read_csv(
    'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

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

possible_ratings = n_users * n_movies
sparsity = 1 - (n_ratings / possible_ratings)
print(f"Sparsity: {sparsity:.2%} of the matrix is EMPTY (this is normal and expected)")

most_rated = (
    ratings.groupby('movie_id').size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name='num_ratings')
    .merge(movies, on='movie_id')
)
print("\nTop 10 most-rated movies:")
print(most_rated[['title', 'num_ratings']])

R_df = ratings.pivot(index='movie_id', columns='user_id', values='rating')

Y = R_df.fillna(0).values

R = (~R_df.isna()).astype(int).values

print(f"\nY shape: {Y.shape}  (movies x users)")
print(f"R shape: {R.shape}  (1 = rated, 0 = not rated)")

def baseline_recommend(n=5):
    stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= 20]  # filter out movies with too few ratings
    top_n = stats.sort_values('mean', ascending=False).head(n)
    result = top_n.merge(movies, on='movie_id')
    return result[['title', 'mean', 'count']]

print("\nBaseline recommendations (top-rated movies overall):")
print(baseline_recommend(5))

np.save('Y.npy', Y)
np.save('R.npy', R)
R_df.index.to_series().to_csv('movie_ids.csv', index=False)
print("\nSaved Y.npy, R.npy, movie_ids.csv for the collaborative filtering step.")
