import numpy as np
import pandas as pd

MIN_RATINGS_THRESHOLD = 5  

movies = pd.read_csv('movies_25m.csv')[['movie_id', 'title']]
ratings = pd.read_csv('ratings_25m_filtered.csv')
movie_ids = pd.read_csv('movie_ids.csv')['movie_id'].values
X = np.load('X_trained.npy')
Theta = np.load('Theta_trained.npy')
Y_mean = np.load('Y_mean.npy')
R = np.load('R.npy')

predictions = X @ Theta.T + Y_mean.reshape(-1, 1)

def popularity_recommend(n=5, min_ratings=20, exclude_movie_ids=None):
    """Top-rated movies overall, ignoring the user entirely."""
    stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= min_ratings]
    if exclude_movie_ids is not None:
        stats = stats[~stats.index.isin(exclude_movie_ids)]
    top_n = stats.sort_values('mean', ascending=False).head(n)
    result = top_n.merge(movies, on='movie_id')
    return result[['title', 'mean', 'count']].rename(columns={'mean': 'avg_rating'})

def personalized_recommend(user_idx, n=5, min_ratings=20):
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

def recommend_with_cold_start(user_id, all_user_ids, n=5):

    num_user_ratings = (ratings['user_id'] == user_id).sum()

    if user_id not in all_user_ids or num_user_ratings < MIN_RATINGS_THRESHOLD:
        recs = popularity_recommend(n=n)
        return recs, "popularity-based (cold-start fallback)"
    else:
        user_idx = list(all_user_ids).index(user_id)
        recs = personalized_recommend(user_idx, n=n)
        return recs, "personalized (collaborative filtering)"

all_user_ids = sorted(ratings['user_id'].unique())
real_existing_user = all_user_ids[0] 

print(f"--- User {real_existing_user} (existing user, plenty of ratings) ---")
recs, method = recommend_with_cold_start(user_id=real_existing_user, all_user_ids=all_user_ids, n=5)
print(f"Method used: {method}")
print(recs)

fake_user_id = max(all_user_ids) + 9999 
print(f"\n--- User {fake_user_id} (simulated brand-new user) ---")
recs, method = recommend_with_cold_start(user_id=fake_user_id, all_user_ids=all_user_ids, n=5)
print(f"Method used: {method}")
print(recs)

rating_counts = ratings.groupby('user_id').size()
sparse_users = rating_counts[rating_counts < MIN_RATINGS_THRESHOLD]
if len(sparse_users) > 0:
    sparse_user_id = sparse_users.index[0]
    print(f"\n--- User {sparse_user_id} (real user, only {sparse_users.iloc[0]} ratings) ---")
    recs, method = recommend_with_cold_start(user_id=sparse_user_id, all_user_ids=all_user_ids, n=5)
    print(f"Method used: {method}")
    print(recs)
else:
    print(f"\n(No real users found with fewer than {MIN_RATINGS_THRESHOLD} ratings in this dataset "
          f"-- this is expected since we filtered to users with 50+ ratings when building the "
          f"ml-25m subset, so cold-start only triggers for genuinely new users, which is "
          f"realistic and worth mentioning in your report.)")