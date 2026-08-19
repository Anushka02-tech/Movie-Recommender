import numpy as np
import pandas as pd

Y = np.load('Y.npy')  
R = np.load('R.npy')   

num_movies, num_users = Y.shape
print(f"Loaded Y: {Y.shape}, R: {R.shape}")

def normalize_ratings(Y, R):
    """Returns Y_mean (per movie) and Y_norm (mean-centered ratings)."""
    m = Y.shape[0]
    Y_mean = np.zeros(m)
    Y_norm = np.zeros(Y.shape)
    for i in range(m):
        rated_idx = np.where(R[i, :] == 1)[0]
        if len(rated_idx) > 0:
            Y_mean[i] = np.mean(Y[i, rated_idx])
            Y_norm[i, rated_idx] = Y[i, rated_idx] - Y_mean[i]
    return Y_norm, Y_mean

Y_norm, Y_mean = normalize_ratings(Y, R)
print("Ratings mean-normalized per movie.")

def cost_function(params, Y, R, num_users, num_movies, num_features, lam):
 
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    predictions = X @ Theta.T

    error = (predictions - Y) * R

    J = 0.5 * np.sum(error ** 2)
    J += (lam / 2) * np.sum(Theta ** 2)
    J += (lam / 2) * np.sum(X ** 2)

    X_grad = error @ Theta + lam * X
    Theta_grad = error.T @ X + lam * Theta

    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad

def train_collaborative_filtering(Y, R, num_features=10, lam=10, alpha=0.001,
                                   num_iters=200, seed=42):
 
    np.random.seed(seed)
    num_movies, num_users = Y.shape

    X = np.random.randn(num_movies, num_features) * 0.1
    Theta = np.random.randn(num_users, num_features) * 0.1

    params = np.concatenate([X.ravel(), Theta.ravel()])

    costs = []
    for i in range(num_iters):
        J, grad = cost_function(params, Y, R, num_users, num_movies, num_features, lam)
        params = params - alpha * grad
        costs.append(J)
        if i % 20 == 0 or i == num_iters - 1:
            print(f"Iteration {i:4d} | Cost: {J:,.2f}")

    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    return X, Theta, costs

print("\nTraining collaborative filtering model (this may take a few minutes)...")
X, Theta, costs = train_collaborative_filtering(
    Y_norm, R,
    num_features=15,
    lam=15,
    alpha=0.0007,
    num_iters=1500
)

predictions = X @ Theta.T + Y_mean.reshape(-1, 1)

movie_ids = pd.read_csv('movie_ids.csv')['movie_id'].values
movies = pd.read_csv('movies_25m.csv')[['movie_id', 'title']]

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

print("\nTop 5 recommendations for user_id=1:")
print(recommend(user_idx=0, n=5))

print("\nTop 5 recommendations for user_id=50:")
print(recommend(user_idx=49, n=5))

np.save('X_trained.npy', X)
np.save('Theta_trained.npy', Theta)
np.save('Y_mean.npy', Y_mean)
print("\nSaved X_trained.npy, Theta_trained.npy, Y_mean.npy for evaluation.")