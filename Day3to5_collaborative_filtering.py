"""
DAYS 3-5: Collaborative Filtering From Scratch
================================================
Implements the exact algorithm from Andrew Ng's Machine Learning course
(Week 9 / Recommender Systems module):

    Cost function:
        J = 1/2 * sum over (i,j) where R(i,j)=1 of (X[i] . Theta[j] - Y[i,j])^2
            + (lambda/2) * sum(Theta^2)
            + (lambda/2) * sum(X^2)

    X     = movie feature matrix (num_movies x num_features)
    Theta = user feature matrix  (num_users x num_features)
    Y     = ratings matrix (movies x users), 0 where unrated
    R     = binary mask, 1 if rated

We learn X and Theta via gradient descent so that X[i] . Theta[j]
approximates the rating user j would give movie i.

Run this AFTER day1_setup.py (it needs Y.npy and R.npy).
"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# STEP 1: Load data saved from Day 1
# -----------------------------------------------------------

Y = np.load('Y.npy')   # shape: (num_movies, num_users)
R = np.load('R.npy')   # shape: (num_movies, num_users), 1 = rated

num_movies, num_users = Y.shape
print(f"Loaded Y: {Y.shape}, R: {R.shape}")

# -----------------------------------------------------------
# STEP 2: Mean-normalize the ratings (IMPORTANT — easy to forget)
# -----------------------------------------------------------
# Without this, users who haven't rated anything would just get
# predictions of 0 for everything, which is meaningless.
# We center each movie's ratings around its own mean rating,
# then add the mean back when predicting.

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

# -----------------------------------------------------------
# STEP 3: Cost function + gradients
# -----------------------------------------------------------

def cost_function(params, Y, R, num_users, num_movies, num_features, lam):
    """
    params: flattened array containing X and Theta concatenated
    Returns: cost J, and gradient (also flattened, same shape as params)
    """
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    # Predicted ratings matrix
    predictions = X @ Theta.T

    # Error, but only where R=1 (i.e. only where a rating actually exists)
    error = (predictions - Y) * R

    # Cost: squared error term + regularization terms
    J = 0.5 * np.sum(error ** 2)
    J += (lam / 2) * np.sum(Theta ** 2)
    J += (lam / 2) * np.sum(X ** 2)

    # Gradients
    X_grad = error @ Theta + lam * X
    Theta_grad = error.T @ X + lam * Theta

    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad

# -----------------------------------------------------------
# STEP 4: Train with gradient descent
# -----------------------------------------------------------

def train_collaborative_filtering(Y, R, num_features=10, lam=10, alpha=0.001,
                                   num_iters=200, seed=42):
    """
    num_features: size of the latent feature vectors (10 is a reasonable start)
    lam:          regularization strength
    alpha:        learning rate
    num_iters:    number of gradient descent iterations
    """
    np.random.seed(seed)
    num_movies, num_users = Y.shape

    # Initialize X and Theta randomly (small values, like the course does)
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

print("\nTraining collaborative filtering model on the larger dataset (may take a few minutes)...")
X, Theta, costs = train_collaborative_filtering(
    Y_norm, R,
    num_features=15,
    lam=15,
    alpha=0.0007,
    num_iters=1500
)

# -----------------------------------------------------------
# STEP 5: Generate predictions and recommendations
# -----------------------------------------------------------

# Full predicted ratings matrix, adding back the per-movie mean we subtracted
predictions = X @ Theta.T + Y_mean.reshape(-1, 1)

movie_ids = pd.read_csv('movie_ids.csv')['movie_id'].values
movies = pd.read_csv('movies_25m.csv')[['movie_id', 'title']]

def recommend(user_idx, n=5, min_ratings=20):
    """
    user_idx: 0-based column index into Y/predictions (NOT the raw user_id).
              user_id 1 in the raw data corresponds to user_idx 0 here.
    min_ratings: filter out movies with fewer than this many total ratings,
                 to avoid recommending obscure movies the model overfit on.
    """
    user_ratings = predictions[:, user_idx]
    already_rated = R[:, user_idx] == 1
    too_obscure = R.sum(axis=1) < min_ratings

    # Mask out movies the user already rated AND movies with too few ratings
    scores = user_ratings.copy()
    scores[already_rated] = -np.inf
    scores[too_obscure] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]
    top_movie_ids = movie_ids[top_indices]
    top_scores = scores[top_indices]

    result = pd.DataFrame({'movie_id': top_movie_ids, 'predicted_rating': top_scores})
    result = result.merge(movies, on='movie_id')
    return result[['title', 'predicted_rating']]

# Sanity check: recommendations for user_id = 1 (user_idx = 0)
print("\nTop 5 recommendations for user_id=1:")
print(recommend(user_idx=0, n=5))

print("\nTop 5 recommendations for user_id=50:")
print(recommend(user_idx=49, n=5))

# -----------------------------------------------------------
# Save everything for Week 2 (evaluation + Streamlit app)
# -----------------------------------------------------------
np.save('X_trained.npy', X)
np.save('Theta_trained.npy', Theta)
np.save('Y_mean.npy', Y_mean)
print("\nSaved X_trained.npy, Theta_trained.npy, Y_mean.npy for evaluation step.")