"""
Evaluation: Train/Test Split + RMSE
======================================
Holds out a test set of ratings, trains only on the rest, then measures
RMSE (Root Mean Squared Error) between predicted and actual held-out
ratings. Lower is better; a perfect model scores 0.

Requires ratings_25m_filtered.csv from day1_setup.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------------------------------------
# Load raw ratings (not the pre-built Y/R this time,
# because the split needs to happen before building the matrix)
# -----------------------------------------------------------

ratings = pd.read_csv('ratings_25m_filtered.csv')

# -----------------------------------------------------------
# Train/test split (80/20)
# -----------------------------------------------------------

shuffled = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(0.8 * len(shuffled))
train_df = shuffled.iloc[:split_point]
test_df = shuffled.iloc[split_point:]

print(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")

# -----------------------------------------------------------
# Build Y_train / R_train from the training set only
# (test ratings are treated as if they don't exist, during training)
# -----------------------------------------------------------

all_movie_ids = sorted(ratings['movie_id'].unique())
all_user_ids = sorted(ratings['user_id'].unique())
movie_id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}
user_id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}

num_movies = len(all_movie_ids)
num_users = len(all_user_ids)

Y_train = np.zeros((num_movies, num_users))
R_train = np.zeros((num_movies, num_users))

train_movie_indices = train_df['movie_id'].map(movie_id_to_idx).values
train_user_indices = train_df['user_id'].map(user_id_to_idx).values
Y_train[train_movie_indices, train_user_indices] = train_df['rating'].values
R_train[train_movie_indices, train_user_indices] = 1

print(f"Y_train shape: {Y_train.shape}")

# -----------------------------------------------------------
# Mean-normalize (using training data only)
# -----------------------------------------------------------

def normalize_ratings(Y, R):
    m = Y.shape[0]
    Y_mean = np.zeros(m)
    Y_norm = np.zeros(Y.shape)
    for i in range(m):
        rated_idx = np.where(R[i, :] == 1)[0]
        if len(rated_idx) > 0:
            Y_mean[i] = np.mean(Y[i, rated_idx])
            Y_norm[i, rated_idx] = Y[i, rated_idx] - Y_mean[i]
    return Y_norm, Y_mean

Y_norm, Y_mean = normalize_ratings(Y_train, R_train)

# -----------------------------------------------------------
# Train the model (same approach as the full training script)
# -----------------------------------------------------------

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

def train_collaborative_filtering(Y, R, num_features=10, lam=15, alpha=0.001,
                                   num_iters=1000, seed=42):
    np.random.seed(seed)
    num_movies, num_users = Y.shape
    X = np.random.randn(num_movies, num_features) * 0.1
    Theta = np.random.randn(num_users, num_features) * 0.1
    params = np.concatenate([X.ravel(), Theta.ravel()])
    for i in range(num_iters):
        J, grad = cost_function(params, Y, R, num_users, num_movies, num_features, lam)
        params = params - alpha * grad
        if i % 100 == 0 or i == num_iters - 1:
            print(f"Iteration {i:4d} | Cost: {J:,.2f}")
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    return X, Theta

print("\nTraining on train set only (holding out 20% for testing)...")
X, Theta = train_collaborative_filtering(Y_norm, R_train, num_features=15,
                                          lam=15, alpha=0.0007, num_iters=1500)

predictions = X @ Theta.T + Y_mean.reshape(-1, 1)

# -----------------------------------------------------------
# Evaluate on the held-out test set (ratings the model never saw)
# -----------------------------------------------------------

def rmse(predictions, df, movie_id_to_idx, user_id_to_idx):
    movie_indices = df['movie_id'].map(movie_id_to_idx).values
    user_indices = df['user_id'].map(user_id_to_idx).values
    preds = predictions[movie_indices, user_indices]
    actuals = df['rating'].values
    return np.sqrt(np.mean((preds - actuals) ** 2))

cf_rmse = rmse(predictions, test_df, movie_id_to_idx, user_id_to_idx)
print(f"\nCollaborative Filtering RMSE on test set: {cf_rmse:.4f}")

# -----------------------------------------------------------
# Compare against the baseline (predict each movie's
# average rating from the TRAINING set, for every user)
# -----------------------------------------------------------

train_movie_means = train_df.groupby('movie_id')['rating'].mean()
global_mean = train_df['rating'].mean()  # fallback for movies unseen in training

test_baseline_preds = test_df['movie_id'].map(train_movie_means).fillna(global_mean).values
baseline_rmse = np.sqrt(np.mean((test_baseline_preds - test_df['rating'].values) ** 2))
print(f"Baseline (movie average) RMSE on test set: {baseline_rmse:.4f}")

# -----------------------------------------------------------
# Summary
# -----------------------------------------------------------

improvement = (baseline_rmse - cf_rmse) / baseline_rmse * 100
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Baseline RMSE:               {baseline_rmse:.4f}")
print(f"Collaborative Filtering RMSE: {cf_rmse:.4f}")
if cf_rmse < baseline_rmse:
    print(f"Collaborative filtering improved RMSE by {improvement:.1f}% over baseline.")
else:
    print(f"Collaborative filtering did NOT beat the baseline this run.")
    print(f"Try: more iterations, tuning lambda, or more features.")