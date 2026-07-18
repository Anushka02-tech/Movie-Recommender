# Movie Recommender System

A collaborative filtering recommender system built from scratch, based on the algorithm taught in Andrew Ng's Machine Learning course (Coursera). Trained on the MovieLens 100K dataset.

## Problem Statement

Given a user's movie ratings, predict how they'd rate movies they haven't seen yet, and recommend the ones they're most likely to enjoy. This is the same problem behind real-world systems like Netflix's or Amazon's recommendation engines.

## Dataset

- **MovieLens 100K** (GroupLens Research)
- 943 users, 1,682 movies, 100,000 ratings
- Ratings are on a 1-5 star scale
- Matrix sparsity: 93.7% (i.e., only 6.3% of all possible user-movie ratings actually exist) — this sparsity is the core challenge collaborative filtering is designed to handle

## Approach

### Collaborative Filtering (from scratch)

Implemented the collaborative filtering algorithm in raw NumPy, following the notation and math from Andrew Ng's course (Recommender Systems module):

- Each movie is represented by a learned feature vector `X[i]` (10-15 latent dimensions)
- Each user is represented by a learned feature vector `Theta[j]`
- Predicted rating for user `j` on movie `i` is the dot product `X[i] · Theta[j]`
- Trained via **gradient descent** to minimize squared error on known ratings, with **L2 regularization** to prevent overfitting

**Key implementation details:**
- Ratings were **mean-centered per movie** before training, so the model can make reasonable predictions even for users with very few ratings
- Final hyperparameters: `num_features=15`, `lambda=15` (regularization), `alpha=0.001` (learning rate), `1000` iterations
- Recommendations exclude movies with fewer than 20 total ratings, to avoid surfacing obscure titles the model may have overfit on

### Cold-Start Handling

Collaborative filtering can't make reliable predictions for users with no (or very few) ratings, since their feature vector `Theta` is essentially untrained. To handle this:
- Users with fewer than 5 ratings (including brand-new users) automatically fall back to a **popularity-based recommender** (highest average-rated movies with at least 20 ratings)
- This fallback is clearly labeled in the app so it's transparent which method produced a given recommendation

Note: the MovieLens 100K dataset guarantees every included user has 20+ ratings, so this fallback path was validated using simulated new users rather than real sparse users in the dataset.

### Baseline for Comparison

A simple **non-personalized baseline** was built for comparison: recommend each movie's average rating across all users, regardless of who's asking. This represents "what you'd do with no machine learning at all."

## Results

Evaluated using an 80/20 train/test split (80,000 training ratings, 20,000 held-out test ratings), measuring **RMSE** (Root Mean Squared Error — lower is better):

| Model | Test RMSE |
|---|---|
| Baseline (movie average) | 1.0264 |
| Collaborative Filtering (this project) | **0.9459** |

**Collaborative filtering improved prediction accuracy by 7.8% over the baseline.**

Hyperparameter note: `num_features` was tested at both 10 and 15; the difference in RMSE was negligible (0.9477 vs 0.9459), suggesting the model had already captured most of the available signal in the data at 10 features.

## Interface

A **Streamlit** web app provides an interactive demo:
- Select any existing user and view their top-rated movies alongside personalized recommendations
- Toggle to "New user" mode to see the cold-start fallback in action
- Adjustable number of recommendations (3-15)

## What I'd Improve With More Time

- **Content-based fallback**: instead of pure popularity for cold-start users, incorporate movie genre/metadata to personalize recommendations even without rating history
- **Cross-validation**: currently using a single 80/20 split; k-fold cross-validation would give a more robust estimate of RMSE
- **Library comparison**: benchmark against `scikit-surprise`'s SVD implementation to validate the from-scratch model against an established library
- **Scalability**: the current implementation loads the full ratings matrix into memory, which won't scale to datasets with millions of users/movies (a production system would need sparse matrix representations or a different architecture, e.g. ALS on Spark)
- **Precision@K / Recall@K**: RMSE measures rating prediction accuracy, but doesn't directly measure whether the *top-N recommendations* are actually good — ranking-specific metrics would complement it

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib streamlit

# 2. Download the MovieLens 100K dataset from https://grouplens.org/datasets/movielens/
#    and unzip it into this folder as ml-100k/

# 3. Run the pipeline in order
python day1_setup.py                          # data loading, exploration, baseline
python day3to5_collaborative_filtering.py      # train the model, generate recommendations
python day8to9_evaluation.py                   # RMSE evaluation vs. baseline
python day10_cold_start.py                     # test cold-start fallback logic

# 4. Launch the interactive demo
streamlit run app.py
```

## Project Structure

```
├── day1_setup.py                        # Data loading, exploration, baseline recommender
├── day3to5_collaborative_filtering.py   # Collaborative filtering from scratch (gradient descent)
├── day8to9_evaluation.py                # Train/test split + RMSE evaluation
├── day10_cold_start.py                  # Cold-start detection and fallback logic
├── app.py                               # Streamlit interactive demo
└── README.md
```

## Acknowledgments

- Algorithm and notation based on Andrew Ng's Machine Learning course (Coursera / Stanford)
- Dataset: MovieLens 100K, provided by GroupLens Research, University of Minnesota
