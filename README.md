# 🎬 Movie Recommender System

A collaborative filtering recommender system built from scratch, based on the algorithm taught in Andrew Ng's Machine Learning course (Coursera). Trained on a filtered subset of the MovieLens 25M dataset, and deployed as a live interactive web app.

**🔗 Live demo:** [movie-recommender-aiwpdnfztxxvjwgu9z95xk.streamlit.app](https://movie-recommender-aiwpdnfztxxvjwgu9z95xk.streamlit.app)

## Problem Statement

Given a user's movie ratings, predict how they'd rate movies they haven't seen yet, and recommend the ones they're most likely to enjoy. This is the same problem behind real-world systems like Netflix's or Amazon's recommendation engines.

## Dataset

- **MovieLens 25M** (GroupLens Research), filtered down to the **8,000 most active users** and **4,000 most-rated movies**, for a total of **6.46 million ratings**
- Ratings are on a 1-5 star scale
- Filtering to active users/popular movies keeps the ratings matrix dense enough for gradient descent to converge in reasonable time, while still being large enough to be a meaningful test of the approach (65x more ratings than the original MovieLens 100K)

## Approach

### Collaborative Filtering (from scratch)

Implemented the collaborative filtering algorithm in raw NumPy, following the notation and math from Andrew Ng's course (Recommender Systems module) — no `scikit-surprise`, no black-box library:

- Each movie is represented by a learned feature vector `X[i]` (15 latent dimensions)
- Each user is represented by a learned feature vector `Theta[j]`
- Predicted rating for user `j` on movie `i` is the dot product `X[i] · Theta[j]`
- Trained via **gradient descent** to minimize squared error on known ratings, with **L2 regularization** to prevent overfitting

**Key implementation details:**
- Ratings were **mean-centered per movie** before training, so the model can make reasonable predictions even for users with very few ratings
- Final hyperparameters: `num_features=15`, `lambda=15` (regularization), `alpha=0.0007` (learning rate), `1500` iterations
- Recommendations exclude movies with fewer than 20 total ratings, to avoid surfacing obscure titles the model may have overfit on

### Cold-Start Handling

Collaborative filtering can't make reliable predictions for users with no (or very few) ratings, since their feature vector `Theta` is essentially untrained. To handle this:
- Users with fewer than 5 ratings (including brand-new users) automatically fall back to a **popularity-based recommender** (highest average-rated movies with at least 20 ratings)
- New users can optionally specify favorite genres instead, which restricts the popularity fallback to matching movies rather than showing generic top-rated titles across all genres
- Every recommendation in the app is labeled with which method produced it (personalized / genre-based / popularity-based), so it's transparent and auditable

### Baseline for Comparison

A simple **non-personalized baseline** was built for comparison: predict each movie's average rating (from the training set), regardless of who's asking. This represents "what you'd do with no machine learning at all."

## Results

Evaluated using an 80/20 train/test split, measuring **RMSE** (Root Mean Squared Error — lower is better), where the model never saw the held-out ratings during training:

| Model | Test RMSE |
|---|---|
| Baseline (movie average) | 0.9220 |
| Collaborative Filtering (this project) | **0.7227** |

**Collaborative filtering improved prediction accuracy by 21.6% over the baseline.**

## Interface

A **Streamlit** web app provides an interactive, deployed demo:
- Sidebar-based controls (user selection, genre picker, number of recommendations) keep the main area focused entirely on results
- Recommendations render as a **poster grid**, with movie posters pulled live from the TMDb API — including a title-cleaning step to handle MovieLens's `", The (Year)"` formatting and original-language alt-titles so lookups match correctly
- A user's rating history / stated genre preferences are shown in a collapsible "Based on" section for context, without cluttering the results
- Loading spinner + toast notification while recommendations compute, and a friendly empty state before the first search
- Toggle to "New user" mode to see the cold-start fallback in action, either via genre matching or generic popularity

## Engineering / Deployment Notes

Beyond the modeling itself, getting this into a shareable, deployed state involved:

- **Secrets management**: the TMDb API key is read via `st.secrets` and stored in a local `.streamlit/secrets.toml` (excluded from git via `.gitignore`), and separately configured in the Streamlit Community Cloud dashboard for the deployed version — never hardcoded or committed
- **Repo size management**: the full ratings file (176MB as CSV) was converted to **Parquet** format (`convert_to_parquet.py`), shrinking it to ~42MB via columnar compression, to fit under GitHub's 100MB per-file limit
- **Memory optimization**: the ratings mask (`R.npy`), originally an 8-byte-per-entry integer array (244MB), was converted to a 1-byte boolean array (`shrink_R.py`), an 8x reduction with no change to the app's logic
- **Deployed via Streamlit Community Cloud**, connected directly to the GitHub repo, rebuilding automatically from `requirements.txt` on each push

## What I'd Improve With More Time

- **Content-based hybrid**: incorporate movie genre/metadata more deeply into the personalized model itself, not just as a cold-start fallback
- **Cross-validation**: currently using a single 80/20 split; k-fold cross-validation would give a more robust RMSE estimate
- **Library comparison**: benchmark against `scikit-surprise`'s SVD implementation to validate the from-scratch model against an established library
- **Scalability**: the current implementation loads the full dense ratings matrix into memory, which won't scale much further (a production system would need a sparse matrix representation or a different architecture, e.g. ALS on Spark)
- **Precision@K / Recall@K**: RMSE measures rating prediction accuracy, but doesn't directly measure whether the *top-N recommendations* are actually good — ranking-specific metrics would complement it
- **User accounts**: currently users are selected from a dropdown of synthetic display names rather than an authenticated login, which is appropriate for a demo but not for a real product

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MovieLens 25M from https://grouplens.org/datasets/movielens/
#    and run the pipeline in order to regenerate all processed files
python day1_setup.py                           # data loading, exploration, baseline
python Day3to5_collaborative_filtering.py      # train the model, generate recommendations
python Day8to9_evaluation.py                   # RMSE evaluation vs. baseline
python day10_cold_start.py                     # test cold-start fallback logic
python Generate_user_names.py                  # generate synthetic display names
python convert_to_parquet.py                   # compress ratings file for deployment
python shrink_R.py                             # shrink the ratings mask to boolean

# 3. Add your TMDb API key (get one free at themoviedb.org)
#    Create .streamlit/secrets.toml with:
#    tmdb_api_key = "your_key_here"

# 4. Launch the interactive demo
streamlit run app.py
```

## Project Structure

```
├── day1_setup.py                          # Data loading, exploration, baseline recommender
├── Day3to5_collaborative_filtering.py     # Collaborative filtering from scratch (gradient descent)
├── Day8to9_evaluation.py                  # Train/test split + RMSE evaluation
├── day10_cold_start.py                    # Cold-start detection and fallback logic
├── Generate_user_names.py                 # Synthetic display name generation for demo users
├── convert_to_parquet.py                  # Compresses ratings CSV to Parquet for deployment
├── shrink_R.py                            # Converts the ratings mask to a boolean array
├── app.py                                 # Streamlit interactive demo (deployed)
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Excludes secrets and raw datasets
└── README.md
```

## Acknowledgments

- Algorithm and notation based on Andrew Ng's Machine Learning course (Coursera / Stanford)
- Dataset: MovieLens 25M, provided by GroupLens Research, University of Minnesota
- Poster images via [The Movie Database (TMDb)](https://www.themoviedb.org/) API
