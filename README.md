# 🎬 Movie Recommendation System

A machine learning project that recommends movies based on movie genres and users' rating preferences. The project covers data exploration, preprocessing, content-based filtering, collaborative filtering, model evaluation, and deployment as an interactive **Streamlit** web app.

## 📌 Overview

With thousands of movies available, finding a movie that matches a user's interests can be challenging. This project implements two recommendation approaches: content-based filtering to discover movies similar to a selected title, and collaborative filtering to recommend movies based on the preferences of similar users.

The application provides an interactive interface where users can explore similar movies, receive personalized recommendations, and view dataset statistics.

## 📊 Dataset

This project uses the **MovieLens `ml-latest-small` dataset**, provided by GroupLens Research.

- **Movies:** 9,742
- **Ratings:** 100,836
- **Features:** Movie IDs, titles, genres, user IDs, ratings, and timestamps
- **Main files:** `movies.csv`, `ratings.csv`, and optionally `links.csv`

### Dataset files

| File | Description |
|---|---|
| `movies.csv` | Movie titles, IDs, and genres |
| `ratings.csv` | User ratings for movies and timestamps |
| `links.csv` | Links between MovieLens movie IDs and external movie IDs |

The dataset is used to build and evaluate the recommendation models.

**Dataset source:** [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/)

## ⚙️ Methodology

1. **Data exploration:** Loaded the movie and rating files, inspected their dimensions, and checked for missing values and duplicate records.
2. **Content-based filtering:** Used TF-IDF vectorization on movie genres and cosine similarity to identify movies with similar genre profiles.
3. **Collaborative filtering:** Created a user-item rating matrix and calculated user similarity to generate recommendations based on ratings from similar users.
4. **Train/test split:** Split the ratings data into training and testing sets using an 80/20 split with `random_state=42`.
5. **Recommendation generation:** Ranked candidate movies using genre similarity or estimated ratings from similar users.
6. **Evaluation:** Evaluated recommendation performance using Precision@10 and Recall@10.

## 🚀 Streamlit App

The application provides an interactive web interface with the following sections:

- **Home:** Introduction to the recommendation system.
- **Movie Recommendation:** Select a movie and discover similar titles based on genres.
- **User Recommendation:** Select a user ID to receive recommendations based on similar users' ratings.
- **Project Insights:** Explore dataset statistics, average ratings, rating distribution, and recorded evaluation metrics.

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Download and extract the MovieLens ml-latest-small dataset
# Place the ml-latest-small folder in the project root.

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📁 Project Structure

```text
├── Movie_Recommendation_Task5_Clean.ipynb  # Data analysis and evaluation
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── ml-latest-small/                        # Download separately
    ├── movies.csv
    ├── ratings.csv
    └── links.csv
```

The MovieLens dataset should be downloaded separately from the official source rather than committed to the public repository.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · TF-IDF · Cosine Similarity · Collaborative Filtering · Streamlit · Jupyter Notebook

## 📈 Results

The project notebook previously recorded the following recommendation metrics:

| Metric | Result |
|---|---:|
| Precision@10 | 2.30% |
| Recall@10 | 3.11% |

These values reflect the evaluation recorded in the notebook. They are not recalculated when the Streamlit application starts.

## 🔍 Key Insights

- Movie genres can be transformed into TF-IDF vectors to measure similarity between movies.
- Collaborative filtering uses rating patterns from similar users to estimate preferences for unseen movies.
- Combining content-based and collaborative filtering provides two complementary ways to discover movies.
- Recommendation quality depends on the evaluation setup, the available ratings, and how relevant recommendations are defined.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Recommendations are generated from the MovieLens dataset and may not reflect every user's actual preferences.

## 📄 Dataset License and Usage

The MovieLens dataset is provided by GroupLens Research and is subject to its applicable terms of use. Review the official [MovieLens dataset page](https://grouplens.org/datasets/movielens/latest/) before redistributing the dataset.

This repository documents the recommendation system code and does not need to include the dataset files.
