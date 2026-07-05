# genre-clusters

# Spotify Music Genre Clustering

An unsupervised machine learning project that implements the **K-Means clustering algorithm from scratch** to classify songs into their respective genres using Spotify audio features (such as tempo, energy, and danceability).

## Features
* **Custom K-Means:** Implemented centroid initialization, distance computation, and updates entirely from scratch.
* **Hungarian Algorithm:** Utilized `scipy.optimize.linear_sum_assignment` to find the optimal mapping between unsupervised clusters and actual genre labels.
* **PCA Visualization:** Projected 14-dimensional audio features into a 2D space for visual cluster evaluation.

## Dataset & Metrics
* **Dataset:** Audio features from classical, pop, and country tracks.
* **Overall Precision:** 93% across genres.
* **Max Accuracy:** 96% accuracy achieved on Classical and Hip-Hop tracks.

## Requirements
To run the code, you will need:
* numpy
* pandas
* matplotlib
* scipy
* scikit-learn

## How to Run
1. Make sure your dataset is saved in `FinalProjectData/classical-pop-country-tracks.csv`.
2. Run the main script:
```bash
python MachineLearningFinalProjectCode.py
