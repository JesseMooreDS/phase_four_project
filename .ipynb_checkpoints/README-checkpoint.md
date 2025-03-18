# Movie Recommendation System

## Jesse Moore

## [LinkedIn](https://www.linkedin.com/in/jesse-moore-datascience/) | [Github]([https://github.com/JesseMooreDS)

![image](images/picking_movie.jpeg)

# Summary

Our project aims to develop a recommendation system for a new streaming service, providing users with five personalized movie recommendations. The model that we developed uses both K-nearest Neighbors (KNN) and Alternating Least Squares modeling to create our recommendation system and to provide users with 5 recommendations. 

Business and Data Understanding:
We utilized the MovieLens dataset, which contains 100,000 ratings and 3,600 tags, making it well-suited for training a recommendation system. This dataset provides rich user-movie interactions and metadata, essential for building effective recommendations.

Data Preparation:
For our modeling, the data required little / no cleaning or preparation aside from merging our ratings data with our movie data. 

Libraries: 
Aside from the standard libraries (numpy, pandas, random, matplotlib.pyplot, seaborn, warnings), we used the following libraries: zipfile for extracting our data; scipy for csr_matrices, scikit-learn for model selection, label encoding, and evaluating cosine similarity; pyspark for Alternate Least Squares (ALS) modelling; and Surprise for our K-nearest Neighbors modelling. 

Modeling:
We initially implemented an Alternating Least Squares (ALS) model using Spark for collaborative filtering. ALS decomposes the user-item matrix to uncover latent factors, optimized via hyperparameter tuning (rank, maxIter, and RegParam). We then introduced a K-Nearest Neighbors (KNN) baseline model using the Surprise package, tuning for optimal k and min_k values. This approach strategizes to combat 'cold-start' problems where users have little to no rating history, or when movies have little to no ratings. 

Evaluation:
Performance was assessed using Root Mean Squared Error (RMSE), where a lower RMSE indicates better predictive accuracy. Grid search tuning found the best model to be KNN Baseline (k=56, min_k=14) with an RMSE of 0.886, slightly outperforming the ALS baseline (RMSE 0.888). ALS remains valuable for handling cold start issues.

Insights and Limitations:
Additional user behavior tracking—such as viewing duration and repeated watches—could improve recommendations. Incorporating explicit user preferences, such as favorite genres and actors, would further refine personalization. Future iterations may explore hybrid deep learning models for enhanced performance.

# Overview

## Business Understanding

### Problem Statement

Our client is launching a new streaming service and wants to implement a recommendation system that provides users with five personalized movie recommendations.

To achieve this, we will use a hybrid approach, combining collaborative filtering and content-based filtering. This will help mitigate the cold start problem, which occurs when:

New users, who haven’t rated any content, receive poor or no recommendations.
Movies with few or no ratings are unlikely to be recommended.
Additionally, we have been tasked with delivering insights into user engagement, enabling our client to maximize engagement with their growing user base.

Evaluation Metrics
To assess the effectiveness of our recommendation system, we will use the following key metrics:

Cosine Similarity – Measures the similarity between movies based on content features. It calculates the angle between feature vectors, helping identify movies with similar characteristics.

Root Mean Squared Error (RMSE) – Evaluates the accuracy of our rating predictions by measuring the average difference between actual and predicted ratings. A lower RMSE indicates better prediction performance.
By leveraging this hybrid approach and these evaluation metrics, we aim to build a recommendation system that delivers high-quality suggestions while providing valuable insights for our client.

These concepts will be explained in further detail below.

### Business Objective


Our business objective is to enhance user enjoyment and engagement with our client's streaming service.

To achieve this, we will recommend five movies to each user that they are likely to enjoy. Success will be measured by ensuring that our recommendations receive higher rating scores compared to a baseline model.

# Data

## Data Understanding

### Data Source

Our dataset is the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset that has been created by the GroupLens research lab at the University of Minnesota. This dataset contains 100,000 ratings, 3,600 tags and was last updated on 9/2018. While the dataset contains several csv files of data, such as 'tags.csv' (which contains user created tags for movies) and 'link.csv' (which contains keys to link our data with imdb and tmdb information), we will not be utilizing this data, instead focusing on the 'movies.csv' and 'ratings.csv' datafiles.

### Data Description

movies.csv - contains the nearly 10,000 movies that have been rated, their title and genre.

- movieId - the Id of the movie, this will used to be merge this information with other data. 
- title - the title of the movie, also generally contains the year of the movie.
- genres -  the genres that the movie comprises. This data will be transformed later, to split genre tags such as 'Action/Adventure/Animation' into their individual components, and will be explained later in this notebook. 

ratings.csv - includes the over 100,000 user ratings.

- userId - the Id of the user who left the rating, this will be used to merge with other data.
- movieId - the Id of the movie, we will use this Id to merge with other data.
- rating - the rating, ranging from 0.5 to 5.0.
- timestamp - the time when the rating was left. 

### Data Cleaning

There was little data cleaning that was done in this notebook, as we only utilized the movies.csv and ratings.csv for our modeling. This data was organized and needed no cleaning. (Note - in our exploratory notebook we have done more extensive cleaning that can be used to implement further testing in the future, see Limitations). 

### Data Preparation

There was little data cleaning that was done in this notebook, as we only utilized the movies.csv and ratings.csv for our modeling. We chose to In future implementations of our model, removing users who have left too few ratings, or movies that have too few ratings can considered to be removed from our modelling data.

## Data Analysis

Our ratings_df has no null values and contains 100,836 ratings. The mean rating is 3.5, with a standard deviation of 1.0, indicating that most ratings fall between 2.5 and 3.5. Specifically, 25% of the ratings are at or below 3.0, 50% are at or below 3.5, and 75% are at or below 4.0. 

### Matrix Sparsity Analysis

Before we can build our hybrid recommender system, we must analyze the matrix sparsity to ensure that the majority of our users have rated more than very few items. While we will build a hybrid recommender system that factors for cold-start issues, we must also contend with a model that will not be able to generalize predictions if there is too much sparsity. Too little, and we may have bias in our dataset. 

Sparsity is calculated as: sparsity = 1 - no# of non-zero interactions / total possible interactions

The results of our matrix sparsity analysis indicates that are dataset is a highly sparse dataset with shape: (610, 9724) and the sparsity of the interaction matrix is 98.30%. Because of the highly sparse nature of our dataset, and because we have explicit ratings, we will utilize a hybrid recommendation system using both KNN and ALS models. 

![image](images/user_interaction_distro.png)

![image](images/item_interaction_distro.png)

This 'long-tail' distribution shows us that the majority of our movies have fewer or no ratings/interactions, while a small amount of popular have a large number of interactions. This can lead to cold-start problems, when the system fails to recommend an item because it has too little information to go on. We will use a hybrid recommendation system to account for this.

![image](images/interaction_heatmap.png)

Since the majority of our heatmap shows off-white, this represents 0 or near-zero values in our dataset, while the darker zones represent larger values. This indicates a highly sparse dataset. 

This shows us that, currently, every user has interacted with at least one item, and every item has been interacted with by at least one user. 

# Modeling

# Alternating Least Squares

We began with Alternating Least Squares (ALS) using Apache Spark as our baseline model. ALS is a collaborative filtering method that factorizes the user-item matrix into two smaller matrices, capturing latent factors of users and items. The model iteratively optimizes one matrix while fixing the other, minimizing the least squared error. We tuned the following hyperparameters:

* rank (number of latent factors),
* maxIter (maximum iterations for optimization),
* RegParam (regularization to prevent overfitting).

While ALS effectively handles the cold-start problem by learning from implicit feedback, it struggled with accuracy for well-established users with substantial rating histories.

# K-nearest Neighbors

To improve recommendations, we implemented a K-Nearest Neighbors (KNN) Baseline model using Surprise. KNN finds the K most similar users or items and predicts ratings based on their aggregated preferences. We tuned:

* k (number of neighbors considered),
* min_k (minimum neighbors required for aggregation).

To optimize both models, we performed grid search, systematically testing hyperparameter combinations to find the best-performing configuration. Our final KNN Baseline model (with k=56 and min_k=14) achieved an RMSE of 0.886, outperforming ALS (RMSE = 0.888). The improvement, while marginal, suggests KNN's advantage in leveraging explicit user rating similarities, making it more effective for established users.

However, KNN struggles with sparsity, making ALS a valuable fallback for new users or unrated items. A hybrid system leveraging both models balances accuracy and cold-start mitigation.

## Evaluation 

Our hybrid recommendation system works, with our KNN model providing an RMSE score of: 0.8649660909040028
while our ALS model has a score of: 0.8863465869481534

When a user has a sufficient user rating history, this model on average provides movies recommended to users within .88 of their predicted rating, however, in cases where the user has little to no user rating history or a movie no too few ratings, we cannot accurately predict what their rating will be. In such cases, our ALS model will be used to recommend movies based upon latent factors. 

During model development, we identified several data limitations that impact recommendation accuracy, particularly in addressing the cold-start problem. Additional user information—such as favorite genres, actors, or directors—could improve recommendations for new users. Tracking unrated movies, watch duration (e.g., whether a user finished a movie or stopped after a few minutes), and rewatch frequency would provide deeper insights into viewing habits. Incorporating these behavioral signals could enhance prediction accuracy and better capture user preferences beyond explicit ratings.

## Conclusion

Implementing a hybrid recommendation system, we combined the strengths of ALS and KNN Baseline to deliver personalized movie suggestions while addressing the cold-start problem. Our grid search optimization identified KNN as the best-performing model (RMSE = 0.886), outperforming the ALS baseline (RMSE = 0.888). While KNN excels for users with rich rating histories, ALS remains essential for recommending movies to new users or sparsely rated items.

Further improvements could be made by incorporating additional user behavior data—such as watch duration, rewatch frequency, and unrated views—to refine predictions. With this hybrid approach, our client can maximize user engagement and deliver high-quality recommendations as their streaming platform grows.

For further details, please refer to the following linked project notebook and presentation:

[project](https://github.com/JesseMooreDS/phase_four_project) | [notebook](https://github.com/JesseMooreDS/phase_four_project/blob/main/project.ipynb) | [presentation](https://github.com/JesseMooreDS/phase_four_project/blob/main/presentation.pdf)
