# Film Recommendation System

## Overview
This project implements a scalable movie recommendation system using collaborative filtering and big data technologies. The recommendation engine is powered by the Alternating Least Squares (ALS) algorithm, integrated with Apache Spark for data processing. The user interface is built with Django and provides personalized recommendations and visualizations.

## Architecture

1. **Data Loading**:
   - Datasets are loaded into Spark DataFrames.
   - Missing data is cleaned, and transformations are applied as necessary.

2. **Data Exploration**:
   - Distribution of ratings.
   - Most popular movies.
   - User rating statistics.

3. **Model Creation**:
   - ALS algorithm is used for collaborative filtering.
   - Parameters are optimized using cross-validation.

4. **Recommendations**:
   - Personalized movie recommendations are generated based on user inputs.

5. **Visualization**:
   - Results are visualized with interactive Plotly charts on the web interface.

---

## Features

- Interactive data visualization using Plotly.
- Personalized movie recommendations.
- Scalable data processing using Apache Spark.
- User-friendly web application built with Django.

---

## Dataset

1. **Files**:
   - `rating.csv`: User ratings for movies.
   - `movie.csv`: Movie details.

2. **Structure**:
   - `rating.csv` contains:
     - `userId`: Unique identifier for users.
     - `movieId`: Unique identifier for movies.
     - `rating`: User's rating for the movie.
   - `movie.csv` contains:
     - `movieId`: Unique identifier for movies.
     - `title`: Movie title.

3. **Data Cleaning**:
   - Missing values removed using `na.drop`.
   - Schema inferred automatically by Spark.

---

## Technologies Used

1. **Apache Spark**:
   - For big data analytics and machine learning.
   - Key components:
     - `SparkSession`: Manages Spark operations.
     - `MLlib`: Provides machine learning algorithms.

2. **Django**:
   - Framework for the web application.

3. **Hadoop**:
   - Configured for Spark’s file system integration.

4. **Python Libraries**:
   - `Plotly`: Interactive visualizations.
   - `pandas`: Data manipulation.
   - `pickle`: Model serialization.
   - `logging`: Error tracking.

---

## Model Details

1. **Algorithm**:
   - Alternating Least Squares (ALS):
     - Collaborative filtering algorithm.
     - Predicts latent factors for users and movies.

2. **Parameters**:
   - `maxIter=10`: Maximum iterations.
   - `rank=10`: Number of latent factors.
   - `regParam=0.01`: Regularization parameter.
   - `coldStartStrategy="drop"`: Excludes incomplete predictions.

3. **Evaluation**:
   - **Root Mean Square Error (RMSE)**:
     - Measures prediction accuracy.
     - Lower RMSE indicates better model performance.
     - Root-mean-square error = 0.7249733074249952

---

## Installation and Usage

### Step 1: Set Up Apache Spark

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/meralkarduz/FilmRecommendation
   cd FilmRecommendation
   ```

2. **Start Docker Containers**:
   ```bash
   docker-compose up -d
   ```
   Access the Spark application UI at [http://localhost:8080](http://localhost:8080).

3. **Prepare Dataset**:
   Place the downloaded dataset in the `archive` folder.
   > [Download the dataset from Kaggle](https://www.kaggle.com/api/v1/datasets/download/grouplens/movielens-20m-dataset)

### Step 2: Set Up Django

1. **Install Python and Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Server**:
   ```bash
   python manage.py runserver
   ```
   Access the application at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Additional Notes

- Ensure JavaScript is enabled in the browser for Plotly visualizations.
- Populate the database with initial data as needed using Django management commands or the admin interface.
- Use logging for debugging and tracking errors during development.

---

## Results

- Personalized recommendations for users.
- Insights supported by interactive visualizations.
- Scalable infrastructure for big data processing and machine learning.

