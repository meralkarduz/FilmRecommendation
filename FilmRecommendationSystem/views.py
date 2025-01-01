from django.shortcuts import render
from django.http import HttpResponse
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS , ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode, col
import logging
import os
import plotly.express as px
import plotly.io as pio
import pickle

# Set HADOOP_HOME environment variable
os.environ['HADOOP_HOME'] = r'C:\hadoop'

# Create your views here.

def index(request):
    return render(request, 'index.html')

def create_spark_session():
    return SparkSession.builder.appName("SparkProject") \
         .config("spark.executor.memory", "4g") \
         .config("spark.driver.memory", "4g") \
         .config("spark.executor.cores", "4") \
         .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .getOrCreate()
    # .config("spark.master", "spark://spark:7077") \


def load_data(spark, file_path):
    try:
        return spark.read.csv(file_path, header=True, inferSchema=True)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def explore_data(ratings, movies):
    logging.info(f"Number of ratings: {ratings.count()}")
    logging.info(f"Number of movies: {movies.count()}")
    ratings.describe().show()
    movies.describe().show()


def tune_als_model(ratings):
    best_model = None
    best_rmse = float("inf")
    for rank in [10]:
        for regParam in [0.01]:
            als = ALS(maxIter=10, rank=rank, regParam=regParam, userCol="userId", itemCol="movieId", ratingCol="rating",
                      coldStartStrategy="drop")
            model = als.fit(ratings)
            predictions = model.transform(ratings)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            logging.info(f"Rank: {rank}, RegParam: {regParam}, RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
    return best_model

# Global variable to store the model parameters in memory
model_params_in_memory = None
model_path = "file:///Users/meral/filmRecommendation-main/als_model"
def save_model(model, path=None):
    global model_path
    model.write().overwrite().save(model_path)
    logging.info("Model parameters saved in memory")

def load_model():
    global model_path
    if model_path:
        try:
            model = ALSModel.load(model_path)
            return model
        except Exception as e:
            print(f"Hata: {e}")
            return None
        logging.info("Model recreated from memory")
        return model
    else:
        logging.info("No model parameters found in memory")
        return None

def question1(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    if ratings is None:
        return render(request, 'error.html', {'message': "Failed to load ratings data."})
    try:
        rating_distribution = ratings.groupBy("rating").count().orderBy("rating")
        rating_distribution_pd = rating_distribution.toPandas()
        fig = px.bar(rating_distribution_pd, x='rating', y='count', title='Rating Distribution')
        plot_html = pio.to_html(fig, full_html=False)
        spark.stop()
        return render(request, 'plot.html', {'plot_html': plot_html})
    except Exception as e:
        logging.error(f"Error in question1: {e}")
        return render(request, 'error.html', {'message': "An error occurred while processing the data."})

def question2(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    movies = load_data(spark, r"./archive/movie.csv")
    if ratings is None or movies is None:
        return render(request, 'error.html', {'message': "Failed to load data."})
    try:
        top_movies = ratings.groupBy("movieId").count().orderBy(col("count").desc()).limit(10)
        top_movies = top_movies.join(movies, "movieId").select("title", "count")
        top_movies_pd = top_movies.toPandas()
        fig = px.bar(top_movies_pd, x='title', y='count', title='Top 10 Most Rated Movies')
        plot_html = pio.to_html(fig, full_html=False)
        spark.stop()
        return render(request, 'plot.html', {'plot_html': plot_html})
    except Exception as e:
        logging.error(f"Error in question2: {e}")
        return render(request, 'error.html', {'message': "An error occurred while processing the data."})

def question3(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    movies = load_data(spark, r"./archive/movie.csv")
    if ratings is None or movies is None:
        return render(request, 'error.html', {'message': "Failed to load data."})
    try:
        avg_ratings = ratings.groupBy("movieId").avg("rating") \
            .withColumnRenamed("avg(rating)", "rating") \
            .orderBy(col("rating").desc()).limit(10)
        avg_ratings = avg_ratings.join(movies, "movieId").select("title", "rating")
        avg_ratings_pd = avg_ratings.toPandas()
        fig = px.bar(avg_ratings_pd, x='title', y='rating', title='Average Rating per Movie')
        plot_html = pio.to_html(fig, full_html=False)
        spark.stop()
        return render(request, 'plot.html', {'plot_html': plot_html})
    except Exception as e:
        logging.error(f"Error in question3: {e}")
        return render(request, 'error.html', {'message': "An error occurred while processing the data."})

def question4(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    if ratings is None:
        return render(request, 'error.html', {'message': "Failed to load ratings data."})
    try:
        user_ratings = ratings.groupBy("userId").count().orderBy(col("count").desc()).limit(10)
        user_ratings_pd = user_ratings.toPandas()
        fig = px.bar(user_ratings_pd, x='userId', y='count', title='Number of Ratings per User')
        plot_html = pio.to_html(fig, full_html=False)
        spark.stop()
        return render(request, 'plot.html', {'plot_html': plot_html})
    except Exception as e:
        logging.error(f"Error in question4: {e}")
        return render(request, 'error.html', {'message': "An error occurred while processing the data."})

def question5(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    movies = load_data(spark, r"./archive/movie.csv")
    if ratings is None or movies is None:
        return render(request, 'error.html', {'message': "Failed to load data."})
    try:
        highest_rated = ratings.groupBy("movieId") \
            .agg({"rating": "avg", "movieId": "count"}) \
            .orderBy(col("count(movieId)").desc(), col("avg(rating)").desc()).limit(10)
        highest_rated = highest_rated.join(movies, "movieId").select("title", "avg(rating)")
        highest_rated_pd = highest_rated.toPandas()
        fig = px.bar(highest_rated_pd, x='title', y='avg(rating)', title='Top 10 Highest Rated Movies')
        plot_html = pio.to_html(fig, full_html=False)
        spark.stop()
        return render(request, 'plot.html', {'plot_html': plot_html})
    except Exception as e:
        logging.error(f"Error in question5: {e}")
        return render(request, 'error.html', {'message': "An error occurred while processing the data."})

def train_model(request):
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    if ratings is None:
        spark.stop()
        return render(request, 'error.html', {'message': "Failed to load ratings data."})
    ratings = ratings.na.drop()
    try:
        als_model = tune_als_model(ratings)
        save_model(als_model)
        spark.stop()
        return render(request, 'success.html', {'message': "Model trained successfully."})
    except Exception as e:
        logging.error(f"Error during ALS model training: {e}")
        spark.stop()
        return render(request, 'error.html', {'message': "An error occurred during model training."})

def recommend_movies(request):
    user_id = request.GET.get('user_id', '1')
    spark = create_spark_session()
    ratings = load_data(spark, r"./archive/rating.csv")
    movies = load_data(spark, r"./archive/movie.csv")
    if ratings is None or movies is None:
        spark.stop()
        return render(request, 'error.html', {'message': "Failed to load data."})
    ratings = ratings.na.drop()
    als_model = load_model()
    if als_model is None:
        spark.stop()
        return render(request, 'error.html', {'message': "Model not found. Please train the model first."})
    try:
        user_id = int(user_id)
        user_df = ratings.filter(ratings.userId == user_id).select('userId').distinct()
        if user_df.count() == 0:
            spark.stop()
            return render(request, 'error.html', {'message': f"No data found for user_id {user_id}."})
        
        user_recommendation = als_model.recommendForUserSubset(user_df, 10)
        user_recommendation = user_recommendation.withColumn("recommendations", explode("recommendations"))
        user_recommendation = user_recommendation.select("userId", col("recommendations.movieId"), col("recommendations.rating"))
        user_recommendation = user_recommendation.join(movies, "movieId").select("userId", "title", "rating")
        result = user_recommendation.collect()
        spark.stop()
        
        recommendations = [{"title": row.title, "rating": row.rating} for row in result]
        
        return render(request, 'recommendations.html', {'user_id': user_id, 'recommendations': recommendations})
    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        spark.stop()
        return render(request, 'error.html', {'message': "An error occurred while generating recommendations."})

