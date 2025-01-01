
# Film Recommendation System

This is a Django-based web application for recommending films. It uses Plotly for interactive plots and integrates with Apache Spark for data processing.

## Features

- Interactive plots using Plotly
- Film recommendations based on user preferences
- Integration with Apache Spark for scalable data processing

## Requirements

- Python 3.8+
- Django 5.1.4
- Plotly 5.24.1
- Apache Spark

---

## Installation and Usage

### Step 1: Docker Compose For Spark

1. **Clone the Repository**
   ```bash
   git clone https://github.com/meralkarduz/FilmRecommendation
   cd film-recommendation-system
   ```

2. **Build and Run Docker Containers**
   Ensure Docker Desktop is running, then execute:
   ```bash
   docker-compose up -d
   ```
   This will build the necessary Docker images and start the services. The spark application ui should be accessible at [http://localhost:8080](http://localhost:8080).

3. **You must download Dataset**
Place the dataset you downloaded using the link into the `archive` folder.
> [Download the dataset from Kaggle](https://www.kaggle.com/api/v1/datasets/download/grouplens/movielens-20m-dataset)


### Step 2: Django Setup

Follow these steps for django:

1. **Install Python and Virtual Environment**
   Ensure Python 3.8+ is installed. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Requirements**
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Development Server**
   Launch the Django development server:
   ```bash
   python manage.py runserver
   ```
   The application will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

### Additional Notes

- **Interactive Plots**: Make sure JavaScript is enabled in your browser to view Plotly plots.
- **Data Initialization**: Populate the database with initial data if required. You can create a management command or use the Django admin interface to add films and user preferences.
