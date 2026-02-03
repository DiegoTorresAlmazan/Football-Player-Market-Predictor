# Football Market Value Predictor

This project is an end-to-end Machine Learning pipeline that predicts the market value of football players based on their performance statistics (goals, assists, minutes played, etc.) and demographic data.

It features a complete workflow from data ingestion to model deployment, utilizing **XGBoost** for regression and **FastAPI** + **Docker** for serving predictions in a production-ready environment.

## Features

- **Robust Data Processing:** Handles missing values and encodes categorical features (Position, Foot) using resilient dictionary mapping.
- **High-Performance Model:** Implements an XGBoost Regressor tuned for accuracy.
- **Production-Ready API:** A fast, asynchronous REST API built with FastAPI.
- **Containerized:** Fully Dockerized application ensuring it runs consistently on any machine.
- **Error Handling:** Includes verbose logging and diagnostic scripts for easy debugging.

## Tech Stack

- **Language:** Python 3.10
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Joblib
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **DevOps:** Docker

## How to Run

### Option 1: Using Docker (Recommended)

You do not need to install Python or any libraries manually.

1.  **Build the Docker Image:**

    ```bash
    docker build -t football-api .
    ```

2.  **Run the Container:**

    ```bash
    docker run -p 8000:8000 football-api
    ```

3.  **Access the API:**
    Open your browser to `http://localhost:8000/docs` to see the interactive Swagger UI.

### Option 2: Running Locally

If you want to develop or modify the code:

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/football-market-value.git](https://github.com/DiegoTorresAlmazan/football-market-value.git)
    cd football-market-value
    ```

2.  **Set up the Environment:**

    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Train the Model:**
    This generates the `.pkl` files in the `models/` directory.

    ```bash
    python src/training.py
    ```

4.  **Start the Server:**
    ```bash
    uvicorn src.main:app --reload
    ```

## API Usage

To get a prediction, send a **POST** request to the `/predict` endpoint.

**Example Request Body:**

```json
{
  "goals": 20,
  "assists": 15,
  "minutes_played": 3000,
  "matches_played": 35,
  "age": 23,
  "height_in_cm": 182,
  "position": "Attack",
  "sub_position": "Centre-Forward",
  "foot": "Right"
}
```

**Example Response:**

```json
{
  "predicted_market_value_eur": 52400000.0,
  "formatted_value": "EUR 52,400,000.00"
}
```

**Poject structure:**

```bash
├── data/               # Raw and processed datasets
├── models/             # Serialized XGBoost model and dictionary encoders
├── src/
│   ├── main.py         # FastAPI application entry point
│   ├── training.py     # Script to train and save the model
│   └── debug_model.py  # Diagnostic script to test model independently
├── Dockerfile          # Instructions for building the Docker image
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

```

The current model evaluates Mean Absolute Error (MAE) during training to ensure prediction accuracy.

Algorithm: XGBRegressor

Key Features: Goals, Assists, Age, Minutes Played, Position
