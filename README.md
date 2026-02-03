# Football Market Value Predictor

An end-to-end Machine Learning pipeline that predicts the market value of football players.
It features a FastAPI backend, a Streamlit dashboard, and SHAP explainability to show why a player is valued at a certain price.

## Features

- **Interactive Dashboard:** specific user interface to input stats and visualize results.
- **Explainable AI:** Uses SHAP values to explain which stats (e.g., Age, Goals) drove the price up or down.
- **Production-Ready API:** Dockerized FastAPI backend with input validation.
- **Automated Tests:** Full test suite ensures API stability.
- **Robust Processing:** Handles missing values and categorical encoding (Position, Foot).

## Tech Stack

- **Core:** Python 3.10, Pandas, Scikit-Learn
- **ML:** XGBoost, SHAP (Explainability)
- **Web:** FastAPI, Streamlit
- **DevOps:** Docker, Pytest, GitHub Actions

## How to Run

### Option 1: The Full Experience (Dashboard + API)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start API (Terminal 1):**
   ```bash
   uvicorn src.main:app --reload
   ```
3. **Start the dashboard (Terminal 2):**
   ```bash
   streamlit run src/app.py
   ```
4. **Access the dashboard at http://localhost:8501**

### Option 2: Using Docker

    ```bash
    docker build -t football-api
    docker run -p 8000:8000 football-api
    ```

### Verify the system is working correctly

    ```bash
    pytest
    ```

### API Usage

**_ POST/predict _**
`json
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
    `

### Project Structure

```bash
├── data/               # Raw datasets
├── models/             # Trained XGBoost model & encoders
├── src/
│   ├── main.py         # FastAPI backend
│   ├── app.py          # Streamlit frontend
│   ├── training.py     # Model training script
│   └── test_api.py     # Automated tests
├── Dockerfile          # Container config
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```
