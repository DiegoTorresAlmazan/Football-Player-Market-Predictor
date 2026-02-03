from fastapi.testclient import TestClient
from src.main import app

#wraps api to send requests without running server
client = TestClient(app)

def test_home_endpoint():
    #check if api is alive
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "football ai is online. go to /docs to test."}

def test_prediction_endpoint():
    #send a sample player data to predict endpoint
    sample_player = {
        "goals": 15,
        "assists": 10,
        "minutes_played": 2500,
        "matches_played": 30,
        "age": 22,
        "height_in_cm": 180,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "Right"
    }
    
    response = client.post("/predict", json=sample_player)

    assert response.status_code == 200 #check if request was successful
    
    #check response structure
    data = response.json()
    assert "predicted_market_value_eur" in data
    assert "formatted_value" in data
    assert "explanation" in data

    #sanity check
    assert data["predicted_market_value_eur"] > 0
