import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. setting up the api
app = FastAPI(
    title="Football Market Value Predictor",
    description="predicting player value with xgboost",
    version="1.0.0"
)

# 2. loading stuff (model + encoders)
MODEL_PATH = "models/price_predictor.pkl"
ENCODERS_PATH = "models/encoders.pkl"

# quick check to make sure files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    raise RuntimeError("[ERROR] models missing. run src/training.py first.")

print("------------------------------------------------")
print("[STARTUP] loading model and encoders...")
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("[STARTUP] resources loaded.")
except Exception as e:
    print(f"[STARTUP ERROR] {e}")
    traceback.print_exc()
print("------------------------------------------------")

# 3. defining what a player looks like
class PlayerStats(BaseModel):
    goals: int
    assists: int
    minutes_played: int
    matches_played: int
    age: int
    height_in_cm: int
    position: str
    sub_position: str
    foot: str

# 4. the prediction endpoint
@app.post("/predict")
def predict_value(player: PlayerStats):
    try:
        print(f"[REQUEST] got request for: {player}")
        
        # handling pydantic versions, whatever
        try:
            player_data = player.model_dump() # v2
        except AttributeError:
            player_data = player.dict() # v1 fallback
            
        # making it a dataframe
        input_data = pd.DataFrame([player_data])
        
        # preprocess: mapping text to numbers with our dict
        cat_cols = ['position', 'sub_position', 'foot']
        
        for col in cat_cols:
            # grab the dict for this column
            mapping = encoders.get(col, {})
            
            # get the value as a string
            val = str(input_data[col].iloc[0])
            
            # look it up, default to 0 if unknown
            input_data[col] = mapping.get(val, 0)
        
        # making sure columns match training order
        features = [
            'goals', 'assists', 'minutes_played', 'matches_played', 
            'age', 'height_in_cm', 'position', 'sub_position', 'foot'
        ]
        
        # predicting
        prediction = model.predict(input_data[features])[0]
        
        print(f"[SUCCESS] prediction: EUR {prediction:,.2f}")
        
        return {
            "predicted_market_value_eur": float(prediction),
            "formatted_value": f"EUR {prediction:,.2f}"
        }
    
    except Exception as e:
        # printing error
        print("[CRITICAL ERROR] api crashed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 5. home page
@app.get("/")
def home():
    return {"message": "football ai is online. go to /docs to test."}