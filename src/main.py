import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import shap

# 1. setting up the api
app = FastAPI(
    title="Football Market Value Predictor",
    description="predicting player value with xgboost",
    version="1.0.0"
)

# 2. loading stuff (model + encoders)
MODEL_PATH = "models/price_predictor.pkl"
ENCODERS_PATH = "models/encoders.pkl"

# global variable for explainer
explainer = None

print("------------------------------------------------")
print("[STARTUP] loading model and encoders...")

# fail fast if files are missing
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"critical error: {MODEL_PATH} not found.")

try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print(f"[DEBUG] model loaded. type: {type(model)}")

    # --- SAFE SHAP LOADING ---
    try:
        print("[STARTUP] attempting to initialize shap explainer...")
        # try the standard way
        explainer = shap.TreeExplainer(model)
        print("[STARTUP] shap explainer loaded successfully.")
    except Exception as e:
        print(f"[WARNING] shap failed to load: {e}")
        print("[WARNING] app will run without explainability features.")
        explainer = None
    # -------------------------

except Exception as e:
    print(f"[CRITICAL ERROR] failed to load model: {e}")
    traceback.print_exc()
    raise e

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
        # handling pydantic versions
        try:
            player_data = player.model_dump() # v2
        except AttributeError:
            player_data = player.dict() # v1 fallback
            
        # making it a dataframe
        input_data = pd.DataFrame([player_data])
        
        # preprocess: mapping text to numbers
        cat_cols = ['position', 'sub_position', 'foot']
        for col in cat_cols:
            mapping = encoders.get(col, {})
            val = str(input_data[col].iloc[0])
            input_data[col] = mapping.get(val, 0)
        
        # ensuring columns match training order
        features = [
            'goals', 'assists', 'minutes_played', 'matches_played', 
            'age', 'height_in_cm', 'position', 'sub_position', 'foot'
        ]
        X_input = input_data[features]
        
        # predicting
        prediction = model.predict(X_input)[0]
        
        # --- SAFE EXPLANATION GENERATION ---
        explanation = {}
        if explainer:
            try:
                shap_values = explainer.shap_values(X_input)
                for i, feature_name in enumerate(features):
                    explanation[feature_name] = float(shap_values[0][i])
            except Exception as e:
                print(f"[ERROR] failed to generate explanation: {e}")
        else:
            explanation = {"error": "shap explainer not available"}
        # -----------------------------------
        
        print(f"[SUCCESS] prediction: EUR {prediction:,.2f}")
        
        return {
            "predicted_market_value_eur": float(prediction),
            "formatted_value": f"EUR {prediction:,.2f}",
            "explanation": explanation
        }
    
    except Exception as e:
        print("[CRITICAL ERROR] api crashed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 5. home page
@app.get("/")
def home():
    return {"message": "football ai is online. go to /docs to test."}