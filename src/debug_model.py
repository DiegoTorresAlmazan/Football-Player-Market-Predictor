import pandas as pd
import joblib
import os

def test_model():
    print("DIAGNOSTIC: Starting Model Check...")
    
    # 1. Check if files exist
    if not os.path.exists("models/price_predictor.pkl"):
        print("ERROR: models/price_predictor.pkl NOT FOUND.")
        return
    if not os.path.exists("models/encoders.pkl"):
        print("ERROR: models/encoders.pkl NOT FOUND.")
        return
        
    print("Files found.")

    # 2. Load Model & Encoders (which are now Dictionaries)
    try:
        model = joblib.load("models/price_predictor.pkl")
        encoders = joblib.load("models/encoders.pkl")
        print("Model and Encoders loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODEL: {e}")
        return

    # 3. Create Dummy Data
    input_data = pd.DataFrame([{
        "goals": 20,
        "assists": 15,
        "minutes_played": 3000,
        "matches_played": 35,
        "age": 23,
        "height_in_cm": 182,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "Right"
    }])

    # 4. Preprocess (Using Dictionary Lookups)
    print("Testing Preprocessing...")
    try:
        cat_cols = ['position', 'sub_position', 'foot']
        for col in cat_cols:
            # Get the mapping dictionary
            mapping = encoders.get(col, {})
            
            # Get value
            val = str(input_data[col].iloc[0])
            
            # Look it up (Default to 0 if unknown)
            if val in mapping:
                input_data[col] = mapping[val]
            else:
                print(f"   Warning: '{val}' not found in encoder for {col}. Using default (0).")
                input_data[col] = 0
                
    except Exception as e:
        print(f"PREPROCESSING CRASHED: {e}")
        return

    # 5. Predict
    print("Attempting Prediction...")
    features = [
        'goals', 'assists', 'minutes_played', 'matches_played', 
        'age', 'height_in_cm', 'position', 'sub_position', 'foot'
    ]
    
    try:
        # Sort columns to match training order
        input_data = input_data[features]
        prediction = model.predict(input_data)[0]
        print("\n" + "="*30)
        print(f"SUCCESS! Prediction: EUR {prediction:,.2f}")
        print("="*30)
        print("The model is working perfectly. You can now start the API.")
    except Exception as e:
        print("\n" + "="*30)
        print(f"PREDICTION FAILED: {e}")
        print("="*30)

if __name__ == "__main__":
    test_model()