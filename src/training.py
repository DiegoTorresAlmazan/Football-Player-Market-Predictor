import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from loguru import logger
import os

class ModelTrainer:
    def __init__(self):
        self.data_path = "data/processed/training_data.csv"
        self.model_path = "models/price_predictor.pkl"
        self.encoders_path = "models/encoders.pkl"
        os.makedirs("models", exist_ok=True)

    def train(self):
        logger.info("Loading processed data for training...")
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Processed data file not found at {self.data_path}. Run data_ingestion.py first.")
            return
        
        # 1. Preprocessing (Convert Categories to Numbers)
        cat_cols = ['position', 'sub_position', 'foot']
        encoders = {}
        
        logger.info("Encoding categorical features...")
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values
                df[col] = df[col].fillna("Unknown")
                # Fit the encoder
                df[col] = le.fit_transform(df[col].astype(str))
                
                # We save a simple dictionary mapping, not the object!
                # Example: {'Striker': 1, 'Goalie': 0}
                mapping = {str(label): int(idx) for label, idx in zip(le.classes_, range(len(le.classes_)))}
                encoders[col] = mapping
        
        # Save the dictionary of mappings
        joblib.dump(encoders, self.encoders_path)

        # 2. Select Features
        features = [
            'goals', 'assists', 'minutes_played', 'matches_played', 
            'age', 'height_in_cm', 'position', 'sub_position', 'foot'
        ]
        
        # Safety check: only use columns that actually exist
        features = [f for f in features if f in df.columns]
        
        X = df[features]
        y = df['market_value']

        logger.info(f"Training on {len(df)} rows with features: {features}")
        
        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Train XGBoost
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

        # 5. Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        logger.info(f"Model Performance - MAE: €{mae:,.2f}")

        # 6. Save the Model
        joblib.dump(model, self.model_path)
        logger.success(f"Trained model saved to {self.model_path}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()