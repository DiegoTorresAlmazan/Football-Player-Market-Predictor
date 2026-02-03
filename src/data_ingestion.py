import pandas as pd
import os
from loguru import logger

class FootballDataPipeline:
    def __init__(self):
        self.raw_data_path = "data/raw"
        self.processed_data_path = "data/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

    def run_pipeline(self):
        logger.info("Starting data ingestion pipeline...")
        
        try:
            #we ignore market value from players csv to avoid conflicts
            #this is because we want the specific market value from the market value csv
            players = pd.read_csv(f"{self.raw_data_path}/players.csv")
            if 'market_value_in_eur' in players.columns:
                players = players.drop(columns=['market_value_in_eur'])

            #load valuations
            if os.path.exists(f"{self.raw_data_path}/player_valuations.csv"):
                valuations = pd.read_csv(f"{self.raw_data_path}/player_valuations.csv")
            else:
                logger.warning("Player valuations file not found. Checking for alternative filename.")
                valuations = pd.read_csv(f"{self.raw_data_path}/player_market_values.csv")

            #handle potential typo in filename (appearances vs appereances)
            if os.path.exists(f"{self.raw_data_path}/appearances.csv"):
                appearances = pd.read_csv(f"{self.raw_data_path}/appearances.csv")
            elif os.path.exists(f"{self.raw_data_path}/appereances.csv"): 
                appearances = pd.read_csv(f"{self.raw_data_path}/appereances.csv")
            else:
                found_files = os.listdir(self.raw_data_path)
                raise FileNotFoundError(f"Appearances file missing. Your 'data/raw' folder contains: {found_files}")
            
            games = pd.read_csv(f"{self.raw_data_path}/games.csv")

            logger.info(f"Loaded {len(players)} players, {len(valuations)} valuations")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return
        
        #determine latest season
        latest_season = games['season'].max()
        logger.info(f"Latest season identified: {latest_season}")

        #filter games for the latest season
        current_games = games[games['season'] == latest_season]
        valid_game_ids = current_games['game_id'].unique()

        #filter stats for the latest season
        current_stats = appearances[appearances['game_id'].isin(valid_game_ids)]
        logger.info(f"Filtered stats for season {latest_season} (Games: {len(current_games)})")

        #aggregate stats
        logger.info("Aggregating player statistics...")
        player_stats = current_stats.groupby('player_id').agg({
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum',
            'player_id': 'count'
        }).rename(columns={'player_id': 'matches_played'}).reset_index()

        #get latest market value 
        logger.info("Merging player valuations...")
        valuations['date'] = pd.to_datetime(valuations['date'])
        #sort by date and take the latest entry per player
        latest_valuation = valuations.sort_values('date').groupby('player_id').tail(1)

        #merge everything
        logger.info("Merging datasets...")
        merged_df = pd.merge(players, latest_valuation[['player_id', 'market_value_in_eur']], left_on='player_id', right_on='player_id', how='inner')

        #now merge results and stats
        merged_df = pd.merge(merged_df, player_stats, on='player_id', how='left')

        #fill missing stats with zeros
        stats_columns = ['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'matches_played']
        merged_df[stats_columns] = merged_df[stats_columns].fillna(0)

        #feature selection
        final_df = pd.DataFrame()
        final_df['player_id'] = merged_df['player_id']
        final_df['name'] = merged_df['name']

        #calculate age
        merged_df['date_of_birth'] = pd.to_datetime(merged_df['date_of_birth'], errors='coerce')
        final_df['age'] = latest_season - merged_df['date_of_birth'].dt.year

        #physical/profile features
        if 'height_in_cm' in merged_df.columns:
            final_df['height_in_cm'] = merged_df['height_in_cm']
        else:
             final_df['height_in_cm'] = merged_df.get('height_cm', 0)
        final_df['foot'] = merged_df['foot']
        final_df['position'] = merged_df['position']
        final_df['sub_position'] = merged_df['sub_position']

        #performance stats
        final_df['goals'] = merged_df['goals']
        final_df['assists'] = merged_df['assists']
        final_df['minutes_played'] = merged_df['minutes_played']
        final_df["matches_played"] = merged_df['matches_played']

        #target
        final_df['market_value'] = merged_df['market_value_in_eur']

        #drop rows with missing target
        final_df = final_df.dropna(subset=['market_value', 'age'])

        output_file = f"{self.processed_data_path}/training_data.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"Data ingestion pipeline completed. Processed data saved to {output_file}")

if __name__ == "__main__":
    pipeline = FootballDataPipeline()
    pipeline.run_pipeline()