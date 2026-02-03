import streamlit as st
import requests
import pandas as pd

# title of the web app
st.title("Football Market Value Predictor")
st.write("Enter player stats below to get an estimated market value.")

# create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    # input fields for numerical values
    goals = st.number_input("Goals", min_value=0, max_value=1000, value=0)
    assists = st.number_input("Assists", min_value=0, max_value=1000, value=0)
    minutes_played = st.number_input("Minutes Played", min_value=0, max_value=80000, value=0)
    matches_played = st.number_input("Matches Played", min_value=0, max_value=1500, value=0)

with col2:
    age = st.number_input("Age", min_value=15, max_value=45, value=24)
    height_in_cm = st.number_input("Height (cm)", min_value=150, max_value=220, value=180)
    
    # dropdowns for text values
    position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
    sub_position_map = {
        "Attack": ["Centre-Forward", "Striker", "Left Winger", "Right Winger", "Second Striker"],
        "Midfield": ["Central Midfield", "Attacking Midfield", "Defensive Midfield", "Left Midfield", "Right Midfield"],
        "Defender": ["Centre-Back", "Fullback", "Left-Back", "Right-Back", "Wing-Back", "Sweeper"],
        "Goalkeeper": ["Goalkeeper"]
    }
    
    # get the list based on what the user picked above
    available_sub_positions = sub_position_map.get(position, [])
    
    # show the filtered dropdown
    sub_position = st.selectbox("Sub-Position", available_sub_positions)
    foot = st.selectbox("Preferred Foot", ["Right", "Left", "Both"])

# button to trigger prediction
if st.button("Predict Value"):
    # prepare the data for the api
    payload = {
        "goals": goals,
        "assists": assists,
        "minutes_played": minutes_played,
        "matches_played": matches_played,
        "age": age,
        "height_in_cm": height_in_cm,
        "position": position,
        "sub_position": sub_position,
        "foot": foot
    }

    try:
        # send request to api
        #response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        #link to deployed api:
        response = requests.post("https://football-player-market-value-predictor.onrender.com/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            value = result['formatted_value']
            st.success(f"Estimated Market Value: **{value}**")

            #for visualizing the explanation
            st.subheader("Why this price?")
            st.write("This chart shows how much each stat influenced the predicted market value.")

            explanation = result["explanation"]

            #conver dictionary to datadame for plotting
            df_exp = pd.DataFrame(list(explanation.items()), columns=['Feature', 'Impact'])
            #sort impact so the biggest factors are on top
            df_exp = df_exp.sort_values(by='Impact', ascending=False)
            #create bar chart
            st.bar_chart(df_exp.set_index('Feature'))
        else:
            st.error(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Is it running?")

st.markdown("---")
with st.expander("Guide to Positions & Stats"):
    st.markdown("""
    ### Defenders
    * **Goalkeeper (GK):** The last line of defense.
    * **Center-Back (CB):** Anchors the defense, blocks shots.
    * **Fullback (RB/LB):** Covers the sides, handles wingers.
    * **Wing-Back:** More attacking defender.

    ### Midfielders
    * **Defensive Midfielder (DM):** Shields the backline.
    * **Central Midfielder (CM):** Box-to-box engine.
    * **Attacking Midfielder (AM):** Creative playmaker.
    * **Wide Midfielder:** Covers the flanks.

    ### Forwards
    * **Striker/CF:** Primary goalscorer.
    * **Winger:** Attacks from the wide areas.
    
    ### Impact on Value
    * **Age:** Younger players have higher potential value.
    * **Goals/Assists:** Attackers usually cost more than defenders.
    * **Minutes Played:** High minutes indicate a reliable, consistent starter.
    """)