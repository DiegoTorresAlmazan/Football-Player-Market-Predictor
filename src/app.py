import streamlit as st
import requests

# title of the web app
st.title("Football Market Value Predictor")
st.write("Enter player stats below to get an estimated market value.")

# create two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    with col1:
    # input fields for numerical values
        goals = st.number_input("Goals", min_value=0, max_value=1000, value=10)
        assists = st.number_input("Assists", min_value=0, max_value=1000, value=5)
        minutes_played = st.number_input("Minutes Played", min_value=0, max_value=80000, value=2000)
        matches_played = st.number_input("Matches Played", min_value=0, max_value=1500, value=25)

with col2:
    age = st.number_input("Age", min_value=15, max_value=45, value=24)
    height_in_cm = st.number_input("Height (cm)", min_value=150, max_value=220, value=180)
    
    # dropdowns for text values
    position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
    sub_position = st.selectbox("Sub-Position", ["Centre-Forward", "Winger", "Attacking Midfield", "Centre-Back"])
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
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            value = result['formatted_value']
            st.success(f"Estimated Market Value: **{value}**")
        else:
            st.error(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Is it running?")