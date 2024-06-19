import streamlit as st
from streamlit import cache_data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load data
@cache_data
def load_data():
    return pd.read_csv('data_kpi.csv').dropna()

data = load_data()

# Preprocessing
le_objective = LabelEncoder().fit(data['Objective'])
le_platform = LabelEncoder().fit(data['Platform'])
data['Objective'] = le_objective.transform(data['Objective'])
data['Platform'] = le_platform.transform(data['Platform'])

# Sidebar
st.sidebar.title('Select Parameters')
objective = st.sidebar.selectbox('Objective', le_objective.classes_)
platform = st.sidebar.radio('Platform', ['Facebook', 'Instagram'])
monthly_budget = st.sidebar.number_input('Enter Monthly Budget (IDR)', min_value=0)

# Display monthly budget without decimal and with IDR symbol
monthly_budget_str = f"Rp {monthly_budget:,.0f}"
st.sidebar.write(f"Monthly Budget (IDR): {monthly_budget_str}")

# Initialize list to store predictions
prediction_records = []

# File path for saving predictions
predictions_file = 'predictions.csv'

# Check if predictions file exists, if not create it
if not os.path.exists(predictions_file):
    df_empty = pd.DataFrame(columns=['Objective', 'Platform', 'Monthly Budget (IDR)', 'Estimated Impressions', 'Estimated Reach'])
    df_empty.to_csv(predictions_file, index=False)

# Function to save predictions to CSV
def save_to_csv(predictions):
    df = pd.DataFrame(predictions)
    df.to_csv(predictions_file, mode='a', index=False, header=False)

# Filter data based on selected objective and platform
filtered_data = data[(data['Objective'] == le_objective.transform([objective])[0]) & (data['Platform'] == le_platform.transform([platform.lower()])[0])]

# Check if filtered data is empty
if filtered_data.empty:
    st.error('No data available for selected Objective and Platform. Please choose different parameters.')
else:
    # Calculate daily budget
    daily_budget = monthly_budget / 30.0

    # Prepare X and y
    X = filtered_data[['Amount spent (IDR)']]
    y_impressions = filtered_data['Impressions']
    y_reach = filtered_data['Reach']

    # Train/test split
    X_train, X_test, y_impressions_train, y_impressions_test, y_reach_train, y_reach_test = train_test_split(X, y_impressions, y_reach, test_size=0.2, random_state=0)

    # Train models
    model_impressions = RandomForestRegressor(random_state=0).fit(X_train, y_impressions_train)
    model_reach = RandomForestRegressor(random_state=0).fit(X_train, y_reach_train)

    if st.sidebar.button('Submit'):
        # Predictions based on daily budget
        daily_budget_input = np.array([[daily_budget]])
        impressions_prediction = int(round(model_impressions.predict(daily_budget_input)[0] * 30))
        reach_prediction = int(round(model_reach.predict(daily_budget_input)[0] * 30))

        # Record predictions
        prediction_records.append({
            'Objective': objective,
            'Platform': platform,
            'Monthly Budget (IDR)': monthly_budget_str,
            'Estimated Impressions': impressions_prediction,
            'Estimated Reach': reach_prediction
        })

        # Save predictions to CSV
        save_to_csv(prediction_records)

# Display recorded predictions
if os.path.exists(predictions_file):
    st.header('Recorded Predictions')
    st.markdown('Impression and reach estimates are estimates for one month with 30 days')
    df_predictions = pd.read_csv(predictions_file)
    
    # Format columns with commas for better readability
    df_predictions['Estimated Impressions'] = df_predictions['Estimated Impressions'].apply(lambda x: '{:,}'.format(x))
    df_predictions['Estimated Reach'] = df_predictions['Estimated Reach'].apply(lambda x: '{:,}'.format(x))
    
    st.table(df_predictions)
