import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model, X_columns = joblib.load('optimal_rf_model_and_columns.pkl')

# Sidebar with enhanced interactivity
st.sidebar.title("📍 **Explore Sections**")
menu = st.sidebar.selectbox(
    "Navigate to", 
    ["🏠 Home", "📈 Prediction"],
    help="Select different sections of the app"
)

# Collapsible section for additional features (Data Insights or extra details)
with st.sidebar.expander("🔍 More Info", expanded=False):
    st.write("""
    - Project Domain: **Real Estate**
    - Key Focus: Data Wrangling, EDA, Model Building, Model Deployment
    - Developed using Python & Machine Learning.
    """)
    st.write("[🔗 GitHub Repo](https://github.com/vishalkannsgit/Singapore-Resale-Flat-Prices-Predicting.git)")

# Home Section
if menu == "🏠 Home":
    st.title("Singapore Resale Flat Price Predictor")
    st.write("""
    ## Project Title: Singapore Resale Flat Prices Predicting
    
    **Skills take away From This Project:**
    - Data Wrangling
    - EDA (Exploratory Data Analysis)
    - Model Building
    - Model Deployment

    **Domain**: Real Estate

    ### Problem Statement:
    The objective of this project is to develop a machine learning model and deploy it
    as a user-friendly web application that predicts the resale prices of flats in
    Singapore. This predictive model will be based on historical data of resale flat
    transactions, aiming to assist both potential buyers and sellers in estimating
    the resale value of a flat.

    ### Motivation:
    The resale flat market in Singapore is highly competitive, with various factors
    affecting resale prices like location, flat type, floor area, and lease duration. 
    A predictive model can help users estimate the resale value by considering 
    these factors.

    ### Scope:
    1. **Data Collection and Preprocessing**: Use historical data of resale flat transactions from 1990 onwards, clean and structure it for machine learning.
    2. **Feature Engineering**: Extract key features like town, flat type, storey range, and more.
    3. **Model Training**: Use machine learning models such as random forests.
    4. **Evaluation**: Measure the model's performance using regression metrics.
    5. **Web Application**: Develop a Streamlit app for users to predict flat prices.
    6. **Deployment**: Deploy the app on a cloud platform like Render.
    7. **Testing and Validation**: Ensure the app performs accurately.

    ### Results:
    This project will assist buyers in making informed decisions by predicting resale prices and help sellers understand their flats' potential market value.

    Navigate to the 'Prediction' section using the sidebar to predict the flat prices.
    """)

# Prediction Section
elif menu == "📈 Prediction":
    st.title("Resale Flat Price Prediction")

    town = st.selectbox("Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
    flat_type = st.selectbox("Flat Type", ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION'])
    storey_range = st.selectbox("Storey Range", ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'])
    floor_area_sqm = st.number_input("Floor Area (sqm)", value=60)
    lease_commence_date = st.number_input("Lease Commence Date", min_value=1960, max_value=2024, step=1, value=1990)

    # Calculate 'flat_age' and 'year'
    flat_age = 2024 - lease_commence_date
    year = 2024

    # Prepare input data
    input_data = pd.DataFrame({
        'town': [town],
        'flat_type': [flat_type],
        'storey_range': [storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'lease_commence_date': [lease_commence_date],
        'flat_age': [flat_age],
        'year': [year]
    })

    # Perform one-hot encoding
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_columns, fill_value=0)

    if st.button("Predict"):
        predicted_price = model.predict(input_data)
        st.write(f"Estimated Resale Price: ${predicted_price[0]:,.2f}")



