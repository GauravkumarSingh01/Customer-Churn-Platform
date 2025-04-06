import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime

# For Gemini API, handle imports carefully
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google Generative AI package is not installed. Please install it using: pip install google-generativeai")

# Page configuration with improved styling
st.set_page_config(
    page_title="Customer Retention Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Customer Retention Predictor\nA tool for predicting and analyzing customer churn."
    }
)

# Apply custom CSS for better UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    .metric-card {
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        background-color: white;
    }
    .risk-high {
        color: #FF4B4B;
        font-weight: bold;
    }
    .risk-med {
        color: #FFA500;
        font-weight: bold;
    }
    .risk-low {
        color: #00CC96;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "file_upload_key" not in st.session_state:
    st.session_state.file_upload_key = 0
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Load the trained XGBoost model with better error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_xgb_model.pkl")
        return model, True
    except Exception as e:
        return str(e), False

model_result = load_model()
if model_result[1]:
    model = model_result[0]
    model_loaded = True
else:
    error_msg = model_result[0]
    st.error(f"Error loading model: {error_msg}")
    model_loaded = False

# Define expected features (same as used during training)
expected_features = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
    'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount',
    'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
    'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
    'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
    'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
    'Gender_Male', 'PreferedOrderCat_Grocery',
    'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile',
    'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

# New helper function to get feature importance from model
@st.cache_data
def get_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        # Create a DataFrame with feature names and their importance scores
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        return feature_importance
    return None

# Helper function to create one-hot encoding
def create_one_hot_encoding(category, selected_value, options_dict):
    result = {}
    for category_option, feature_names in options_dict.items():
        if category == category_option:
            for feature in feature_names:
                feature_value = 1 if feature.endswith(f"_{selected_value}") else 0
                result[feature] = feature_value
    return result

# Enhanced function to generate responses from Gemini API with better error handling
def generate_ai_response(messages, model_name):
    try:
        # Configure the Gemini API
        genai.configure(api_key=st.session_state.api_key)
        
        # Extract system message (the first message with role "system")
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        
        # Convert messages to the format Gemini expects
        conversation = []
        user_messages = [msg for msg in messages if msg["role"] in ["user", "assistant"]]
        
        for i, msg in enumerate(user_messages):
            if msg["role"] == "user" and i == 0 and system_message:
                # For the first user message, prepend the system message
                content = f"[System Instructions: {system_message}]\n\n{msg['content']}"
                conversation.append({"role": "user", "parts": [content]})
            else:
                role = "user" if msg["role"] == "user" else "model"
                conversation.append({"role": role, "parts": [msg["content"]]})
        
        # Select the appropriate Gemini model
        if model_name == "gemini-1.5-pro":
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        else:  # Default to a standard model for gpt-3.5-turbo equivalent
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
        # Start the chat
        chat = gemini_model.start_chat(history=conversation[:-1])
        
        # Get the response (streaming)
        response = chat.send_message(
            conversation[-1]["parts"][0],
            stream=True
        )
        
        # Stream the response
        full_response = ""
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                content = chunk.text
                full_response += content
                yield content
            
    except Exception as e:
        error_message = f"Error: {str(e)}"
        yield error_message

# Function to convert prediction data to CSV for download
def convert_predictions_to_csv():
    if not st.session_state.prediction_history:
        return None
    
    df = pd.DataFrame(st.session_state.prediction_history)
    csv = df.to_csv(index=False)
    return csv

# Function to create batch predictions from uploaded file
def process_batch_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Check for required columns
        required_numerical = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                              'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress']
        required_categorical = ['Gender', 'MaritalStatus', 'PreferredLoginDevice', 
                              'PreferredPaymentMode', 'PreferedOrderCat', 'Complain']
        
        missing_cols = [col for col in required_numerical + required_categorical if col not in df.columns]
        
        if missing_cols:
            st.error(f"The following required columns are missing: {', '.join(missing_cols)}")
            return None
        
        # Create proper input format for batch prediction
        processed_rows = []
        for _, row in df.iterrows():
            # Process numerical features
            numerical_data = {col: row[col] for col in required_numerical}
            
            # Add other numeric columns if they exist
            if 'OrderAmountHikeFromlastYear' in df.columns:
                numerical_data['OrderAmountHikeFromlastYear'] = row['OrderAmountHikeFromlastYear']
            else:
                numerical_data['OrderAmountHikeFromlastYear'] = 0
                
            if 'CouponUsed' in df.columns:
                numerical_data['CouponUsed'] = row['CouponUsed']
            else:
                numerical_data['CouponUsed'] = 0
                
            if 'OrderCount' in df.columns:
                numerical_data['OrderCount'] = row['OrderCount']
            else:
                numerical_data['OrderCount'] = 0
                
            if 'DaySinceLastOrder' in df.columns:
                numerical_data['DaySinceLastOrder'] = row['DaySinceLastOrder']
            else:
                numerical_data['DaySinceLastOrder'] = 0
                
            if 'CashbackAmount' in df.columns:
                numerical_data['CashbackAmount'] = row['CashbackAmount']
            else:
                numerical_data['CashbackAmount'] = 0
            
            # Process Complain as numeric
            numerical_data['Complain'] = 1 if row['Complain'] in [1, '1', 'Yes', 'yes', True, 'true', 'TRUE'] else 0
            
            # Prepare categorical inputs (one-hot encoded)
            categorical_mappings = {
                'PreferredLoginDevice': ['PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone'],
                'PreferredPaymentMode': ['PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
                                        'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                                        'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI'],
                'Gender': ['Gender_Male'],
                'PreferedOrderCat': ['PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
                                    'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
                                    'PreferedOrderCat_Others'],
                'MaritalStatus': ['MaritalStatus_Married', 'MaritalStatus_Single']
            }
            
            # Initialize all categorical features to 0
            categorical_data = {}
            for features in categorical_mappings.values():
                for feature in features:
                    categorical_data[feature] = 0
            
            # Set appropriate features to 1 based on data
            categorical_data['Gender_Male'] = 1 if row['Gender'] in ['Male', 'male', 'M', 'm'] else 0
            
            marital_status = row['MaritalStatus']
            if marital_status in ['Married', 'married']:
                categorical_data['MaritalStatus_Married'] = 1
            elif marital_status in ['Single', 'single']:
                categorical_data['MaritalStatus_Single'] = 1
            
            login_device = row['PreferredLoginDevice']
            if login_device in categorical_mappings['PreferredLoginDevice']:
                for device in categorical_mappings['PreferredLoginDevice']:
                    if device.endswith(f"_{login_device}"):
                        categorical_data[device] = 1
            
            payment_mode = row['PreferredPaymentMode']
            if any(payment in payment_mode for payment in ['COD', 'Cash on Delivery', 'Credit Card', 
                                                         'Debit Card', 'E wallet', 'UPI']):
                for mode in categorical_mappings['PreferredPaymentMode']:
                    if mode.endswith(f"_{payment_mode}"):
                        categorical_data[mode] = 1
            
            order_category = row['PreferedOrderCat']
            if any(category in order_category for category in ['Grocery', 'Laptop & Accessory', 
                                                             'Mobile', 'Mobile Phone', 'Others']):
                for category in categorical_mappings['PreferedOrderCat']:
                    if category.endswith(f"_{order_category}"):
                        categorical_data[category] = 1
            
            # Combine all features
            input_data = {**numerical_data, **categorical_data}
            processed_rows.append(input_data)
        
        # Create DataFrame with proper column order for all rows
        input_df = pd.DataFrame(processed_rows)
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        return input_df, df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Create sidebar for app navigation and settings
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio("Select Mode", ["Customer Prediction", "Batch Prediction", 
                                      "Retention Assistant", "Analytics Dashboard"])
    
    # Theme switcher
    st.markdown("---")
    st.subheader("Settings")
    theme = st.radio("Theme", ["Light", "Dark"], 
                    index=0 if st.session_state.theme == "light" else 1)
    st.session_state.theme = theme.lower()
    
    # Export data option (only shown if there are predictions)
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("Export Data")
        csv_data = convert_predictions_to_csv()
        st.download_button(
            label="Download Prediction History",
            data=csv_data,
            file_name=f"customer_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # About section
    st.markdown("---")
    with st.expander("About this app"):
        st.write("""
        This application helps predict customer churn using machine learning.
        It provides individual predictions, batch processing capabilities,
        and an AI assistant to help with retention strategies.
        
        Built with Streamlit and XGBoost.
        """)

# Main application based on selected mode
if app_mode == "Customer Prediction":
    st.title("Customer Retention Prediction")
    st.write("Enter customer information to predict likelihood of churn")

    # Create a form for better user experience
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Customer Profile Section
        with col1:
            st.subheader("Customer Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            city_tier = st.selectbox("City Tier", [1, 2, 3], 
                                    help="1: Metro cities, 2: Small cities, 3: Rural")
            tenure = st.number_input("Customer Tenure (months)", min_value=0, max_value=100, value=12)
            satisfaction = st.slider("Satisfaction Score", 1, 5, 3,
                                   help="Customer's rating from 1 (very dissatisfied) to 5 (very satisfied)")
            complaint = st.selectbox("Has Filed Complaint?", ["No", "Yes"], 
                                   help="Whether customer has filed a complaint in the past")
            
        # Shopping Behavior Section
        with col2:
            st.subheader("Shopping Behavior")
            login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
            payment_mode = st.selectbox("Preferred Payment Mode", 
                                       ["Credit Card", "Debit Card", "UPI", "E wallet", "COD", "Cash on Delivery"])
            order_category = st.selectbox("Preferred Order Category", 
                                         ["Mobile Phone", "Mobile", "Grocery", "Laptop & Accessory", "Others"])
            hours_on_app = st.number_input("Hours Spent on App (per week)", min_value=0.0, max_value=24.0, value=2.5)
            devices_registered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=2)
            addresses_saved = st.number_input("Number of Addresses Saved", min_value=1, max_value=10, value=2)
        
        # Order Details Section
        with col3:
            st.subheader("Order Details")
            order_count = st.number_input("Total Orders", min_value=1, max_value=1000, value=50)
            days_since_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
            order_amount_hike = st.number_input("Order Amount Increase from Last Year (%)", 
                                              min_value=0, max_value=100, value=10)
            coupons_used = st.number_input("Number of Coupons Used", min_value=0, max_value=100, value=5)
            cashback_amount = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=1000.0, value=50.0)
            warehouse_distance = st.number_input("Distance to Nearest Warehouse (km)", 
                                               min_value=0, max_value=100, value=10)

        # Add customer ID field for easier identification
        customer_id = st.text_input("Customer ID (optional)", 
                                   help="Enter customer identifier for your reference")

        # Submit button
        submitted = st.form_submit_button("Predict Customer Retention")

    # Process form submission
    if submitted and model_loaded:
        # Prepare numerical inputs
        numerical_data = {
            "Tenure": tenure,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_distance,
            "HourSpendOnApp": hours_on_app,
            "NumberOfDeviceRegistered": devices_registered,
            "SatisfactionScore": satisfaction,
            "NumberOfAddress": addresses_saved,
            "Complain": 1 if complaint == "Yes" else 0,
            "OrderAmountHikeFromlastYear": order_amount_hike,
            "CouponUsed": coupons_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": days_since_order,
            "CashbackAmount": cashback_amount
        }
        
        # Prepare categorical inputs (one-hot encoded)
        categorical_mappings = {
            'PreferredLoginDevice': ['PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone'],
            'PreferredPaymentMode': ['PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
                                    'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                                    'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI'],
            'Gender': ['Gender_Male'],
            'PreferedOrderCat': ['PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
                                'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
                                'PreferedOrderCat_Others'],
            'MaritalStatus': ['MaritalStatus_Married', 'MaritalStatus_Single']
        }
        
        # Initialize all categorical features to 0
        categorical_data = {}
        for features in categorical_mappings.values():
            for feature in features:
                categorical_data[feature] = 0
        
        # Set appropriate features to 1 based on selections
        categorical_data['Gender_Male'] = 1 if gender == "Male" else 0
        categorical_data[f'MaritalStatus_{marital_status}'] = 1
        categorical_data[f'PreferredLoginDevice_{login_device}'] = 1
        categorical_data[f'PreferredPaymentMode_{payment_mode}'] = 1
        categorical_data[f'PreferedOrderCat_{order_category}'] = 1
        
        # Combine all features
        input_data = {**numerical_data, **categorical_data}
        
        # Create DataFrame with proper column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Make prediction
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            churn_probability = prediction_proba[0][1] * 100  # Get probability of class 1 (churn)
            
            # Store prediction results in session state for chatbot to access
            st.session_state.last_prediction = {
                "will_churn": bool(prediction[0]),
                "churn_probability": churn_probability,
                "customer_data": input_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "customer_id": customer_id if customer_id else f"CUST-{len(st.session_state.prediction_history) + 1}"
            }
            
            # Add to prediction history
            st.session_state.prediction_history.append(st.session_state.last_prediction)
            
            # Display result with styling
            st.markdown("---")
            
            # Create a three-column layout for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Display churn probability with a gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = churn_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': churn_probability
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction verdict
                if churn_probability > 70:
                    st.error("⚠ **High risk of churn!**")
                elif churn_probability > 30:
                    st.warning("⚠ **Medium risk of churn**")
                # else:
                    # st.success("✅ **Low risk of churn**")
            
            with col2:
                # Show key risk factors based on inputs
                st.markdown("### Key Factors")
                risk_factors = []
                
                if satisfaction <= 2:
                    risk_factors.append("Low satisfaction score")
                if days_since_order > 60:
                    risk_factors.append("Long time since last order")
                if complaint == "Yes":
                    risk_factors.append("Has filed complaint")
                if tenure < 6:
                    risk_factors.append("New customer (low tenure)")
                
                if risk_factors:
                    st.markdown("*Risk factors:*")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                    with col1:
                        st.error("⚠ *Customer likely to churn*")
                else:
                    with col1:
                        st.success("✅ *Customer likely to stay*")
            
            with col3:
                # Display customer profile
                st.subheader("Customer Profile")
                st.markdown(f"**ID:** {customer_id if customer_id else 'Not provided'}")
                st.markdown(f"**Gender:** {gender}")
                st.markdown(f"**Tenure:** {tenure} months")
                st.markdown(f"**Total Orders:** {order_count}")
                st.markdown(f"**Satisfaction:** {satisfaction}/5")
                st.markdown(f"**Last Order:** {days_since_order} days ago")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif app_mode == "Batch Prediction":
    st.title("Batch Customer Retention Prediction")
    st.write("Upload a file containing customer data to predict churn for multiple customers at once.")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Customer Data (CSV or Excel)",
        type=["csv", "xlsx"],
        key=st.session_state.file_upload_key
    )

    if uploaded_file is not None:
        # Process the uploaded file
        processed_data = process_batch_file(uploaded_file)
        if processed_data is not None:
            input_df, original_df = processed_data

            # Make batch predictions
            try:
                predictions = model.predict(input_df)
                prediction_probas = model.predict_proba(input_df)[:, 1] * 100  # Probability of churn

                # Add predictions to the original DataFrame
                original_df["Churn Probability"] = prediction_probas
                original_df["Predicted Churn"] = predictions

                # Display results
                st.subheader("Batch Prediction Results")
                st.dataframe(original_df)

                # Download results
                csv = original_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during batch prediction: {str(e)}")

elif app_mode == "Retention Assistant":
    st.title("Retention Strategy Assistant")
    st.write("Get AI-powered recommendations to improve customer retention.")

    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        st.error("Gemini API is not available. Please install the required package.")
    else:
        # API key input
        if not st.session_state.api_key:
            st.session_state.api_key = st.text_input("Enter your Gemini API Key", type="password")
        
        if st.session_state.api_key:
            st.success("API key loaded successfully!")

            # Chat interface
            st.subheader("Chat with the Retention Assistant")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            if prompt := st.chat_input("Ask a question about retention strategies"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate AI response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in generate_ai_response(st.session_state.messages, "gemini-1.5-pro"):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

elif app_mode == "Analytics Dashboard":
    st.title("Customer Retention Analytics")
    st.write("Explore insights and trends from historical predictions.")

    if not st.session_state.prediction_history:
        st.warning("No prediction history available. Make some predictions first!")
    else:
        # Convert prediction history to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            churn_rate = history_df["will_churn"].mean() * 100
            st.metric("Average Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            avg_prob = history_df["churn_probability"].mean()
            st.metric("Average Churn Probability", f"{avg_prob:.1f}%")

        # Churn probability distribution
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(history_df, x="churn_probability", nbins=20, 
                           labels={"churn_probability": "Churn Probability (%)"},
                           title="Distribution of Churn Probabilities")
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance visualization
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_imp = get_feature_importance(model, expected_features)
            fig = px.bar(feature_imp.head(10), x="Importance", y="Feature", 
                         title="Top 10 Features by Importance", orientation='h')
            st.plotly_chart(fig, use_container_width=True)

        # Time-series analysis (if timestamps are available)
        if "timestamp" in history_df.columns:
            st.subheader("Predictions Over Time")
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df.set_index("timestamp", inplace=True)
            resampled_df = history_df.resample("D").mean().reset_index()
            fig = px.line(resampled_df, x="timestamp", y="churn_probability", 
                          title="Daily Average Churn Probability Over Time")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Gaurav Kumar Singh")