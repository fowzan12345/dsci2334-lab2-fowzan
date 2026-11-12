# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# #
# def load_model():
#     model_file = 'best_knn_model.joblib'
#     scaler_file = 'scaler.joblib'
#     columns_file = 'feature_columns.joblib'

#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     model_path = os.path.join(current_dir,model_file)
#     scaler_path = os.path.join(current_dir,scaler_file)
#     columns_path= os.path.join(current_dir,columns_file)

#     if all(os.path.exists(p)for p in [model_path,scaler_path,columns_path]):
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         feature_columns = joblib.load(columns_path)
#         st.success("loaded model and preprocessing files: ")
#         return model, scaler, feature_columns
    
#     else:
#         st.error("Required files not found in root directory")
#         st.stop()

# model,scaler,feature_columns = load_model()

# sleep_quality_map = {'poor': 0 , 'Fair': 1, 'Good': 2, 'Excellent':3}
# stress_level_map_revrce = {0:"low", 1: "Medium",2:"High"}

# st.title("Stress Level Predicator")
# st.markdown("Predict your stress level based on health and lifestyle inputs.")

# st.header("Input Parameters")
# age = st.slider("Age",18,100,42)
# gender = st.selectbox("Gender", ['Male','Female','Other'],index=0)
# country = st.selectbox("Country",['Brazil','USA','India','Germany','Other'],index=0)
# occupation =  st.selectbox("Occupation",['Office','Manual','Student','Other'],index=0)
# coffee_intake = st.slider("Coffee Intake (cups/day)", 0.0, 10.0, 5.3)
# caffeine_mg = st.slider("Caffeine intake (mg/day)",0,800,503)
# sleep_hours = st.slider("Sleep Hours",0.0,12.00,5.9)
# sleep_quality = st.selectbox("Sleep Quality",['Poor','Fair','Good','Excellent'],index=1)
# bmi = st.slider("BMI",10.0,50.0,22.7)
# heart_rate = st.slider("Heart Rate (bpm)",40,120,59)
# physical_activity_per_hours = st.slider("Physical Activity Hours / Week",0.0,20.0,11.2)
# smoking = st.radio("Smoking",[0,1],index=0)
# alcohol_consumption = st.radio("Alcohol Consumption",[0,1],index=0)


# def preprocess_input():

#     df_input= pd.DataFrame(0,index=[0],columns=feature_columns)

#     numeric_feature= ['Age','BMI','Caffeine_mg','coffee_intake','physical_activity_per_hours','Heart_Rate','Sleep_Hours']

#     raw_numeric= np.array([[age, bmi,caffeine_mg, coffee_intake , sleep_hours,  heart_rate , physical_activity_per_hours ]])

#     scaled_numeric=scaler.transform(raw_numeric)

#     for i,col in enumerate(numeric_feature):
#         if col in df_input.columns:
#             df_input[0,col]= scaled_numeric[0,i]

#     df_input.at[0,'Sleep_Quality']=sleep_quality_map[sleep_quality]
#     df_input.at[0,'Smoking']=smoking
#     df_input.at[0,'Alcohol_consumption']=alcohol_consumption

#     #one- Hot encoded
#     gender_col = f'Gender_{gender}'
#     if gender_col in df_input.columns:
#         df_input.at[0,gender_col] = 1

#     country_col = f"Country_{country}"
#     if country_col in df_input.columns:
#         df_input.at[0,country_col] = 1

#     occupation_col = f"Occupation_{occupation}"
#     if occupation_col in df_input.columns:
#         df_input.at[0,occupation_col] = 1
    
#     return df_input

# if st.button("Predict Stress Level"):
#     with st.spinner("Processing"):
#         X_user = preprocess_input()

#         try:
#             prediction = model.predict(X_user)
#             probabilities = model.predict_proba(X_user)[0]

#             stress_level = stress_level_map_revrce[prediction]


#             st.success(f"Predicted Stress Level: **{stress_level}")

#             st.header("Predication Confidence")

#         except Exception as e:
#             st.error(f'Predication Error: {e}')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Load artifacts
# -----------------------------
def load_model():
    model_file   = 'best_knn_model.joblib'
    scaler_file  = 'scaler.joblib'
    columns_file = 'feature_columns.joblib'

    current_dir  = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(current_dir, model_file)
    scaler_path  = os.path.join(current_dir, scaler_file)
    columns_path = os.path.join(current_dir, columns_file)

    if all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
        model           = joblib.load(model_path)
        scaler          = joblib.load(scaler_path)
        feature_columns = joblib.load(columns_path)
        st.success("Loaded model and preprocessing files.")
        return model, scaler, feature_columns
    else:
        st.error("Required files not found in the app directory.")
        st.stop()

model, scaler, feature_columns = load_model()

# -----------------------------
# Mappings (normalize cases/spaces)
# -----------------------------
# Your UI options use capitalized labels; keep keys identical.
sleep_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
stress_level_map_reverse = {0: "Low", 1: "Medium", 2: "High"}  # adjust to your classes

# -----------------------------
# UI
# -----------------------------
st.title("Stress Level Predictor")
st.markdown("Predict your stress level based on health and lifestyle inputs.")

st.header("Input Parameters")
age      = st.slider("Age", 18, 100, 42)
gender   = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=0)
country  = st.selectbox("Country", ['Brazil', 'USA', 'India', 'Germany', 'Other'], index=0)
occupation = st.selectbox("Occupation", ['Office', 'Manual', 'Student', 'Other'], index=0)

coffee_intake = st.slider("Coffee Intake (cups/day)", 0.0, 10.0, 5.3)
caffeine_mg   = st.slider("Caffeine intake (mg/day)", 0, 800, 503)
sleep_hours   = st.slider("Sleep Hours (per day)", 0.0, 12.0, 5.9)
sleep_quality = st.selectbox("Sleep Quality", ['Poor','Fair','Good','Excellent'], index=1)

bmi          = st.slider("BMI", 10.0, 50.0, 22.7)
heart_rate   = st.slider("Heart Rate (bpm)", 40, 120, 59)
physical_activity_per_hours = st.slider("Physical Activity Hours / Week", 0.0, 20.0, 11.2)

smoking              = st.radio("Smoking", [0, 1], index=0)
alcohol_consumption  = st.radio("Alcohol Consumption", [0, 1], index=0)

# -----------------------------
# Preprocess
# -----------------------------
def preprocess_input():
    # FIX 1: 'inedx' -> 'index'
    df_input = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)

    # IMPORTANT: These names **must** match the columns used during training.
    # Adjust the list below to exactly the training-time feature names.
    numeric_features = [
        'Age',
        'BMI',
        'Caffeine_mg',
        'coffee_intake',
        'Sleep_Hours',
        'Heart_Rate',
        'physical_activity_per_hours'
    ]

    # Keep the order consistent with numeric_features
    raw_numeric = np.array([[
        age,
        bmi,
        caffeine_mg,
        coffee_intake,
        sleep_hours,
        heart_rate,
        physical_activity_per_hours
    ]])

    # Scale numeric block
    scaled_numeric = scaler.transform(raw_numeric)

    # Write scaled values back (FIX 2: use .at, not df_input[0,col])
    for i, col in enumerate(numeric_features):
        if col in df_input.columns:
            df_input.at[0, col] = float(scaled_numeric[0, i])

    # Ordinal/label features (ensure names match training columns)
    if 'Sleep_Quality' in df_input.columns:
        df_input.at[0, 'Sleep_Quality'] = float(sleep_quality_map[sleep_quality])

    if 'Smoking' in df_input.columns:
        df_input.at[0, 'Smoking'] = float(smoking)

    if 'Alcohol_consumption' in df_input.columns:
        df_input.at[0, 'Alcohol_consumption'] = float(alcohol_consumption)

    # One-hot features (only set 1 where the column exists)
    gender_col = f'Gender_{gender}'
    if gender_col in df_input.columns:
        df_input.at[0, gender_col] = 1.0

    country_col = f'Country_{country}'
    if country_col in df_input.columns:
        df_input.at[0, country_col] = 1.0

    occupation_col = f'Occupation_{occupation}'
    if occupation_col in df_input.columns:
        df_input.at[0, occupation_col] = 1.0

    return df_input

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Stress Level"):
    with st.spinner("Processing..."):
        try:
            X_user = preprocess_input()
            prediction = model.predict(X_user)
            proba = model.predict_proba(X_user)[0] if hasattr(model, "predict_proba") else None

            # FIX 3: prediction is an array; index it
            label_idx = int(prediction[0])
            stress_level = stress_level_map_reverse.get(label_idx, str(label_idx))

            st.success(f"Predicted Stress Level: **{stress_level}**")

            if proba is not None:
                st.header("Prediction Confidence")
                # Display per-class probabilities if available
                for cls_idx, p in enumerate(proba):
                    st.write(f"- Class {cls_idx}: {p:.2%}")

        except Exception as e:
            # FIX 4: typos "Predication" -> "Prediction"
            st.error(f'Prediction Error: {e}')
