# CSI 5810 - Forest Cover Type Prediction Project.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Forest Cover Type Prediction", layout="wide")


@st.cache_resource
def load_model():
    model_path = Path("best_forest_model.pkl")
    if not model_path.exists():
        st.error(
            "Model file 'best_forest_model.pkl' not found. "
            "Please run train_forest_cover.py in this directory first."
        )
        st.stop()
    model = joblib.load(model_path)
    return model


model = load_model()
st.success("✅ Model loaded successfully!")

cover_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

st.title("🌲 Forest Cover Type Prediction App")
st.write(
    "This application uses a trained machine learning model to predict the "
    "forest cover type for a 30m x 30m plot in Roosevelt National Forest "
    "based on cartographic features."
)

tab_single, tab_batch, tab_info = st.tabs(
    ["Single Prediction", "Batch Prediction (CSV)", "Model Info"]
)

#single prediction
with tab_single:
    st.subheader("Single Instance Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.number_input("Elevation (m)", min_value=1500, max_value=4500, value=2500)
        slope = st.number_input("Slope (degrees)", min_value=0, max_value=90, value=10)
        aspect = st.number_input("Aspect (0-360)", min_value=0, max_value=360, value=180)
        hd_hydro = st.number_input("Horizontal_Distance_To_Hydrology", min_value=0, max_value=10000, value=100)

    with col2:
        vd_hydro = st.number_input("Vertical_Distance_To_Hydrology", min_value=-500, max_value=500, value=0)
        hd_road = st.number_input("Horizontal_Distance_To_Roadways", min_value=0, max_value=10000, value=500)
        hd_fire = st.number_input("Horizontal_Distance_To_Fire_Points", min_value=0, max_value=10000, value=1000)
        hill_9 = st.number_input("Hillshade_9am", min_value=0, max_value=255, value=200)

    with col3:
        hill_noon = st.number_input("Hillshade_Noon", min_value=0, max_value=255, value=220)
        hill_3 = st.number_input("Hillshade_3pm", min_value=0, max_value=255, value=200)
        wilderness_area = st.selectbox(
            "Wilderness Area",
            options=["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]
        )
        soil_type = st.number_input("Soil_Type (1-40)", min_value=1, max_value=40, value=10)

    if st.button("Predict Cover Type"):
        input_data = {
            "Elevation": elevation,
            "Aspect": aspect,
            "Slope": slope,
            "Horizontal_Distance_To_Hydrology": hd_hydro,
            "Vertical_Distance_To_Hydrology": vd_hydro,
            "Horizontal_Distance_To_Roadways": hd_road,
            "Hillshade_9am": hill_9,
            "Hillshade_Noon": hill_noon,
            "Hillshade_3pm": hill_3,
            "Horizontal_Distance_To_Fire_Points": hd_fire,
        }

        for i in range(1, 5):
            col = f"Wilderness_Area{i}"
            input_data[col] = 1 if wilderness_area == col else 0

        for i in range(1, 41):
            col = f"Soil_Type{i}"
            input_data[col] = 1 if soil_type == i else 0

        X_input = pd.DataFrame([input_data])

        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            for col in needed:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[needed]

        pred = model.predict(X_input)[0]
        label = cover_map.get(int(pred), str(pred))
        st.success(f"Predicted Cover Type: {label}")

#batch prediction
with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.write(
        "Upload a CSV file containing the same feature columns as the training set "
        "(e.g., Elevation, Aspect, Slope, Hillshade_9am, Wilderness_Area1-4, Soil_Type1-40). "
        "Columns 'Id' and 'Cover_Type' will be ignored if present."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        df_input = pd.read_csv(uploaded)

        for col in ["Id", "Cover_Type"]:
            if col in df_input.columns:
                df_input = df_input.drop(columns=[col])

        st.write("Preview of uploaded data:")
        st.dataframe(df_input.head())

        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            for col in needed:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[needed]

        preds = model.predict(df_input)
        labels = [cover_map.get(int(p), str(p)) for p in preds]

        result_df = df_input.copy()
        result_df["Predicted_Cover_Type"] = labels

        st.write("Prediction Results (first 20 rows):")
        st.dataframe(result_df.head(20))

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_out,
            file_name="forest_cover_predictions.csv",
            mime="text/csv"
        )

#model info
with tab_info:
    st.subheader("Model Information")
    st.write(
        "This app loads the best-performing classical model trained on the Kaggle "
        "Forest Cover Type dataset (train.csv) using 10-fold stratified cross-validation."
    )
    st.write(
        "The training script evaluates k-Nearest Neighbors, Logistic Regression, "
        "Naive Bayes, and Linear Discriminant Analysis, selects the best based on "
        "mean validation accuracy, and saves the full preprocessing + model "
        "pipeline as 'best_forest_model.pkl'."
    )
    st.write(
        "Use the Single Prediction tab to test individual feature combinations, "
        "or the Batch Prediction tab to evaluate multiple records from a CSV file."
    )
