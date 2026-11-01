
import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# Load trained model and dataset
@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as file:
        pipe = pickle.load(file)

    with open("df.pkl", "rb") as file:
        df = pickle.load(file)

    return pipe, df


try:
    pipe, df = load_model()
except Exception as e:
    st.error("Unable to load the ML model.")
    st.exception(e)
    st.stop()


st.title("💻 Laptop Price Predictor")
st.write("Predict laptop prices based on hardware specifications.")

# Brand
company = st.selectbox(
    "Brand",
    sorted(df["Company"].dropna().unique())
)

# Laptop Type
laptop_type = st.selectbox(
    "Type",
    sorted(df["TypeName"].dropna().unique())
)

# RAM
ram = st.selectbox(
    "RAM (in GB)",
    [2, 4, 6, 8, 12, 16, 24, 32, 64]
)

# Weight
weight = st.number_input(
    "Weight (in kg)",
    min_value=0.1,
    max_value=10.0,
    value=1.5,
    step=0.1
)

# Touchscreen
touchscreen = st.selectbox(
    "Touchscreen",
    ["No", "Yes"]
)

# IPS
ips = st.selectbox(
    "IPS Display",
    ["No", "Yes"]
)

# Screen Size
screen_size = st.slider(
    "Screen Size (in inches)",
    10.0, 18.0, 13.0
)

# Resolution
resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080",
        "1366x768",
        "1600x900",
        "3840x2160",
        "3200x1800",
        "2880x1800",
        "2560x1600",
        "2560x1440",
        "2304x1440"
    ]
)

# CPU
cpu = st.selectbox(
    "CPU",
    sorted(df["Cpu brand"].dropna().unique())
)

# HDD
hdd = st.selectbox(
    "HDD (in GB)",
    [0, 128, 256, 512, 1024, 2048]
)

# SSD
ssd = st.selectbox(
    "SSD (in GB)",
    [0, 8, 128, 256, 512, 1024]
)

# GPU
gpu = st.selectbox(
    "GPU",
    sorted(df["Gpu brand"].dropna().unique())
)

# Operating System
operating_system = st.selectbox(
    "Operating System",
    sorted(df["os"].dropna().unique())
)


if st.button("Predict Price", type="primary"):

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])

    ppi = (
        (X_res ** 2 + Y_res ** 2) ** 0.5
    ) / screen_size

    touchscreen_value = 1 if touchscreen == "Yes" else 0
    ips_value = 1 if ips == "Yes" else 0

    query = np.array([
        company,
        laptop_type,
        ram,
        weight,
        touchscreen_value,
        ips_value,
        ppi,
        cpu,
        hdd,
        ssd,
        gpu,
        operating_system
    ], dtype=object)

    query = query.reshape(1, 12)

    try:
        prediction = pipe.predict(query)[0]

        predicted_price = int(np.exp(prediction))

        st.success(
            f"Estimated Laptop Price: ₹{predicted_price:,}"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
