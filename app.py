import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛍️", layout="centered")

# ---------------------------
# Sidebar Theme Toggle
# ---------------------------
theme = st.sidebar.radio("🎨 Theme", ["🌞 Light Mode", "🌙 Dark Mode"])

# CSS Styles for themes
if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stButton>button {
            background-color: #262730;
            color: white;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #FFFFFF;
            color: black;
        }
        .stButton>button {
            background-color: #f0f2f6;
            color: black;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Load Models & Data
# ---------------------------
try:
    scaler = joblib.load("scaler.joblib")
    centroid_model = joblib.load("centroid_model.joblib")
except:
    scaler, centroid_model = None, None

try:
    df = pd.read_csv("online_retail.csv")
except FileNotFoundError:
    df = pd.DataFrame()

# ---------------------------
# Main Title
# ---------------------------
st.markdown("<h1 style='text-align: center;'>🛍️ Shopper Spectrum</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:gray;'>Customer Segmentation & Product Recommendation System</p>", unsafe_allow_html=True)

# ---------------------------
# Tabs for Two Phases
# ---------------------------
tab1, tab2 = st.tabs(["📊 Customer Segmentation", "🎯 Product Recommendation"])

# ---------------------------
# Phase 1: Customer Segmentation
# ---------------------------
with tab1:
    st.markdown("### 📊 Phase 1: Customer Segmentation (RFM Analysis)")

    recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("🛒 Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("💰 Monetary (total spent)", min_value=0.0, step=10.0)

    if st.button("🔍 Predict Cluster", key="predict_cluster", use_container_width=True):
        if scaler and centroid_model:
            try:
                rfm_input = np.array([[recency, frequency, monetary]])
                scaled_input = scaler.transform(rfm_input)
                cluster = centroid_model.predict(scaled_input)[0]

                cluster_labels = {
                    0: "🌟 High-Value Customer",
                    1: "📅 Regular Customer",
                    2: "🛍️ Occasional Shopper",
                    3: "⚠️ At-Risk Customer"
                }

                st.success(f"Predicted Cluster: **{cluster_labels.get(cluster, cluster)}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("⚠️ Model files not found. Please train and save them.")

# ---------------------------
# Phase 2: Product Recommendation
# ---------------------------
with tab2:
    st.markdown("### 🎯 Phase 2: Product Recommendation")

    if df.empty:
        st.warning("⚠️ Product dataset not found. Please upload the dataset.")
    else:
        df['StockCode'] = df['StockCode'].astype(str)

        product_names = (
            df[['StockCode', 'Description']]
            .drop_duplicates(subset=['StockCode'])
            .set_index('StockCode')['Description']
            .fillna("Unknown Product")
            .to_dict()
        )

        product_list = sorted(product_names.keys(), key=lambda code: str(product_names.get(code, "")))

        product_code = st.selectbox(
            "📦 Select Product",
            options=product_list,
            format_func=lambda code: f"{code} - {product_names.get(code, 'Unknown Product')}"
        )

        if st.button("✨ Recommend Products", key="recommend_products", use_container_width=True):
            try:
                pivot_table = df.pivot_table(index="CustomerID", columns="StockCode", values="Quantity", fill_value=0)
                product_similarity = cosine_similarity(pivot_table.T)
                product_similarity_df = pd.DataFrame(product_similarity, index=pivot_table.columns, columns=pivot_table.columns)

                if product_code in product_similarity_df.index:
                    similar_products = product_similarity_df[product_code].sort_values(ascending=False)[1:6]
                    st.subheader("🔥 Top Recommendations")
                    for code in similar_products.index:
                        st.write(f"- {code}: {product_names.get(code, 'Unknown Product')}")
                else:
                    st.warning("⚠️ Selected product not found in similarity matrix.")
            except Exception as e:
                st.error(f"Recommendation failed: {e}")
