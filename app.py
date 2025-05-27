
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Dự đoán Tác nhân và Gợi ý Kháng sinh", layout="centered")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def load_data():
    df = pd.read_csv("Mô hình AI.csv")
    df["Tuoi"] = pd.to_numeric(df["Tuoi"], errors="coerce")
    df = df.dropna(subset=["Tuoi"])
    def encode_abx(val):
        if val == "R": return 1.0
        if val == "S": return 0.5
        return 0.0
    cols = [col for col in df.columns if col not in ["ID", "Tuoi", "Tac nhan", "So ngay dieu tri"]]
    df[cols] = df[cols].applymap(encode_abx)
    return df

@st.cache_resource
def train_model():
    df = load_data()
    df = df.drop(columns=["ID", "So ngay dieu tri"], errors="ignore").dropna()
    le = LabelEncoder()
    df["Tac nhan"] = le.fit_transform(df["Tac nhan"])
    X = df.drop(columns=["Tac nhan"])
    y = df["Tac nhan"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), le

abx_df = pd.read_csv("Mô hình KSD.csv")

def suggest_antibiotics(pathogen):
    if pathogen == "RSV":
        return []
    elif pathogen == "M. pneumonia":
        return ["Clarithromycin", "Azithromycin", "Levofloxacin", "Doxycycline"]
    else:
        row = abx_df[abx_df["Tac nhan"] == pathogen]
        if row.empty:
            return []
        row_vals = row.drop(columns=["Tac nhan"]).T
        return row_vals[row_vals[row.columns[0]] >= 0.5].index.tolist()

model, feature_cols, label_encoder = train_model()

st.header("📋 Nhập dữ liệu lâm sàng")
user_input = {}
for col in feature_cols:
    if col.lower() in ["tuoi", "sp02", "mạch", "nhiệt độ", "crp", "bạch cầu", "nhịp thở", "bệnh ngày thứ"]:
        user_input[col] = st.number_input(col, min_value=0.0, format="%.2f")
    else:
        user_input[col] = st.number_input(col, value=0.0)

if st.button("🔍 Dự đoán"):
    df_input = pd.DataFrame([user_input])[feature_cols]
    y_pred = model.predict(df_input)[0]
    label = label_encoder.inverse_transform([y_pred])[0]
    st.success(f"✅ Tác nhân gây bệnh được dự đoán: **{label}**")

    st.subheader("💊 Kháng sinh gợi ý:")
    abx_list = suggest_antibiotics(label)
    if abx_list:
        for abx in abx_list:
            st.markdown(f"- {abx}")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")
