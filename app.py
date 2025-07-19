import streamlit as st
import pandas as pd
import pickle
import Feature_extraction_new as feature
from urllib.parse import urlparse
import requests

# Function to query local Gemma model
def lmstudio(url2):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "gemma-3-1b-it",
        "messages": [
            {"role": "user", "content": url2}
        ],
        "temperature": 0,
        "max_tokens": 1,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        output = response.json()
        reply = output['choices'][0]['message']['content'].strip()
        print("[Gemma Prediction]:", reply)
        return reply.lower() in ["legitimate", "safe", "yes", "true", "1"]
    except Exception as e:
        print("Error:", e)
        print("Full response:", response.text)
        return False

# Load the trained model
try:
    with open("model_svm_rbf.pkl", "rb") as f:
        model_svm_rbf = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'model_svm_rbf.pkl' not found.")
    st.stop()

# Set up page configuration
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🔍",
    layout="centered",
)

# Background image logic
bg_image = "Blue.jpg"  # Default

# User Input
st.markdown('<h1 style="text-align:center; color:#ffcc00;">🔍 Phishing Website Detection</h1>', unsafe_allow_html=True)
st.write("Enter a URL below to check if it's **Legitimate** or **Phishing**.")
url = st.text_input("Enter URL (e.g., https://example.com)", "")

if st.button("🔎 Predict", use_container_width=True):
    if not url:
        st.warning("⚠️ Please enter a valid URL.")
    elif urlparse(url).scheme not in ["http", "https"]:
        st.warning("⚠️ Please enter a valid URL with 'http://' or 'https://'")
    else:
        try:
            with st.spinner("🔄 Extracting Features..."):
                y_for_test = feature.get_data_set(url)

            if y_for_test is None or y_for_test.empty:
                st.error("❌ Failed to extract features from the URL.")
                st.stop()

            val = y_for_test.fillna(0)
            pred = model_svm_rbf.predict(val)

            gemma_result = lmstudio(url)

            if pred[0] == 1:
                st.success("✅ Legitimate Website")
                bg_image = "Green.jpg"
            else:
                st.error("🚨 Phishing Website Detected!")
                bg_image = "Red.jpg"

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Inject dynamic CSS for background tiling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({bg_image});
        background-repeat: repeat;
        background-size: 1024px 1024px;
        background-attachment: fixed;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# About Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="font-size:24px; color:#ffcc00;">🛡️ About Phishing</p>', unsafe_allow_html=True)
st.write("""
Phishing is a form of cyber attack where attackers trick users into providing sensitive data, 
such as login credentials and financial information, by masquerading as trustworthy entities.
""")

st.markdown('<p style="font-size:24px; color:#ffcc00;">💡 How Machine Learning Helps</p>', unsafe_allow_html=True)
st.write("""
Machine Learning can detect phishing websites by analyzing various URL features.
By continuously learning from new data, ML models provide **faster** and **more accurate** threat detection.
""")

# Sidebar
st.sidebar.title("ℹ️ About the Model")
st.sidebar.write("**Model:** Support Vector Machine (SVM) with an RBF Kernel.")
st.sidebar.write("**Accuracy:** 94.02%")
st.sidebar.subheader("🔍 Features Analyzed")
st.sidebar.write("""
- URL Length  
- Presence of '@'  
- Subdomains  
- HTTPS Usage  
- Google Indexing  
... and more
""")

st.sidebar.subheader("📂 Dataset")
st.sidebar.write("Trained on a **Phishing URLs Dataset** from Kaggle.")

st.markdown("<br><br>", unsafe_allow_html=True)
