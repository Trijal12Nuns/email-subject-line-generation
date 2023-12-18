import streamlit as st
import requests

def main():
    st.title("Email Subject Line Predictor")

    # User input
    email_text = st.text_area("Enter Email Text")

    # Predict button
    if st.button("Predict Subject Line"):
        if email_text:
            # Call FastAPI endpoint for prediction
            response = requests.post("https://your-heroku-app-url/predict", data={"email_text": email_text})
            result = response.json()
            subject_line = result.get("subject_line", "Unable to predict")
            st.success(f"Predicted Subject Line: {subject_line}")
        else:
            st.warning("Please enter email text.")

if __name__ == "__main__":
    main()
