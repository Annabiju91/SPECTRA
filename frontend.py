import streamlit as st
import requests

st.title("AI Physics Tutor")

question = st.text_input("Ask a Physics Question")

if st.button("Ask"):

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "message": question
        }
    )

    data = response.json()

    st.write("### Answer")
    st.write(data["reply"])

    if "source" in data:
        st.write("### Source")
        st.write(data["source"])