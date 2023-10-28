import streamlit as st

from backend_app import upload_pdf


def pdf_component():
    st.header('PDF Reader')
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        upload_pdf(uploaded_file)
        return uploaded_file.name
    else:
        return None

