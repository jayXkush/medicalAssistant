import streamlit as st
from utils.api import upload_pdfs_api


def render_uploader():
    st.sidebar.header("Upload Medical documents (.PDFs)")
    uploaded_files=st.sidebar.file_uploader("Upload multiple PDFs",type="pdf",accept_multiple_files=True)
    if st.sidebar.button("Upload DB") and uploaded_files:
        with st.spinner("Uploading and processing PDFs..."):
            response=upload_pdfs_api(uploaded_files)
            if response.status_code==200:
                st.sidebar.success("✅ Uploaded successfully! Documents are now available for querying.")
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text
                st.sidebar.error(f"❌ Error: {error_msg}")