from docx import Document
import streamlit as st

st.title("Create a DOCX file with Streamlit")

if st.button("Generate DOCX"):
    doc = Document()
    doc.add_paragraph("Hello from Streamlit!")
    doc.save("example.docx")
    st.success("example.docx file created!")
