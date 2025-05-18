import streamlit as st
from docx import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Title
st.title("🤖 Chat with a Word Document (No OpenAI API Key Needed)")

# Upload Word file
uploaded_file = st.file_uploader("Upload a Word (.docx) file", type=["docx"])

if uploaded_file:
    # Read the Word document text
    doc = Document(uploaded_file)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

    # Show preview
    st.write("Document preview:")
    st.write(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)

    # Split text into chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)

    # Create embeddings with HuggingFace (no API key required)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store from text chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Load a local HuggingFace model for Q&A (you can customize this)
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":256})

    # Setup RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Input question
    query = st.text_input("Ask something about the document:")

    if query:
        with st.spinner("Searching answer..."):
            answer = qa.run(query)
        st.write("**Answer:**")
        st.write(answer)
