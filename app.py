import streamlit as st
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import tempfile
import os

st.set_page_config(page_title="Word Doc Chatbot")
st.title("🤖 Chat with a Word Document")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a Word (.docx) file", type=["docx"])
query = st.text_input("Ask something about the document:")

if openai_api_key and uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    document = Document(tmp_file_path)
    full_text = "\n".join([para.text for para in document.paragraphs])

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    docs = vectorstore.similarity_search(query)
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

    st.write("🧠 Answer:", response)
