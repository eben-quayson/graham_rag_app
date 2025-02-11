import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv


load_dotenv()

# Set up environment variable for Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embeddings)

# Load your documents
file_path = "./data/paul_graham_essay.txt"
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Add documents to the vector store
vector_store.add_documents(texts)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# FastAPI application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/process_query")
async def process_query(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"answer": response}




import streamlit as st
import requests

# Initialize session state for queries
if 'queries' not in st.session_state:
    st.session_state.queries = []

st.title(":rainbow[GrahamPedia]")
st.write("What do you want to know about Graham's essays:")

# Function to add a new query input box
def add_query():
    st.session_state.queries.append({'query': '', 'response': ''})

# Display existing queries and their responses
for i, q in enumerate(st.session_state.queries):
    query_key = f'query_{i}'
    response_key = f'response_{i}'

    # Display the query input box
    st.session_state.queries[i]['query'] = st.text_input(
        f"Ask Graham:", value=q['query'], key=query_key)

    # If the query is not empty and response is empty, process the query
    if st.session_state.queries[i]['query'] and not st.session_state.queries[i]['response']:
        with st.spinner("Processing..."):
            response = requests.post(
                "http://localhost:8000/process_query",
                json={"query": st.session_state.queries[i]['query']}
            )
            if response.status_code == 200:
                st.session_state.queries[i]['response'] = response.json().get("answer")
            else:
                st.session_state.queries[i]['response'] = "Error processing the query."

    # Display the response
    if st.session_state.queries[i]['response']:
        st.write("Response:")
        st.write(st.session_state.queries[i]['response'])

# Button to add a new query input box
if st.button("Ask New Question"):
    add_query()
