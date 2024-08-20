import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFLoader("uploaded_file.pdf")  # Data Ingestion from saved PDF
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# Set custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Josefin Sans', sans-serif;
    }
    .stButton>button {
        background-color: #ff5733;
        color: white;
        border-radius: 12px;
        padding: 0.8em 1.5em;
        font-size: 1.5em;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #ff704d;
        transform: scale(1.05);
    }
    .stTextInput>div>input {
        border: 3px solid #0073e6;
        border-radius: 12px;
        padding: 0.8em;
        font-size: 1.2em;
    }
    .stExpander {
        background-color: #e6f2ff;
        border-radius: 12px;
        padding: 1em;
    }
    .header-title {
        text-align: center;
        font-size: 3em;
        color: #003366;
        background-color: white;
        padding: 0.5em;
        border-radius: 12px;
        margin-bottom: 1em;
    }
    .navbar {
        background-color: #ff5733;
        padding: 1em;
        text-align: center;
        font-size: 1.8em;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        margin-bottom: 1em;
    }
    
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'>NVIDIA NIM Demo</div>", unsafe_allow_html=True)

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# LLM model initialization
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Input field for questions
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to start document embedding
if st.button("Analyze PDF"):
    if uploaded_file is not None:
        with st.spinner("Analyzing document..."):
            vector_embedding(uploaded_file)
        st.success("Vector Store DB Is Ready")
    else:
        st.warning("Please upload a PDF file.")

if prompt1 and "vectors" in st.session_state:
    # Custom spinner for the retrieval process
    st.markdown("<div class='spinner'></div>", unsafe_allow_html=True)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    process_time = time.process_time() - start
    st.write(f"Response time: {process_time:.2f} seconds")
    st.write(response['answer'])

    # Expander for document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.write("--------------------------------")

# Optional: Add a footer or more styling as needed
st.markdown("<footer style='text-align:center; font-size: 1.2em;'>Powered by NVIDIA NIM and Streamlit</footer>", unsafe_allow_html=True)


# import streamlit as st
# import os
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=NVIDIAEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
#         print("hEllo")
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

# # Inject custom CSS
# st.markdown("""
#     <style>
#     .custom-title {
#         border: 2px solid pink;
#         background-color: blackfade;
#         color: white;
#         padding: 30px;
#         text-align: center;
#         transition: all 0.3s ease-in-out;
#         margin-bottom: 20px;
#     }
#     .custom-title:hover {
#         transform: scale(1.1);
#         color: red;
#         border-color: red;
#         background-color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<h1 class="custom-title">Nvidia NIM Demo</h1>', unsafe_allow_html=True)

# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
