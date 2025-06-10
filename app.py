import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

if 'page' not in st.session_state:
    st.session_state.page = "welcome"
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embedding' not in st.session_state:
    st.session_state.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if 'llm' not in st.session_state:
    st.session_state.llm = OllamaLLM(model="mistral")


def load_and_embed_pdf(pdf_file):
    pdf_path = f"./temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=st.session_state.embedding,
        persist_directory="./chroma_db"
    )
    

    st.session_state.pdf_path = pdf_path
    st.session_state.chunks = chunks
    st.session_state.vectordb = vectordb

def summarize_pdf():
    text = "\n\n".join([chunk.page_content for chunk in st.session_state.chunks[:5]])
    prompt = f"Summarize the following document:\n\n{text}"
    return st.session_state.llm.invoke(prompt)

def ask_question(question):
    qa = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        retriever=st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
    )
    result = qa.run(question)
    return result

if st.session_state.page == "welcome":
    st.title(" AI PDF Assistant")
    st.write("Summarize or ask questions about any PDF using local LLaMA!")
    if st.button(" Enter"):
        st.session_state.page = "upload"
        st.rerun()

elif st.session_state.page == "upload":
    st.title(" Upload Your PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        load_and_embed_pdf(uploaded_file)
        st.success(" PDF uploaded and processed!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(" Summarize"):
                st.session_state.page = "summary"
                st.rerun()

        with col2:
            question = st.text_input(" Ask a question about the PDF")
            if st.button(" Submit Question") and question.strip():
                st.session_state.question = question
                st.session_state.page = "qa"
                st.rerun()
    else:
        st.info("Please upload a PDF to proceed.")

elif st.session_state.page == "summary":
    st.title(" Summary")
    

    if st.button(" Back"):
        st.session_state.page = "upload"
        st.rerun()
    else:
        with st.spinner("Summarizing..."):
              
            summary = summarize_pdf()
        st.markdown(summary)


elif st.session_state.page == "qa":
    st.title(" Question & Answer")
    


    if st.button(" Back"):
        st.session_state.page = "upload"
        st.rerun()
    else:
        with st.spinner("Thinking..."):
            
            answer = ask_question(st.session_state.question)
        st.markdown(f"**Q:** {st.session_state.question}")
        st.markdown(f"**A:** {answer}")
