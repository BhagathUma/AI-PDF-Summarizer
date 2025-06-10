# This is a Streamlit-based AI assistant that allows users to:
* Upload a PDF
* Summarize the content using a local LLaMA-based language model (like Mistral via Ollama)
* Ask questions related to the PDF and get context-aware answers

Built using LangChain, HuggingFace embeddings, and ChromaDB, it demonstrates how Retrieval-Augmented Generation (RAG) pipelines can be implemented using local models — with no paid APIs.

> [!NOTE]
> This is a very basic project that I used to learn LangChain, Vector embeddings and Retrieval-Augmented Generation

## Steps to run the program

1) Cretae Virtual Environment for the project

2) Install all dependencies
   To install all dependencies run 
   >pip install -r requirements.txt

   in your console

3) Install and Run Ollama
   
   Download Olamma from [Ollama Website](https://ollama.com/download)

   Run these Commands on console to load them

   >ollama pull mistral

   >ollama run mistral

   You can also use llama2 or llama3 instead of mistral


4) Launch the Streamlit App

    >streamlit run app.py


