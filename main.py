from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


llm = OllamaLLM(model="mistral")  

# response = llm.invoke("Explain the difference between AI and ML in 3 sentences.")
# print(response)


loader = PyPDFLoader("trial.pdf")  
pages = loader.load()



splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          
    chunk_overlap=200        
)

chunks = splitter.split_documents(pages)

text = "\n\n".join([chunk.page_content for chunk in chunks[:5]]) 

# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {i+1}:\n", chunk.page_content, "\n")



embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db" 
)


vectordb = Chroma(
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# results = vectordb.similarity_search("What is the course name and code?", k=2)
# for doc in results:
#     print(doc.page_content)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True  
)

# while True:
#     query = input("\n Ask a question about the PDF (or type 'exit'): ")
#     if query.lower() == "exit":
#         break

#     result = qa(query)
#     print("\n Answer:", result["result"])

# query = "explain the document in 3 sentences"
# result = qa.invoke(query)
# print("\n Answer:", result["result"])



prompt = f"Summarize the following document:\n\n{text}"
summary = llm.invoke(prompt)

print("\n Summary:\n", summary)
