from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

def load_vectorstore():
    vectorstore = FAISS.load_local("services/index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

retriever = load_vectorstore()
