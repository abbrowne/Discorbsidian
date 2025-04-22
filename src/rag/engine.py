import os
from typing import List, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

class RAGEngine:
    def __init__(self, persist_directory: str = "./data/vector_db"):
        """Initialize the RAG engine with vector store and conversation chain."""
        load_dotenv()
        
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_documents(self, texts: List[str]):
        """Add documents to the vector store."""
        docs = self.text_splitter.create_documents(texts)
        self.vectorstore.add_documents(docs)
        self.vectorstore.persist()

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        result = self.qa_chain({"question": question})
        return result["answer"]

    def clear(self):
        """Clear the conversation memory."""
        self.memory.clear() 