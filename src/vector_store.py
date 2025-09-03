from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class VectorStoreBuilder:
    def __init__(self, csv_path:str, persist_dir:str="chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_and_save_vector_store(self):

        loader = CSVLoader(
            file_path = self.csv_path,
            encoding = "utf-8",
            metadata = []
        )
        data = loader.load()

        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        docs = splitter.split_documents(data)

        vector_db = Chroma.from_documents(docs, self.embeddings, persist_directory = self.persist_dir)
        vector_db.persist()

    def load_vector_store(self):
        return Chroma(persist_directory = self.persist_dir, embedding_function = self.embeddings)





