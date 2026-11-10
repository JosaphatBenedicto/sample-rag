from langchain_community.vectorstores import Chroma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever()

