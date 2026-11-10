import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, web_path: str = None):
        self.web_path = web_path
        self.docs = self.indexing()
        self.chunked = self.splitter()
        self.vectorstore = self.embedding()

    def indexing(self) -> None:
        """
        Index the documents from the web page and store them in self.docs.
        Returns the list of indexed documents.
        """
        loader = WebBaseLoader(
            web_path=self.web_path,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            )
        )
        return loader.load()
    
    def splitter(self) -> None:
        """
        Split the indexed documents into chunks and store them in self.chunked.
        Returns None.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(self.docs)

    def embedding(self) -> None:
        """
        Embed the split documents and store the vectorstore in self.vectorstore.
        Returns None.
        """

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        return Chroma.from_documents(self.chunked, embeddings)