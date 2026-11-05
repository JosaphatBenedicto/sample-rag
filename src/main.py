import bs4
from langchainhub import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import indexing
from src.indexing import Indexer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_PATH = "https://lilianweng.github.io/posts/2023-06-23-agent/"

def main():
    retriever = Indexer(web_path=WEB_PATH).retriever

if __name__ == "__main__":
    main()