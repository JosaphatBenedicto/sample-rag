import bs4
from langchainhub import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class Generator:
    def __init__(self):
        self.client = Client()

    def prompt(self):
        return self.client.pull("rlm/rag-prompt")

    def llm(self):
        llm = ChatGoogleGenerativeAI(model_name="gemini-2.5-flash-lite", temperature=0)
        return llm

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate(self, docs):
        formatted_docs = self.format_docs(docs)
        prompt = self.prompt()
        llm = self.llm()
        return llm.generate(f"{prompt}\n\n{formatted_docs}")