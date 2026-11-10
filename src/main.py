from indexing import Indexer
from retriever import Retriever
from generator import Generator

from config.environment import setup_env

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_PATH = "https://lilianweng.github.io/posts/2023-06-23-agent/"

def main():
    setup_env("src/config/config.json")
    indexer = Indexer(web_path=WEB_PATH)
    retriever = Retriever(vectorstore=indexer.vectorstore)
    generator = Generator()
    docs = retriever.retriever.get_relevant_documents("agent")
    result = generator.generate(docs)
    print(result)

if __name__ == "__main__":
    main()