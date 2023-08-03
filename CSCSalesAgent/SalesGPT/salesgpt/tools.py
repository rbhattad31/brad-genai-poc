from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from pydantic import BaseModel, Field, validator



def add_knowledge_base_products_to_cache(product_catalog: str = None):
    """
        We assume that the product catalog is simply a text string.
        """
    # load the document and split it into chunks
    print("Inside Add Knowledge Base")
    loader = TextLoader(product_catalog,encoding='utf8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

def setup_knowledge_base(product_catalog: str = None):
    print("Inside Set Up Knowledge Base")
    """
    We assume that the product catalog is simply a text string.
    """
    # load product catalog
    # with open(product_catalog, "r" , encoding="utf8") as f:
    #     product_catalog = f.read()
    #
    # text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    # texts = text_splitter.split_text(product_catalog)

    #llm = OpenAI(temperature=0)
    #embeddings = OpenAIEmbeddings()
    #embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test",chunk_size = 10)
    # docsearch = Chroma.from_texts(
    #     texts, embeddings, collection_name="product-knowledge-base"
    # )
    llm = AzureOpenAI(temperature=0.2, deployment_name="bradsol-openai-test", model_name="gpt-35-turbo")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )
    return knowledge_base


def get_tools(knowledge_base):
    # we only use one tool for now, but this is highly extensible!
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        )
    ]

    return tools
