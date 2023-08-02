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

def add_knowledge_base_products_to_cache(product_catalog: str = None):
    """
        We assume that the product catalog is simply a text string.
        """
    # load the document and split it into chunks
    print("Inside Add Knowledge Base")
    loader = TextLoader(product_catalog,encoding='utf8')
    documents = loader.load()
    docs = [
        Document(
            page_content="Columbia Men Grey Crestwood Waterproof (Water Resistant)",
            metadata={"Category": "Men", "Sub-Category": "Footwear", "Price": "8999", "Rating": "0",
                      "Sizes Available": "UK-6, UK-7, UK-8, UK-9",
                      "Colours": "Kettle, Black,Black, Columbia Grey,Oatmeal, Beach,,",
                      "Description": "The Columbia Crestwood Waterproof trail shoe is lightweight yet durable and provides comfortable waterproof protection and excellent support while you're out on the trail.",
                      "Features": "Omni-Grip non-marking traction rubber Techlite lightweight midsole for long lasting comfort superior cushioning and high energy return Combination upper featuring leather mesh and webbing Omni-Tech waterproof breathable construction."},

        ),
        Document(
            page_content="Columbia Men Brown Plateau Venture",
            metadata={"Category": "Men", "Sub-Category": "No Product Subcategory for this product", "Price": "7999",
                      "Rating": "0", "Sizes Available": "UK-6, UK-7, UK-8, UK-9, UK-10, UK-11, UK-12", "Colours": ",",
                      "Description": "HIKE LIGHT\n\nThis lightweight hiker is quick on the trails with advanced repellency, a comfy midsole, and a non-marking, high-traction outsole.\n\n\nEXTRA PROTECTION\n\nSuede overlays give you a modern look and additional protection.\n\n\nOmni-SHIELD water and stain resistant treatment\n\nSuede textile and overlays for outdoor protection and a modern hike expression\n\nBreathable textile collar and tongue for comfort\n\nTechlite lightweight midsole for long lasting comfort, superior cushioning, and high energy return\n\nOmni-Grip non-marking traction rubber\n\nUses: Trail\n\nImported",
                      "Features": "Omni-Grip non-marking traction rubber Techlite lightweight midsole for long lasting comfort superior cushioning and high energy return Omni-SHIELD water and stain resistant treatment Suede textile and overlays for outdoor protection and a moder"},

        )
        ]
    #text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    #docs = text_splitter.split_documents(documents)
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
    metadata_field_info = [
        AttributeInfo(
            name="Category",
            description="The category or gender of the product",
            type="string",
        ),
        AttributeInfo(
            name="Sub-Category",
            description="Type of product",
            type="string",
        ),
        AttributeInfo(
            name="Price",
            description="Price of product",
            type="float",
        ),
        AttributeInfo(
            name="Rating", description="A 0-5 ratings for the product", type="integer"
        ),
        AttributeInfo(
            name="Sizes Available",
            description="Sizes in which product is available",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="Colours",
            description="Colours in which product is available",
            type="string",
        ),
        AttributeInfo(
            name="Description",
            description="Detailed description of product",
            type="string",
        )
    ]
    document_content_description = "Brief summary of Columbia Sportswear Products"
    llm = AzureOpenAI(temperature=0.2, deployment_name="bradsol-openai-test", model_name="gpt-35-turbo")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = SelfQueryRetriever.from_llm(
        llm, db, document_content_description, metadata_field_info, verbose=False
    )
    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
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
