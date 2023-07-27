import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os



# Azure Details:

if os.getenv("OPENAI_API_TYPE"):
   OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
else:
   OPENAI_API_TYPE = st.secrets["OPENAI_API_TYPE"]
if os.getenv("OPENAI_API_BASE"):
   OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
else:
   OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]
if os.getenv("OPENAI_API_KEY"):
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
   OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
if os.getenv("OPENAI_API_VERSION"):
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
else:
   OPENAI_API_VERSION = st.secrets["OPENAI_API_VERSION"]    



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test")
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(deployment_name="bradsol-openai-test", model_name="gpt-35-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    print(st.session_state.chat_history)
    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Check the conversation exist in session and intialize it.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    if "conversation" in st.session_state:
       user_question = st.chat_input("Say something")
       print(user_question)
       if user_question:
           handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Clear chat history
                if "chat_history" in st.session_state:
                    st.session_state.chat_history = None

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain - st.session_state[Holds the memmory until session ends]
                st.session_state.conversation = get_conversation_chain(vectorstore)

                print("File uploaded Successfully")


if __name__ == '__main__':
    main()
