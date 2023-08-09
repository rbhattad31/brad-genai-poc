import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pdf2image import convert_from_path
from pytesseract import image_to_string
import os
import shutil
# Azure Details:

OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")


class Pdferror(Exception):
    pass


# Create Directory
def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        except OSError as e:
            print(f"Error creating directory: {directory_path}")
            print(e)


def convert_pdf_to_img(pdf_file):
    """
    @desc: this function converts a PDF into Image

    @params:
        - pdf_file: the file to be converted

    @returns:
        - an interable containing image format of all the pages of the PDF
    """
    return convert_from_path(pdf_file, 500, poppler_path=r"C:\Program Files (x86)\poppler-0.68.0\bin")


def convert_image_to_text(file):
    """
    @desc: this function extracts text from image

    @params:
        - file: the image file to extract the content

    @returns:
        - the textual content of single image
    """

    text = image_to_string(file)
    return text


def get_text_from_any_pdf(pdf_file):
    """
    @desc: this function is our final system combining the previous functions

    @params:
        - file: the original PDF File

    @returns:
        - the textual content of ALL the pages
    """
    images = convert_pdf_to_img(pdf_file)
    final_text = ""
    for pg, img in enumerate(images):
        final_text += convert_image_to_text(img)
        # print("Page n°{}".format(pg))
        # print(convert_image_to_text(img))

    return final_text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        result = ""
        print(pdf.name)
        tempTuple = os.path.splitext(pdf.name)
        pdf_name = tempTuple[0]
        Temp_Directory = os.path.join('Uploaded Files', pdf_name)
        check_and_create_directory(Temp_Directory)
        with open(os.path.join(Temp_Directory, pdf.name), "wb") as f:
            f.write(pdf.getbuffer())
            pdf_path = os.path.join(Temp_Directory, pdf.name)
        pdf_reader = PdfReader(pdf_path)

        file_size = os.path.getsize(pdf_path)
        size = (file_size / 1024)

        if size > 20 * 1024:
            st.write("File size should be less than 20 MB")

            def remove_directory(directory_path):
                shutil.rmtree(directory_path)
            remove_directory(Temp_Directory)
            raise Pdferror("Uploaded File Size Limit Exceeded")

        for page in pdf_reader.pages:
            result += page.extract_text()

        if result is None or result == "":
            result += get_text_from_any_pdf(pdf_path)
        print(result)
        text += result
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
    embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test", chunk_size=1)
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

    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main_1():

    try:
        st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
        st.write(css, unsafe_allow_html=True)

        # Check the conversation exist in session and intialize it.
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "summary_ans" not in st.session_state:
            st.session_state.summary_ans = None

        st.header("Chat with multiple PDFs :books:")

        if "conversation" in st.session_state:
            user_question = st.chat_input("Say something")
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
                    summary = get_conversation_chain(vectorstore)
                    summary_ans = summary({'question': "Summary of the data"})
                    st.session_state.summary_ans = summary_ans["answer"]
                    st.write("Summary:\n")
                    # st.markdown('<div style="text-align: justify;">' + st.session_state.summary_ans + '</div>',unsafe_allow_html=True)
                    print("File uploaded Successfully")
            if "summary_ans" in st.session_state:
                summ_ans = st.session_state.summary_ans
                if summ_ans is None or summ_ans == "":
                    pass
                else:
                    st.markdown('<div style="text-align: justify;">' + summ_ans + '</div>',
                                unsafe_allow_html=True)

    except Pdferror as pd:
        st.header(pd)
    except Exception as e:
        print(e)
        st.header("Error occurred please try again after sometime!!!")


# if __name__ == '__main__':
#     main_1()
