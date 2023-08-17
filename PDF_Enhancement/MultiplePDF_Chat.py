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
    result = {}
    pdf_names = []
    for i, pdf in enumerate(pdf_docs):
        pdf_names.insert(i, pdf.name)
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
        # print(size)

        if size > 20 * 1024:
            st.write("File size Exceeded Limit")
            print("File Size Huge")
            os.remove(pdf_path)
            os.rmdir(Temp_Directory)
            raise Pdferror("File size Exceeded Limit")
        else:
            print("File Size is less")
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        result[i] = text

        if result[i] is None or result[i] == "":
            result[i] = get_text_from_any_pdf(pdf_path)
    return result, pdf_names


def get_text_chunks(text_dict, pdf_names):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len)
    chunks = {}
    for i in range(len(text_dict)):
        text = text_dict[i]
        with open(str(i)+'.txt', 'w',encoding='utf-8') as file:
            text_to_write = text
            file.write(text_to_write)
        chunks[pdf_names[i]] = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, pdf_names):
    embeddings = OpenAIEmbeddings(deployment="bradsol-embedding-test", chunk_size=1, request_timeout=10)
    llm = AzureChatOpenAI(deployment_name="bradsol-openai-test", model_name="gpt-35-turbo", request_timeout=10)
    for i in range(len(text_chunks)):
        text = text_chunks[pdf_names[i]]
        vectorstore_temp = FAISS.from_texts(texts=text, embedding=embeddings)
        if i > 0:
            print(i)
            vectorstore.merge_from(vectorstore_temp)
        else:
            print(i)
            vectorstore = vectorstore_temp
    print(vectorstore)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(deployment_name="bradsol-openai-test", model_name="gpt-35-turbo", request_timeout=10)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
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
        st.header("Chat with multiple PDFs :books:")

        # Check the conversation exist in session and initialize it.
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "summary_ans" not in st.session_state:
            st.session_state.summary_ans = None

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
                    text_chunks = get_text_chunks(raw_text[0], raw_text[1])

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks, raw_text[1])

                    # create conversation chain - st.session_state[Holds the memmory until session ends]
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    summary = get_conversation_chain(vectorstore)
                    summary_ans = summary({'question': "Summary of the data 2 lines and give 3 sample questions to ask in number format"})
                    st.session_state.summary_ans = summary_ans["answer"]
                    # st.write("Summary:\n")
                    # st.markdown('<div style="text-align: justify;">' + st.session_state.summary_ans + '</div>',unsafe_allow_html=True)
                    print("File uploaded Successfully")
        if "summary_ans" in st.session_state:
            summ_ans = st.session_state.summary_ans
            if summ_ans is None or summ_ans == "":
                pass
            else:
                st.header("Summary:\n")
                # st.markdown('<div style=float:right; width: 30%; padding-left: 20px;style=text-align: justify;>'+ summ_ans +'</div>',unsafe_allow_html=True)
                st.markdown('<div style="text-align: justify;">' + summ_ans + '</div>', unsafe_allow_html=True)

        if "conversation" in st.session_state:
            user_question = st.chat_input("Say something")
            if user_question:
                handle_userinput(user_question)

    except Pdferror as pd:
        st.header(pd)
    except Exception as e:
        print(e)
        st.header("Error occurred please try again after sometime!!!")


if __name__ == '__main__':
    main_1()
