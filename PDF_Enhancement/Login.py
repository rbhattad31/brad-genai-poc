import streamlit as st
import MultiplePDF_Chat
from htmlTemplates import css
import pyautogui
import os

def main():

    if "page" not in st.session_state:
        st.session_state.page = "Login"

    if st.session_state.page == "Login":
        login_page()
    elif st.session_state.page == "Chat":
        pdf_page()


def login_page():
    valid_username = os.getenv("PDF_username")
    valid_password = os.getenv("PDF_password")

    st.header("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == valid_username and password == valid_password:
            st.success("Login successful!")
            st.session_state.page = "Chat"
            pyautogui.hotkey("Enter")

        else:
            st.write("Invalid Credits")


def pdf_page():
    MultiplePDF_Chat.main_1()


def generate_session_id(username):
    # Replace this with your actual session ID generation logic
    return username + "_session"


if __name__ == "__main__":
    main()
