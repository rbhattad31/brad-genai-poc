import argparse

import os
import json
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import CallbackManager
from loguru import logger

from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from salesgpt.tools import get_tools, setup_knowledge_base, add_knowledge_base_products_to_cache

from salesgpt.callbackhandler import MyCustomHandler

if __name__ == "__main__":

    # import your OpenAI key (put in your .env file)
    # with open('.env','r') as f:
        # env_file = f.readlines()
    # envs_dict = {key.strip("'") :value.strip("\n") for key, value in [(i.split('=')) for i in env_file]}
    #print(envs_dict)
    os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
    if os.getenv("OPENAI_API_TYPE"):
        openai_api_type = os.getenv("OPENAI_API_TYPE")
    else:
        openai_api_type = st.secrets["OPENAI_API_TYPE"]

    # openai_api_base = os.getenv("OPENAI_API_BASE")
    if os.getenv("OPENAI_API_BASE"):
        openai_api_base = os.getenv("OPENAI_API_BASE")
    else:
        openai_api_base = st.secrets["OPENAI_API_BASE"]

    # openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
    else:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    # openai_api_version = os.getenv("OPENAI_API_VERSION")
    if os.getenv("OPENAI_API_VERSION"):
        openai_api_version = os.getenv("OPENAI_API_VERSION")
    else:
        serpapi_api_key = st.secrets["OPENAI_API_VERSION"]
    if os.getenv("SERPAPI_API_KEY"):
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    else:
        openai_api_version = st.secrets["SERPAPI_API_KEY"]    
      
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to agent config file', default='')
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation', default=10)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns


    #handlers
    handler = StdOutCallbackHandler()
    customhandler=MyCustomHandler()

    logfile = "examples/output.log"

    logger.add(logfile, colorize=True, enqueue=True)
    filehandler = FileCallbackHandler(logfile)

    #llm = ChatOpenAI(temperature=0.2)
    llm = AzureChatOpenAI(temperature=0.6, deployment_name="bradsol-openai-test", model_name="gpt-35-turbo",callbacks=[customhandler,filehandler],request_timeout=10,max_retries=3)
    if not os.path.isdir('faiss_index'):
        add_knowledge_base_products_to_cache("sample_product_catalog.txt")

    if config_path=='':
        print('No agent config specified, using a standard config')
        USE_TOOLS=True
        if USE_TOOLS:
            config = dict(
                salesperson_name="Ted Lasso",
                salesperson_role="Business Development Representative",
                company_name="Columbia Sports Wear",
                company_business="The Columbia Sportswear Company is an American company that manufactures and distributes outerwear, sportswear, and footwear, as well as headgear, camping equipment, ski apparel, and outerwear accessories",
                company_values="At Columbia Sportswear Company, we're more than just a leader in the global active lifestyle apparel, footwear, accessories and equipment industry. We connect active people with their passions.",
                conversation_purpose="find out whether they are looking to buy sportswear",
                conversation_history=[],
                conversation_type="chat",
                use_tools=True,
                product_catalog="examples/sample_product_catalog.txt"
            )
            sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
        else:
            sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path,'r') as f:
            config = json.load(f)
        print(f'Agent config {config}')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)


    st.header('Columbia SportsWear Chatbot')
    # History is empty then it needs to execute

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.sales_agent = sales_agent
        # init sales agent
        st.session_state.sales_agent.seed_agent()
        logger.info("Init Done")

    if human := st.chat_input():
        print("\n")
        logger.info("Human "+human)
        st.session_state.chat_history.append(human)
        st.session_state.sales_agent.human_step(human)

    st.session_state.sales_agent.determine_conversation_stage()
    st.session_state.sales_agent.step()
    print("\n")
    # print('='*10)
    # cnt = 0
    # while cnt !=max_num_turns:
    #     cnt+=1
    #     if cnt==max_num_turns:
    #         print('Maximum number of turns reached - ending the conversation.')
    #         break
    #     sales_agent.step()
    #
    #     # end conversation
    #     if '<END_OF_CALL>' in sales_agent.conversation_history[-1]:
    #         print('Sales Agent determined it is time to end the conversation.')
    #         break
    #     human_input = input('Your response: ')
    #     sales_agent.human_step(human_input)
    #     print('='*10)
