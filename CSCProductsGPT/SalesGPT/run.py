import argparse

import os
import json
import streamlit as st

from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from salesgpt.tools import get_tools, setup_knowledge_base, add_knowledge_base_products_to_cache

if __name__ == "__main__":

    # import your OpenAI key (put in your .env file)
    with open('.env','r') as f:
        env_file = f.readlines()
    envs_dict = {key.strip("'") :value.strip("\n") for key, value in [(i.split('=')) for i in env_file]}
    print(envs_dict)
    os.environ['OPENAI_API_TYPE'] = "azure"
    os.environ['OPENAI_API_BASE'] = "https://bradsolopenai.openai.azure.com/"
    os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
      
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

    #llm = ChatOpenAI(temperature=0.2)
    llm = AzureChatOpenAI(temperature=0.2, deployment_name="bradsol-openai-test", model_name="gpt-35-turbo")
    
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
                conversation_type="call",
                use_tools=True,
                product_catalog="examples/columbia_sportswear_product_data.txt"
            )
            sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
        else:
            sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path,'r') as f:
            config = json.load(f)
        print(f'Agent config {config}')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)


    #sales_agent.seed_agent()
    st.header('Columbia SportsWear Chatbot')
    # st.title("SalesGPT Chatbot")
    # History is empty then it needs to execute

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.sales_agent = sales_agent
        # init sales agent

        print("Conversation stage" + sales_agent.conversation_stage_id)
        st.session_state.sales_agent.seed_agent()
        add_knowledge_base_products_to_cache(product_catalog = 'examples/columbia_sportswear_product_data.txt')
        print("Init done")

    if human := st.chat_input():
        print("\n")
        #human = 'user' + ": " + human
        #print("Human Input" + human)
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
