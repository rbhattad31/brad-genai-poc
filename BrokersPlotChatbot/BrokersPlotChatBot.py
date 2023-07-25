import streamlit as st
import openai
import json
import os

from azure.cosmos import CosmosClient


def azure_openai_get_response(new_message):
    if os.getenv("OPENAI_API_TYPE"):
       openai_api_type = os.getenv("OPENAI_API_TYPE")
    else:
       openai_api_type = st.secrets["OPENAI_API_TYPE"]
    if os.getenv("OPENAI_API_BASE"):
       openai_api_base = os.getenv("OPENAI_API_BASE")
    else:
       openai_api_type = st.secrets["OPENAI_API_BASE"]
    if os.getenv("OPENAI_API_KEY"):
       openai_api_key = os.getenv("OPENAI_API_KEY")
    else:
       openai_api_type = st.secrets["OPENAI_API_KEY"]
    if os.getenv("OPENAI_API_VERSION"):
        openai_api_version = os.getenv("OPENAI_API_VERSION")
    else:
       openai_api_type = st.secrets["OPENAI_API_VERSION"]    
    
    
   

    response = openai.ChatCompletion.create(
        engine="bradsol-openai-test",
        # messages=[{"role": "system", "content": "You are an AI assistant that helps people find information."},
        #      {"role": "user", "content": "hi how are you"}, {"role": "assistant",
        #                                                      "content": "As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?"}],
        messages=new_message,
        temperature=0.1,

        api_type=openai_api_type,
        api_version=openai_api_version,
        api_key=openai_api_key,
        api_base=openai_api_base,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    return response


def first_prompt_execution(user_input, message_array):
    common_template = " : give response in only JSON Data, intent for the input in json format . if user intent is related to property then intent key in json needs to contain property and action the respective action and json format should be with all the given parameters even the values are not found. don't skip the parameters ond don't add other parameters to JSON -  json parameters are fixed and they are - "
    all_json_fields = ''' "\"intent\":\"\",\"action\":\"\", \"permit_number\": \"\", \"reference_number\": \"\", \"offering_type\": \"\", \"property_type\": \"\", \"price_on_application\": \"\", \"price\": , \"city\": \"\", \"community\": \"\", \"sub_community\": \"\", \"property_name\": \"\", \"title_en\": \"\", \"description_en\": \"completion_status\": \"\", \"amenities\": \"\", \"size\": \"\", \"bedroom\": \"\", \"bathroom\": \"\", \"agent_name\": \"\", \"agent_email\": \"\", \"agent_phone\": \"\", \"license_no\": \"\", \"parking\": \"\", \"geopoints\": \"\" , "Agent_Name": "","Agent_Contact": "", "Agent_Email": "", "user_message"'''
    mandatory_fields = "'property_name', 'community', 'property_type'"
    mandatory_fields_message = "in the mentioned Json parameters, following are the mandatory fields," + mandatory_fields + " if any of the mandatory fields are not identified, update the 'user_message' json parameter with the phrase that should contain the message to ask user to input the fields that are not having values of the given mandatory fields list in JSON, don't consider fields other than mandatory fields for this comparison, please be sure while updating the 'user_message'. if all mandatory fields are filled, then update this user_message with 'processing' "

    str_first_run_template = common_template + all_json_fields + mandatory_fields_message
    user_message = ''  # initializing user_message variable

    role = "user"
    content = user_input + str_first_run_template
    dict_user_role_content = {"role": role, "content": content}
    temp_message_array = message_array.copy()
    temp_message_array.append(dict_user_role_content)

    str_azure_openai_response = azure_openai_get_response(new_message=temp_message_array)

    return [str_azure_openai_response, dict_user_role_content]


def get_data_from_first_response(str_azure_openai_response):
    # extract Role and content from openai response object
    role_of_the_response = str_azure_openai_response["choices"][0]["message"]["role"]
    content_of_the_response = str_azure_openai_response["choices"][0]["message"]["content"]
    # # print(role_of_the_response)
    # print("******************************")
    # print("response of the Azure OpenAI:")
    # print(content_of_the_response)
    # print("******************************")

    # extract only json from the response
    str_response_content = str(content_of_the_response)
    json_open_index = str_response_content.find('{')
    json_close_index = str_response_content.rfind('}') + 1
    json_from_the_content = str_response_content[json_open_index:json_close_index]
    # print("******************************")
    # print("JSON extracted from the Response:")
    # print(json_from_the_content)
    # print("******************************")
    if len(json_from_the_content) == 0:
        # print("I’m so sorry, I’m not sure how to help with that. Can you try rephrasing? \n")
        intent = "not found"
        user_message = "I’m so sorry, I’m not sure how to help with that. Can you try rephrasing?"
        return [intent, user_message, role_of_the_response, content_of_the_response, json_from_the_content]
    else:
        parsed_json_content = json.loads(json_from_the_content)
        user_message = str(parsed_json_content["user_message"]).lower().replace('_', ' ')
        intent = str(parsed_json_content["intent"]).lower()
        return [intent, user_message, role_of_the_response, content_of_the_response, json_from_the_content]


def build_db_query_from_json(input_json):
    # Construct the SQL query dynamically with partial field matching
    query = 'SELECT * FROM c WHERE '
    # Iterate through input fields and check for partial matching with Cosmos DB fields
    for key, value in input_json.items():
        if value != "":
            if key == "intent" or key == "action" or key == "user_message" or key == "amenities":
                pass
            # elif key == "amenities":
            #     query += f"LOWER(c.{key}) like LOWER('%{value}%') AND "
            else:
                query += f"LOWER(c.{key}) = LOWER('{value}') AND "
    # To Remove Last AND after successful query generated
    query = query[:-5]
    return query


def get_cosmos_db_data(db_query):
    cosmos_db_endpoint = os.getenv("COSMOS_DB_ENDPOINT")
    cosmos_db_key = os.getenv("COSMOS_DB_KEY")
    cosmos_db_name = os.getenv("COSMOS_DB_NAME")
    cosmos_db_container_name = os.getenv("COSMOS_DB_CONTAINER_NAME")

    endpoint = cosmos_db_endpoint
    key = cosmos_db_key
    database_name = cosmos_db_name
    container_name = cosmos_db_container_name
    # Initialize the Cosmos DB client
    client = CosmosClient(endpoint, key)

    # Get a reference to the desired database and container
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)
    items_json_in_array = list(container.query_items(db_query, enable_cross_partition_query=True))
    return items_json_in_array


def display_output_in_table(json_array_from_cosmos):
    fields_to_display = ['property_name', 'community', 'sub_community', 'property_type', 'title_en', 'Agent_Name',
                         'Agent_Contact', 'Agent_Email']
    selected_fields = [{field: item[field] for field in fields_to_display} for item in json_array_from_cosmos]
    st.table(selected_fields)


def main():
    st.header('Responsive Chatbot with Cosmos DB Integration')
    st.title("BrokersPlot Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello, How can I help you?"}]
    if "message_array" not in st.session_state:
        st.session_state["message_array"] = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        last_message = st.session_state.messages[-1]
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # first prompt or retry prompt processing
        if last_message["content"] == "Hello, How can I help you?" or last_message["content"] == "I’m so sorry, I’m not sure how to help with that. Can you try rephrasing?":
            st.session_state.message_array = [{"role": "system",
                                               "content": "You are an AI assistant that helps people find information and formatting text to JSON"},
                                              {"role": "user", "content": "hi how are you"},
                                              {"role": "assistant",
                                               "content": "As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?"}
                                              ]

            first_prompt_response = first_prompt_execution(prompt, st.session_state.message_array)

            azure_openai_response = first_prompt_response[0]
            # print(azure_openai_response)
            dict_user_role_content = first_prompt_response[1]
            # print(dict_user_role_content)

            extracted_data = get_data_from_first_response(azure_openai_response)
            intent = extracted_data[0]
            # print(intent)
            str_to_be_user_message = extracted_data[1]
            # print(str_to_be_user_message)

            if "property" in intent:
                # appending relevant user prompt to message array
                # as the prompt is related to property only, appending the user message to the messages list to
                # have continuous chat with openai.
                st.session_state.message_array.append(dict_user_role_content)
                role_of_the_response = extracted_data[2]
                content_of_the_response = extracted_data[3]
                json_from_the_content = extracted_data[4]
                parsed_json_content = json.loads(json_from_the_content)
                # create a new dictionary to append response from the openai
                response_message = {"role": role_of_the_response, "content": content_of_the_response}

                # append response message to messages array
                st.session_state.message_array.append(response_message)

                # if all the values of important json fields are filled, then this condition would be true
                if str_to_be_user_message == "processing":
                    db_query = build_db_query_from_json(parsed_json_content)
                    # print("DB Query is : ", db_query)

                    # Connect to Cosmos DB and get the response from it
                    json_array_from_cosmos = get_cosmos_db_data(db_query)
                    # print("Data from Cosmos DB", json_array_from_cosmos)

                    cosmos_db_output_length = len(json_array_from_cosmos)
                    print(cosmos_db_output_length)
                    # print("Data from Cosmos DB", json_array_from_cosmos)
                    if cosmos_db_output_length != 0:
                        display_output_in_table(json_array_from_cosmos)
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": "Apologies!! Couldn't find data with the given details"})
                        st.chat_message("user").write("Apologies!! Couldn't find data with the given details")


                else:
                    st.session_state.messages.append({"role": "assistant", "content": str_to_be_user_message})
                    st.chat_message("assistant").write(str_to_be_user_message)

            else:
                # print("first run - irrelevant prompt given by user, asking user again to give the correct prompt ")
                st.session_state.messages.append({"role": "assistant", "content": str_to_be_user_message})
                st.chat_message("assistant").write(str_to_be_user_message)
        else:
            user_new_prompt = "user input is: " + prompt + ", please update the Json as per the new user prompt"
            role = "user"
            new_user_message = {"role": role, "content": user_new_prompt}
            new_temp_message_array = st.session_state.message_array.copy()
            new_temp_message_array.append(new_user_message)
            str_azure_openai_response_new = azure_openai_get_response(new_message=new_temp_message_array)
            role_of_the_response = str_azure_openai_response_new["choices"][0]["message"]["role"]
            content_of_the_response = str_azure_openai_response_new["choices"][0]["message"]["content"]
            # print(content_of_the_response)
            new_response_message = {"role": role_of_the_response, "content": content_of_the_response}

            # extract only json from the response
            str_response_content = str(content_of_the_response)
            json_open_index = str_response_content.find('{')
            json_close_index = str_response_content.rfind('}') + 1
            json_from_the_content = str_response_content[json_open_index:json_close_index]

            parsed_json_content = json.loads(json_from_the_content)
            user_message = str(parsed_json_content["user_message"]).lower().replace('_', ' ')
            # print(user_message)

            if user_message != "processing":
                st.session_state.message_array.append(new_user_message)
                st.session_state.message_array.append(new_response_message)
                st.session_state.messages.append({"role": "assistant", "content": user_message})
                st.chat_message("assistant").write(user_message)
            else:
                print( st.session_state.message_array)

                db_query = build_db_query_from_json(parsed_json_content)
                # print("DB Query is : ", db_query)

                # Connect to Cosmos DB and get the response from it
                json_array_from_cosmos = get_cosmos_db_data(db_query)

                cosmos_db_output_length = len(json_array_from_cosmos)
                print(cosmos_db_output_length)
                # print("Data from Cosmos DB", json_array_from_cosmos)
                if cosmos_db_output_length != 0:
                    display_output_in_table(json_array_from_cosmos)
                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Apologies!! Couldn't find data with the given details"})
                    st.chat_message("user").write("Apologies!! Couldn't find data with the given details")


if __name__ == "__main__":
    main()
