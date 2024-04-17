import requests
import streamlit as st

# Setting up Streamlit page
st.set_page_config(
    page_title="Query Analysis Tool",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Query Analysis💬")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = []


# Define function to handle form submission
def handle_form_submission(question, json_string,query_text):
    # Send POST request to server as form
    url = "http://127.0.0.1:5000/EAO_chat"
    data = {"question": question, "json_string": json_string, "query_text" : query_text}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        response_json = response.json()
        st.session_state.messages.append({"role": "User", "content": question})
        st.session_state.messages.append({"role": "Query", "content": json_string})
        st.session_state.messages.append({"role": "Query", "content": query_text})
        st.session_state.messages.append(
            {"role": "Assistant", "content": response_json["response"]}
        )
        # Display the response immediately
        # Display chat history
        for message in st.session_state.messages:
            with st.expander(message["role"]):
                st.write(message["content"])
    else:
        st.error("Error: Unable to get response from server")


# Create form
with st.form(key="my_form"):
    question = st.text_input(
        label="Question",
        value="Point out the main reasons for the poor performance of this query",
    )
    json_string = st.text_area(label="JSON String")
    query_text = st.text_area(label="QUERY String")
    submit_button = st.form_submit_button(label="Submit")

# Display chat history
for message in st.session_state.messages:
    with st.expander(message["role"]):
        st.write(message["content"])

# Handle form submission
if submit_button:
    handle_form_submission(question, json_string,query_text)