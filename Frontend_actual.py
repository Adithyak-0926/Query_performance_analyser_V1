import pandas as pd
import base64
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setting up Streamlit page
st.set_page_config(page_title="Query Analysis Tool", layout="centered", initial_sidebar_state="auto")
st.title("Query Analysis💬")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_row(question, json_string, query_text):
    missing_fields = []
    if not question:
        missing_fields.append("question")
    if not json_string:
        missing_fields.append("json_stats")
    if not query_text:
        missing_fields.append("query")

    if missing_fields:
        return f"Missing field(s): {', '.join(missing_fields)}", 0  # No processing time for missing fields
    url = "http://127.0.0.1:8000/EAO"
    data = {"question": question, "json_string": json_string, "query_text": query_text}
    start_time = time.time()
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        result = response.json()['response']
    except requests.exceptions.RequestException as e:
        result = f"Request error occurred: {str(e)}"
    processing_time = round(time.time() - start_time, 2)
    return result, processing_time

# Dropdown for selecting mode
mode = st.selectbox("Choose your input type:", ["Single Query", "CSV Upload"])

if mode == "Single Query":
    with st.form("single_query_form"):
        question = st.text_input("Question", "point out the main reasons for the queries poor performance.")
        json_string = st.text_area("JSON String")
        query_text = st.text_area("QUERY String")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            result, processing_time = process_row(question, json_string, query_text)
            # Append to session state for history tracking
            st.session_state.messages.append({
                "role": "User",
                "content": f"Question: {question}\nJSON: {json_string}\nQuery: {query_text}"
            })
            st.session_state.messages.append({
                "role": "Assistant", "content": f"Response: {result}"
            })

    # Display chat history
    for message in st.session_state.messages:
        with st.expander(message["role"]):
            st.write(message["content"])

elif mode == "CSV Upload":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        queries = df["query"].tolist()
        json_strings = df["json_stats"].tolist()
        questions = df["question"].tolist()

        if st.button("Process CSV"):
            start_time = time.time()
            responses = []
            batch_size = 10
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                future_to_request = {}
                for i in range(0, len(queries), batch_size):
                    batch_queries = queries[i:i+batch_size]
                    batch_jsons = json_strings[i:i+batch_size]
                    batch_questions = questions[i:i+batch_size]
                    for j, (question, json_string, query) in enumerate(
                            zip(batch_questions, batch_jsons, batch_queries)):
                        if j > 0:
                            time.sleep(1)  # delay to avoid server throttling
                        future = executor.submit(process_row, question, json_string, query)
                        futures.append(future)
                        future_to_request[future] = (question, json_string, query)

                for future in as_completed(futures):
                    question, json_string, query = future_to_request[future]
                    try:
                        response, processing_time = future.result()
                        responses.append((question, query, json_string, response, processing_time))
                    except Exception as e:
                        responses.append((question, query, json_string, str(e), None))

            response_df = pd.DataFrame(
                responses, columns=["Question", "Query", "JSON Stats", "Response", "Processing Time"]
            )
            csv = response_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_results.csv">Download Processed Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            total_time = time.time() - start_time
            st.write(f"Total time taken for this whole CSV to generate is {total_time:.2f}s")
