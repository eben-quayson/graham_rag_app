
import streamlit as st
import requests

# Initialize session state for queries
if 'queries' not in st.session_state:
    st.session_state.queries = []

st.title(":rainbow[GrahamPedia]")
st.write("What do you want to know about Graham's essays:")

# Function to add a new query input box
def add_query():
    st.session_state.queries.append({'query': '', 'response': ''})

# Display existing queries and their responses
for i, q in enumerate(st.session_state.queries):
    query_key = f'query_{i}'
    response_key = f'response_{i}'

    # Display the query input box
    st.session_state.queries[i]['query'] = st.text_input(
        f"Ask Graham:", value=q['query'], key=query_key)

    #Base URL
    BASE_URL = "https://graham-rag-app-1.onrender.com/"

    # If the query is not empty and response is empty, process the query
    if st.session_state.queries[i]['query'] and not st.session_state.queries[i]['response']:
        with st.spinner("Processing..."):
            response = requests.post(
                BASE_URL + "/process_query",
                json={"query": st.session_state.queries[i]['query']}
            )
            if response.status_code == 200:
                st.session_state.queries[i]['response'] = response.json().get("answer")
            else:
                st.session_state.queries[i]['response'] = "Error processing the query."

    # Display the response
    if st.session_state.queries[i]['response']:
        st.write("Response:")
        st.components.html(st.session_state.queries[i]['response'])

# Button to add a new query input box
if st.button("Ask New Question"):
    add_query()
