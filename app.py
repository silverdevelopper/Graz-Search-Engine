import streamlit as st
from utility import get_search_query_engine,query,convert_to_mark_down_result
st.title("Search Engine Demo :gear:")
path = 'demo-graz.parquet'

is_api_results = True
with st.sidebar:
    is_api_results = st.checkbox('Include API results',value=True)

    if is_api_results:
        st.write('Api result will be added to search results!!')
        
    select = st.selectbox('Select index', ['demo-graz.parquet', 'demo-simplewiki.parquet.gz','demo-snapshot.parquet.gz'])
    
    st.write('Index: ', select)
    

if f"engine_{path}" not in st.session_state:
    st.session_state[f"engine_{path}"] = get_search_query_engine(path)


def search(q):
    return query(f"'{q}'",st.session_state[f"engine_{path}"],is_api_results)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hey there search anything you want"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# React to user input
if prompt := st.chat_input("Search anything"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Searching..."):    
        response,img_url = search(prompt)
    # Add user message to chat history
    
    st.session_state.messages.append({"role": "user", "content": prompt})
  
    with st.chat_message("assistant"):
        result_markdown = convert_to_mark_down_result(response,img_url)
        st.markdown(result_markdown,unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": result_markdown})