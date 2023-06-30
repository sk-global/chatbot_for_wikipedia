import streamlit as st
from llama_index import download_loader
from llama_index import LLMPredictor
from llama_index import GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain import OpenAI
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

if 'response' not in st.session_state:
    st.session_state.response = ''

def send_click():
    
    query_engine = index.as_query_engine(service_context=service_context, \
                                         verbose=True, response_mode="compact")
    st.session_state.response = query_engine.query(st.session_state.prompt)

index = None
st.title("Chatbot for Wikipedia")

sidebar_placeholder = st.sidebar.container()
user_input = st.text_input("""Enter the wikipedia page 
                              title and press enter""", \
                           "Artificial intelligence")

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=[user_input])

sidebar_placeholder.header('Current Processing Wiki Page:')
sidebar_placeholder.subheader(user_input)
sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, \
                                        model_name="text-davinci-003"))

max_input_size = 4096
num_output = 256
max_chunk_overlap =0.2
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, \
                                               prompt_helper=prompt_helper)

index = GPTVectorStoreIndex.from_documents(
        documents, \
  service_context=service_context, \
  prompt_helper=prompt_helper
    )

if index != None:
    st.text_input("""Ask something based on above 
                      wikipedia page and press send button: """, \
                  key='prompt')
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")


