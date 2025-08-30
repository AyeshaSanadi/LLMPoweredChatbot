import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

# Creating prompt template
template = """You are an AI chatbot having a conversation with a humna.

{history}
Human: {human_input}
AI:"""

# Adding prompt
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
print(prompt)


msgs = StreamlitChatMessageHistory(key="chat_history")

# Adding memory
memory = ConversationBufferMemory(memory_key = "history", chat_memory=msgs)

def load_chain():
    llm = ChatOpenAI(temperature=0)
    # Adding an LLMChain with memory and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return llm_chain


def initialize_session_state():
    if "chain" not in st.session_state:
        st.session_state.chain = load_chain()


initialize_session_state()

st.set_page_config(page_title= "Langchain Chatbot Demo")
st.header("Chatbot using Langchain")

def submit():
    st.session_state.user_input = st.session_state.widget_input
    st.session_state.widget_input = ""


st.text_input("You:", key="widget_input", on_change=submit)


if st.session_state.user_input:
    output = st.session_state.chain.invoke(st.session_state.user_input)["text"]
    st.session_state.past.append(st.session_state.user_input)
    st.session_state.generated.append(output)
    st.session_state.user_input = ""

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
