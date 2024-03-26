import streamlit as st
from lang_bot import chat_with_bot

#Title
st.title('This is a title')
st.title('_Streamlit_ is :blue[cool] :sunglasses:')

#Input chat
# def text_prompt():
text_input = st.text_input("ad",placeholder= "Say Something!", label_visibility='collapsed')
st.write(text_input)

#Chat submit button
st.button("Submit", type="primary")

if text_input:
    model_text_output = chat_with_bot(query= text_input)
    llama2_out = st.text_area(
    "Llama2",
    model_text_output
    )

#Returning user input to backend
# print(type(title))
