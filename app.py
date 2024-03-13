import streamlit as st
from lang_bot import get_out_for_text_inp

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
    model_text_output = get_out_for_text_inp(text_input)
    llama2_out = st.text_area(
    "Llama2",
    model_text_output
    )

#Returning user input to backend
# print(type(title))
