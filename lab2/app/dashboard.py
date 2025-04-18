import streamlit as st

st.title("Simple Streamlit App")

st.write("Hello from inside Docker!")

# Add more Streamlit components here if needed
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")
