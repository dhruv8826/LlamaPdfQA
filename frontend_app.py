import streamlit as st

from pdf_app import pdf_component

from backend_app import initialize_model, get_response

st.title("Llama2-powered PDF Q/A App")

loading_message = "Please wait for the model to load the pdf file."

# print(st.session_state)

if 'pdf_uploaded_name' not in st.session_state:
    st.session_state.pdf_uploaded_name = None

if 'file_uploaded_and_model_uploaded' not in st.session_state:
    st.session_state.file_uploaded_and_model_uploaded = False

with st.sidebar:
    pdf_uploaded_name = pdf_component()
    # print(st.session_state.pdf_uploaded_name)
    # print(pdf_uploaded_name)
    # print(st.session_state.file_uploaded_and_model_uploaded)
    if pdf_uploaded_name != st.session_state.pdf_uploaded_name or pdf_uploaded_name is None:
        st.session_state.pdf_uploaded_name = pdf_uploaded_name
        st.session_state.file_uploaded_and_model_uploaded = False


def generate_response(input_text):
    return get_response(input_text)


def submit_action(question):
    return generate_response(question)


# pdf_uploaded = pdf_component()


if not st.session_state.file_uploaded_and_model_uploaded and st.session_state.pdf_uploaded_name is not None:
    # print("\n\ninitialised\n\n")
    error = initialize_model()
    if error != "Success":
        st.error("Error occurred")
    st.session_state.file_uploaded_and_model_uploaded = True

with st.form("my_form"):
    # print(st.session_state.file_uploaded_and_model_uploaded)
    question = st.text_input("Enter your question here:")
    # st.warning("Enter 'q' to quit")
    submitted = st.form_submit_button("Submit", disabled=not st.session_state.file_uploaded_and_model_uploaded)
    if not st.session_state.file_uploaded_and_model_uploaded:
        st.info(loading_message)

if st.session_state.file_uploaded_and_model_uploaded and submitted:
    # print("response of action submit")
    answer = submit_action(question)
    st.write(answer)
