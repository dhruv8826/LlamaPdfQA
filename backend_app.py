import sys

from llama_cpp_methods import initialize_llama, return_answer
from pdf_reader import load_pdf
from generate_log import set_log_filename
from set_hugging_face_embeddings_for_chroma_vectorstore import save_embeddings_on_vector_store
from generate_log import log_update

import time
import os

doc = None
chain = None

set_log_filename("logs/analyser1.log")


def save_uploaded_file(uploaded_file):
    with open(os.path.join("resources", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return True


def upload_pdf(uploaded_file):
    log_update("uploading file")
    global doc
    if doc is None:
        if save_uploaded_file(uploaded_file):
            doc = load_pdf(uploaded_file.name)
            return True
        # doc = load_pdf("resources/RilDoc.pdf")


def upload_dummy_pdf():
    global doc
    doc = load_pdf("RilDoc.pdf")


def initialize_model():
    if doc is not None:
        global chain
        vectordb = save_embeddings_on_vector_store(doc)
        chain = initialize_llama(vectordb)
        return "Success"
    else:
        return "Error"


# answer = return_answer(chain, "who wrote innovator's dilemma?")

def chat():
    user_input = input("\n Enter your message (or 'q' to quit): ")
    if user_input == 'q':
        print("Program terminated.")
        return
    else:
        print(f"You entered: {user_input}")
        return_answer(chain, user_input)
        time.sleep(2)
        chat()


def get_response(user_input):
    # if user_input == 'q':
    #     sys.exit()
    # else:
    return return_answer(chain, user_input)


def dummy_run():
    upload_dummy_pdf()
    initialize_model()
    chat()


# dummy_run()
