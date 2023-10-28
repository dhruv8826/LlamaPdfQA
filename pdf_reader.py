from langchain.document_loaders import PyPDFLoader

from generate_log import log_update


def load_pdf(pdf_file):
    # Loading the pdf file
    print("type of file passed- ", type(pdf_file))
    loader = PyPDFLoader("resources/" + pdf_file)
    log_update("reading pdf - %s" % pdf_file, 1)
    documents = loader.load()
    log_update("read and loaded pdf", 2)

    # Quick check on the document loaded
    log_update("Document loaded details ->", 0)
    log_update("doc length - " + str(len(documents)), 0)

    # return the documents loaded
    return documents
