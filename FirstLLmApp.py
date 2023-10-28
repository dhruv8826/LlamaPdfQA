# import logging
#
# # Create and configure logger
# logging.basicConfig(filename="logs/firstLLmApp.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')
#
# logger=logging.getLogger()
# logger.setLevel(logging.DEBUG)
#
# def logUpdate(message):
#     logger.info(message)
#     print(message)



from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

# for token-wise streaming, so you'll see the answer gets generated token by token when Llama is answering your question
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="../models/llama-2-7b.Q5_K_S.gguf",
    temperature=0.0,
    top_p=1,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True
)

# question = "who wrote the book Innovator's dilemma?"
# answer = llm(question)

# a more flexible way to ask Llama general questions using LangChain's PromptTemplate and LLMChain
prompt = PromptTemplate.from_template(
    "who wrote {book}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("innovator's dilemma")

print(answer)



from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("resources/RilDoc.pdf")
logUpdate("reading pdf")
documents = loader.load()
logUpdate("read and loaded pdf")

# quick check on the loaded document for the correct pages etc
print(len(documents))
# print(documents[1])



from langchain.vectorstores import Chroma

# embeddings are numerical representations of the question and answer text
from langchain.embeddings import HuggingFaceEmbeddings

# use a common text splitter to split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


# split the loaded documents into chunks
logUpdate("creating chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
logUpdate("all splits/chunks created")

# create the vector db to store all the split chunks as embeddings
logUpdate("Creating embeddings")
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
)
logUpdate("vectordb set")




from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
logUpdate("creating chain")


question = "Which company is this annual report for?"
result = qa_chain({"query": question})

logUpdate(f"question - {question}")
logUpdate(f"answer - {result}")