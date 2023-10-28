from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import RetrievalQA

from generate_log import log_update

# for token-wise streaming, so you'll see the answer gets generated token by token when Llama is answering your question
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def prompt_template():
    # prompt template
    # template = """Question: {question}
    #
    # Answer: Let's work this out in a step by step way to be sure we have the right answer."""
    #
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # prompt = PromptTemplate.from_template(
    #     "who wrote {book}?"
    # )

    log_update("Initialising chat prompt template", 2)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "You're an assistant that helps in answering questions based on the context."
    #      "You return precise and to the point answers. You try to answer the questions in the minimum words possible "
    #      "without loosing any relevant information that the question asks for."
    #      "--------------------------"
    #      "{context}"),
    #     ("human", "{question}"),
    # ])
    prompt = PromptTemplate.from_template(
        "Context : {context}"
        "- You are an assistant that answers user questions based on the above context."
        "- You return precise and to the point answers. You try to answer the questions in the minimum words possible "
        "without loosing any relevant information that the answer needs from the context."
        ""
        "Question : {question}"
    )
    return prompt


def create_general_llm_chain(llm, prompt):
    # creating llm chain
    log_update("Creating the llm chain", 2)
    return LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())


def create_retriever_llm_chain(llm, prompt, vectordb):
    # creating retriever llm chain
    log_update("Creating a retriever llm chain", 2)
    # return RetrievalQA.from_chain_type(llm=llm, prompt=prompt, output_parser=StrOutputParser(), retriever=vectordb.as_retriever())
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt
        },
    )


def create_llm_chain(llm, prompt, vectordb=None):
    if vectordb is not None:
        return create_retriever_llm_chain(llm, prompt, vectordb)
    else:
        return create_general_llm_chain()


def call_chain(chain, question):
    # calling chain with the user inputs
    log_update("Running chain with user inputs", 2)
    return chain.run(question)


def initialize_llama(vectordb=None):
    # Initialize llama model
    log_update("Starting Initialization of Llama 2 model", 1)
    prompt_temp = prompt_template()

    n_gpu_layers = 1  # determines how many layers of the model are offloaded to your Metal GPU, in the most case, set it to 1 is enough for Metal
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

    log_update("Creating model", 2)
    llm = LlamaCpp(
        model_path="../models/llama-2-7b.Q5_K_S.gguf",
        temperature=0.5,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        max_tokens=200,
        top_p=1,
        n_ctx=6000,
        callback_manager=callback_manager,
        verbose=True
    )
    chain = create_llm_chain(llm, prompt_temp, vectordb)
    log_update("Initialization of Llama 2 model completed", 1)
    return chain


def return_answer(chain, question):
    # Answering user question
    return call_chain(chain, question)
