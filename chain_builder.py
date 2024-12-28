from langchain_mistralai import ChatMistralAI
from utils import get_vector_store
from config import Config
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

#LLM Configuration

llm = ChatMistralAI(model=Config.LLM_MODEL,
                    api_key=Config.MISTRALAI_API_KEY,
                    temperature=0,
                    timeout=600,
                    streaming=False)

vector_store_instance = get_vector_store()
retriever = vector_store_instance.as_retriever()

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", Config.CONTEXTUALIZE_Q_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", Config.SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create Retrieval Chains
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)


def rag_query(query, chat_history=None):
    if chat_history is None:
        chat_history = []

    # Invoke the RAG chain
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return response['answer']
