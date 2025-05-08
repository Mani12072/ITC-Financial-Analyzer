import streamlit as st
import zipfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="📈 ITC Ltd. Financial Insight Bot", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>💬 Financial Assistant for ITC Ltd.</h1>", unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.header("💼 App Settings")
    st.markdown("This assistant analyzes ITC Ltd.'s earnings calls & financial transcripts.")
    st.markdown("---")
    st.info("Data Source: CHROMA_DB_BACKUP.zip", icon="📦")
    st.caption("Powered by Gemini | Vector Search via Chroma | Embeddings from HuggingFace")

# --- Extract Chroma DB ---
with zipfile.ZipFile("chroma_db1.zip", "r") as zip_ref:
    zip_ref.extractall("chroma_storage")

# --- Vector DB Setup ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_storage", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7})

# --- Helper: Extract Context ---
def extract_context(question):
    documents = retriever.get_relevant_documents(question)
    content = "\n\n".join(doc.page_content for doc in documents)
    return {"question": question, "context": content, "docs": documents}

# --- Prompt Template ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a financial analyst bot. Your responses must be derived strictly from the following transcript context: {context}

Rules:
- Extract only factual data—avoid assumptions.
- If data is missing, state: "Required data not found in transcript."
- Present responses in bullet points where relevant.
- Tailor insights to ITC Ltd., but keep logic modular for generalization.

Respond concisely to the question below using the context above.
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}")
])

# --- LLM & Chains Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser = StrOutputParser()
chat_memory = {"chat_history": []}

initial_chain = RunnableLambda(lambda x: {"input": x["input"], **extract_context(x["input"])})
contextual_chain = initial_chain | RunnableLambda(lambda x: {
    "llm_input": {"input": x["input"], "context": x["context"]},
    "docs": x["docs"]
})
response_chain = contextual_chain | RunnableLambda(lambda x: {
    "result": (chat_prompt | llm | parser).invoke(x["llm_input"]),
    "source_documents": x["docs"]
})
final_chain = RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: chat_memory["chat_history"])) | response_chain

# --- Chat History Display ---
for message in chat_memory["chat_history"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        bubble_color = "#E8F0F2" if role == "user" else "#F0F5E1"
        st.markdown(
            f"<div style='background-color:{bubble_color}; padding:10px; border-radius:10px;'>{message.content}</div>",
            unsafe_allow_html=True
        )

# --- Chat Input ---
user_input = st.chat_input("Type your financial question about ITC Ltd...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    chat_memory["chat_history"].append(HumanMessage(content=user_input))
    result = final_chain.invoke({"input": user_input})
    reply = result["result"]
    sources = result.get("source_documents", [])

    chat_memory["chat_history"].append(AIMessage(content=reply))
    with st.chat_message("assistant"):
        st.markdown(f"<div style='background-color:#F0F5E1; padding:10px; border-radius:10px;'>{reply}</div>", unsafe_allow_html=True)

        if sources:
            st.markdown("#### 🔗 Source Documents:")
            for doc in sources:
                st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")

