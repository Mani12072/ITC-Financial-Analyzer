import streamlit as st
import os
import zipfile
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# Page setup
st.set_page_config(page_title="Financial QA - ITC Ltd.", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header { text-align: center; padding: 20px; background-color: #007bff; color: white; border-radius: 10px; }
    .stTextInput>input { border-radius: 5px; padding: 10px; }
    .stButton>button { background-color: #28a745; color: white; border-radius: 5px; padding: 10px; width: 100%; }
    .answer-box { background-color: #e9ecef; border-radius: 10px; padding: 15px; margin-top: 10px; }
    .source-expander { background-color: #f1f3f5; border-radius: 5px; }
    .sidebar .stSelectbox { margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# Header
with st.container():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("📊 Financial Q&A Chatbot (ITC Ltd.)")
    st.markdown("Ask financial questions about ITC Ltd. based on transcript data, powered by AI.")
    st.markdown('</div>', unsafe_allow_html=True)


# Safe way to access secrets
GOOGLE_API_KEY = st.secrets.get('genai')

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embedding)


retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})
llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash", temperature=1)
parser = StrOutputParser()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a domain-specific AI financial analyst focused on company-level performance evaluation.
Your task is to analyze and respond to user financial queries strictly based on the provided transcript data: {context}.
Rules:
1. ONLY extract facts, figures, and insights that are explicitly available in the transcript.
2. If data is missing or partially available, clearly state: "The required data is not available in the current transcript." Then provide a generic but relevant explanation based on standard financial principles.
3. Maintain numerical accuracy and avoid interpretation beyond data boundaries.
4. Prioritize answers relevant to ITC Ltd., but keep response format adaptable to other firms and fiscal years.
5. Clearly present year-wise or metric-wise insights using bullet points or structured formats if applicable.
Your goals:
- Ensure 100% fidelity to source transcript.
- Do not assume or hallucinate missing numbers.
- Use clear, reproducible reasoning steps (e.g., show which line items support your conclusion).
- Output should be modular enough to scale across other companies and time periods.
Respond only to this question from the user."""
    ),
    ("human", "{question}")
])

# Helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_and_answer(question):
    if not retriever or not llm:
        return "Cannot process query: Retriever or LLM not initialized.", []
    docs = retriever.invoke(question)
    context = format_docs(docs)
    final_input = {"question": question, "context": context}
    result = (prompt | llm | parser).invoke(final_input)
    return result, docs

# Query input form
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Enter your question about ITC's financials:", placeholder="e.g., What was ITC's revenue in FY 2023?")

with col2:
    clear = st.button("Clear ❌")

submit_button = st.button("Get Answer ✅")

# Clear the input (simulate reset)
if clear:
   st.rerun()


if submit_button:
    if not query.strip():
        st.warning("Please enter a valid question.")
    elif not GOOGLE_API_KEY:
        st.error("Google API Key not configured. Set it in Hugging Face Secrets to proceed.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer, source_docs = retrieve_and_answer(query)
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### ✅ Answer")
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📄 Source Documents", expanded=False):
                    if source_docs:
                        for doc in source_docs:
                            st.markdown(f"- **Source**: {doc.metadata.get('source', 'Unknown document')}")
                            st.markdown(f"  **Content**: {doc.page_content}")
                    else:
                        st.write("No source documents found.")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
