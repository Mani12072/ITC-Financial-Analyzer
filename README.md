# 📊 ITC Financial Analyzer APP – AI-Powered Financial Q&A

An interactive AI tool designed to explore **ITC Ltd’s financial journey** — revenues, profitability, and performance — via a **chat-based interface** powered by **web scraping**, **vector embeddings**, and **LLMs**.

## You can access the Streamlit app from the link below:
### 📊 [ITC Financial Analyzer APP – AI-Powered Financial Q&A](https://huggingface.co/spaces/Mpavan45/ITC_Financial_Analysis)

**🌟 Key Capabilities**

**🤖 Smart Data Scraper:** Real-time financial disclosures from ITC’s website using Tavily API.  
**🧬 Embedding Engine:** Transforms financial text into  vector embeddings for intelligent retrieval.  
**🧠 Conversational AI:** Chat with a financial assistant that pulls insights directly from ITC’s disclosures.  
**🖥️ Streamlit Interface:** Clean, interactive dashboard to ask financial queries and receive grounded answers.

## 🔧 Project Modules and Pipeline
## 📁 Project Structure

```
itc-financial-analysis/
├── Data Scraping               # Tavily API scripts for data extraction
├── Database-SQLite             # SQLite schema and queries for data storage
├── Embeddings                  # Embedding generation and Chroma DB setup
├── LLM Implementation          # LLM query handling with Google Gemini
├──data based
Chrom db zip and Unziped Files    # Embeddings data base
├── app.py                        # Streamlit UI for interactive queries
├── requirements.txt              # Project dependencies
└── README.md                     # Setup, usage

```
---

## 📁 1. Data Scraping (/scraper)

**Goal**: Extract ITC’s 2023 & 2024 annual reports, quarterly reports, and consolidated/standalone financial statements.

**Tools Used**:  
**🔥 Tavily API:** Scrapes financial data and metadata (source, year) from ITC’s investor relations page.

**Scraped Content**:

- Annual Reports (2023, 2024)  
- Quarterly Reports (Q1–Q4 for 2023, 2024)  
- Consolidated & Standalone Financial Statements

**Terminology**:

- **Annual Reports**: Comprehensive yearly financial disclosures covering revenue, profit, and operational performance.  
- **Quarterly Reports**: Financial updates released every three months, detailing short-term performance.  
- **Consolidated Statements**: Financials combining ITC and its subsidiaries.  
- **Standalone Statements**: Financials of ITC as a single entity, excluding subsidiaries.

**Implementation**:  
- The Tavily API scrapes PDFs from ITC’s website (https://www.itcportal.com/investor-relations/financial-reports/).
-  Metadata (source, year) is extracted alongside text to ensure traceability. Refer to `Data Scraping/Data Scraping and Cleaning.py` for details.

**Example Code**:
```python
from tavily import Tavily

# Initialize Tavily client
api_key = "your_tavily_api_key"
client = Tavily(api_key)

# Scrape financial report
url = "https://www.itcportal.com/investor-relations/financial-reports/"
response = client.scrape_pdf(url)

# Extract text and metadata
extracted_text = response['text']
metadata = response['metadata']

print(f"Extracted Text: {extracted_text[:500]}...")
print(f"Metadata: {metadata}")
```

## 📁 2. Data Storage in SQLite (/Database-Sql)
**Goal**: Store scraped financial data in a structured SQLite database for efficient querying and analysis.

**Steps**:  
- Created a SQLite database to store raw financial data and metadata (source).  
- Stored document details like content and source for traceability.  
- Saved extracted text for downstream processing.
```
SQLite Schema:
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    source TEXT
);
```
**Implementation:**
The database is initialized using the above schema, and scraped data is inserted with metadata (source, year). 
- Refer to database/setup_db.py for schema creation and data insertion logic.

##  📁 3. Embedding Layer (/Embeddings)

**Goal:** Convert scraped financial data into searchable vector embeddings for retrieval.
Steps:  

**Text Preprocessing:** Cleaned and chunked text using RecursiveCharacterTextSplitter.  
**Embedding Generation:** Used Google’s embedding model (textembedding-gecko) to convert text into vectors.  
**Storage:** Stored embeddings in a Chroma vector database (chroma_db1.zip).  
**Retrieval Testing:** Performed sample retrieval tests to validate embedding quality.

**Chroma Vector Database Setup:** 

- 📁 ZIP File: database/chroma_db1.zip  
- 📂 Unzipped Path: database/chroma_db1/  
- ⚙️ The Chroma DB is loaded into memory during execution using LangChain’s Chroma vector store.  
- 🔒 Ensure the extracted folder is writable to avoid read-only errors.

**Usage Flow:**

- Extract chroma_db1.zip to database/chroma_db1/.

- Load the Chroma DB with Chroma
(persist_directory='database/chroma_db1', ...).  

- Use the vector store as the retriever for the RAG pipeline.

**Implementation:**
- Refer to Embeddings/embeddings.py for embedding generation and Chroma DB setup.

## 📁 4. LLM Query Interface (/llm)

**Goal:** Enable natural language queries using retrieved financial data as context.
Components:  

- 🤖 Google Gemini 2.0 Flash: Integrated via langchain-google-genai for answering queries.  
- 🎯 MMR-based Retrieval: Ensures diverse and relevant document chunks are retrieved.  
- 📄 Source Citation: Provides transparency (e.g., "Source: ITC Annual Report 2023, Page 12").

**Example Queries:**

- “What was ITC’s revenue in 2024?”  
- “Compare profitability in 2023 vs 2024.”  
- “What was ITC’s stock price on May 10, 2025?”  
- “Is ITC’s revenue trending upward (2023 vs. 2024)?”

**Implementation:** A ChatPromptTemplate is used to structure LLM queries with retrieved context. 
- Refer to llm/query_llm.py for details.

## 📁 5. Streamlit Chat App (app.py)
**Goal:** Provide an interactive front-end for financial Q&A.

**Features:** 

- 🧠 Chat with Memory: Maintains conversation context across queries.  
- 🧾 Source-backed Answers: Displays citations for transparency.  
- 📊 Bullet-format Summaries: Summarizes key financial KPIs.  
- 🎯 Year-wise Breakdowns: Organizes answers by year for clarity.

**Implementation:**
The Streamlit app integrates all modules (scraper, database, embeddings, LLM) into a unified interface. 
- Refer to app.py for the complete code.

## 🚀 Setup Instructions
- Local Setup

**Clone the Repository:**
- git clone https://github.com/your-username/itc-financial-analysis.git
- cd itc-financial-analysis


**Install Dependencies:**

- pip install -r requirements.txt


**Set Up Environment Variables:**

- Create a .env file in the root directory.
- Add your Tavily API key and Google API key:

        TAVILY_API_KEY=your_tavily_api_key
        GOOGLE_API_KEY=your_google_api_key




**Extract Chroma DB:**

- Unzip database/chroma_db1.zip to database/chroma_db1/.
- Ensure the folder is writable.


- Run the Streamlit App:
        
        streamlit run app.py


**Access the App:**

- Open your browser and navigate to http://localhost:8501.
- Start asking financial questions about ITC!



**Hugging Face Spaces Setup**
 
 - You can deploy the Streamlit app on Hugging Face Spaces for easy access and sharing.

- Create a Hugging Face Space:

- Go to Hugging Face Spaces and sign in.
- Click Create new Space.
- Choose Streamlit as the framework and name your Space (e.g., ITC-Financial-Analyzer).
- Set visibility to Public or Private.


**Upload the Repository:**

- Clone your GitHub repository locally.
- Push to your Hugging Face Space:git remote add space https://huggingface.co/spaces/your-username/ITC-Financial-Analyzer
- git push space main




**Configure Environment Variables:**

- Go to Settings > Variables and Secrets in your Space.
- Add:

        TAVILY_API_KEY: Your Tavily API key.
        GOOGLE_API_KEY: Your Google API key.




**Add Chroma DB:**

- Upload database/chroma_db1.zip to the Space’s file system.
- Update app.py to extract the ZIP file during initialization:import zipfile
```
    import os

    zip_path = "database/chroma_db1.zip"
    extract_path = "database/chroma_db1"
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

```
**Deploy the Space:**

- Hugging Face will build and deploy the app.
- Access the app via the provided URL (e.g., https://huggingface.co/spaces/your-username/ITC-Financial-Analyzer).


**Test the App:**

- Open the app URL and interact with the financial Q&A interface.
- Verify Chroma DB loading and LLM query responses.




## 📜 Requirements
- See requirements.txt for dependencies.
 **Key libraries:**

    tavily: Web scraping
    sqlite3: Database storage
    langchain: Embeddings and LLM integration
    streamlit: Interactive UI
    google-generativeai: Google Gemini LLM
    chromadb: Vector storage

## 📝 Notes

**Accuracy:** LLM answers are grounded in scraped data, validated against ITC’s reports.  
**Reproducibility:** Fully documented for local and Hugging Face deployment.  
**Scalability:** Modular code, extensible to other years/companies.  



