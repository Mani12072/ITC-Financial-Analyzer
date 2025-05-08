# 📊 ITC Financial Analyzer APP – AI-Powered Financial Q&A

An interactive AI tool designed to explore **ITC Ltd’s financial journey** — revenues, profitability, and performance — via a **chat-based interface** powered by **web scraping**, **vector embeddings**, and **LLMs**.

---

## 🌟 Key Capabilities

- 🤖 **Smart Data Scraper**: Real-time financial disclosures from ITC’s website using *Firecrawl*, *Crew AI*, or *Tavily*.
- 🧬 **Embedding Engine**: Transforms financial text into vector embeddings for intelligent retrieval.
- 🧠 **Conversational AI**: Chat with a financial assistant that pulls insights directly from ITC’s disclosures.
- 🖥️ **Streamlit Interface**: Clean, interactive dashboard to ask financial queries and receive grounded answers.

---

## 🔧 Project Modules and Pipeline

### 📁 1. Data Scraping (`/scraper`)
**Goal**: Extract ITC’s 2023 & 2024 quarterly reports and consolidated financial presentations.

**Tools Used**:  
- 🔥 Firecrawl or 🤖 Tavily API  
- 📄 Scraped PDF content from:
  - Annual Reports (2023, 2024)
  - Quarterly Reports (Q1–Q4)
  - Consolidated & Standalone Statements

### 🧑‍💻 Example: Using **Tavily API** for Scraping

Tavily API helps scrape and extract content from PDFs, including detailed metadata. Below is an example of how to use the **Tavily API** to scrape financial reports:

#### Step 1: Install Tavily

```bash
from tavily import Tavily

# Initialize the Tavily client with your API key
api_key = "your_tavily_api_key"
client = Tavily(api_key)

# Define the URL of the financial report you want to scrape
url = "https://www.itcportal.com/investor-relations/financial-reports/"

# Scrape the PDF data (ensure you have the correct URL for the document)
response = client.scrape_pdf(url)

# Extract text and metadata
extracted_text = response['text']  # Raw financial data text
metadata = response['metadata']   # Metadata like document type, date, etc.

# Print the extracted data
print(f"Extracted Text: {extracted_text[:500]}...")  # First 500 characters of the report
print(f"Metadata: {metadata}")

```
**Output**: Extracted raw text + metadata


### 📁 2. Data Storage in SQLite (`/database`)
**Goal**: Store the scraped financial data in a structured SQLite database for easy querying and analysis.

**Steps**:
- **Create a SQLite database** to store scraped financial documents.
- Store relevant metadata, such as document type, date, and page number, along with the extracted text.
- Store the extracted **raw financial data** for further processing.

**SQLite Schema Example**:
```sql
CREATE TABLE financial_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_type TEXT,
    year INTEGER,
    quarter INTEGER,
    report_text TEXT,
    metadata JSON
);
```
### 🧠 3. Embedding Layer (`/embeddings`)
**Goal**: Convert scraped data into searchable vector format

**Steps**:
- 🔄 Clean & chunk text using `RecursiveCharacterTextSplitter`
- ⚙️ Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- 🗃️ Store in local **Chroma DB** (`CHROMA_DB1zip`)
- 🧳 Zip DB folder for deployment and reuse

---

### 💬 4. LLM Query Interface (`/llm`)
**Goal**: Answer natural language questions using **retrieved context**

**Components**:
- 🤖 **Google Gemini 2.0 Flash** via `langchain-google-genai`
- 🎯 **MMR-based Retrieval** for diverse, relevant chunks
- 📄 **Source Citation** for transparency (e.g., "ITC Annual Report 2023, Page 12")

**Example Queries**:
- “What was ITC’s revenue in 2024?”
- “Compare profitability in 2023 vs 2024.”
- “Show stock price trend around Q1 2023.”

---

### 🖼️ 5. Streamlit Chat App (`app.py`)
**Goal**: Visual front-end for financial Q&A

**Features**:
- 🧠 Chat with memory to maintain context
- 🧾 Answers backed by source docs
- 📊 Bullet-format summaries of financial KPIs
- 🎯 Clear year-wise breakdowns


