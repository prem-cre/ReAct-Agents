# 🛡️ Guardian AI – Compliance Policy Evaluation Agent

**Guardian AI** is an intelligent compliance analysis agent built with **LangChain**, **FAISS**, and **ReAct reasoning**.  
It uses **retrieval-augmented generation (RAG)** to analyze text against **company policy documents**, providing automated compliance checks, explanations, and grounded evidence.

---

## 🚀 Features

- 🧠 **ReAct-based reasoning agent** (retrieval + reasoning + action)
- 📚 **RAG pipeline** powered by FAISS and HuggingFace embeddings
- 🧾 **Policy document ingestion** (PDF support)
- 🔍 **Context-aware compliance checking** using vector search
- 🗣️ **Structured natural-language analysis** with grounded citations
- 🪶 Runs fully in **Google Colab**

---

## 🏗️ System Architecture

Input Text ─┐
│
▼
Guardian Agent (ReAct)
├── Uses policy_search_tool()
│ └── Queries FAISS Vector DB
├── Invokes LLM (Reason + Act)
└── Outputs:
- Compliance verdict
- Supporting policy text
- Explanation


---

## ⚙️ Setup (Colab or Local)

### 1️⃣ Clone or open notebook
```bash
git clone https://github.com/yourusername/guardian-ai.git
cd guardian-ai


or open the Colab notebook:

Guardian_AI_Colab_Notebook.ipynb

2️⃣ Install dependencies
!pip install langchain faiss-cpu pypdf sentence-transformers

3️⃣ Load environment

Create a .env file with your model key:

GROQ_API_KEY=your_api_key_here

4️⃣ Index your policy PDFs

Place your PDF files in /content/policies/, then run:

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = PyPDFLoader("policies/national_policy_on_software_products-2019.pdf").load()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

5️⃣ Connect the ReAct agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq

llm = ChatGroq(model="gemma2-9b-it", temperature=0)

🔧 Guardian Policy Search Tool
from langchain_core.tools import tool

@tool
def policy_search_tool(query: str) -> str:
    """Search indexed company policy PDFs for relevant information."""
    if retriever is None:
        return "No retriever found. Please ensure documents were indexed."

    results = retriever.invoke(query)
    if not results:
        return f"No relevant policy found for: {query}"

    formatted = "\n\n".join([
        f"📄 Source: {r.metadata.get('source')}\n{r.page_content[:800]}"
        for r in results[:5]
    ])
    return formatted

🧩 Agent Construction
tools = [policy_search_tool]
agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

🧪 Example Usage
user_query = """
The company allows third-party vendors to process user data
without explicit consent. Check compliance with Indian data policy.
"""

response = agent_executor.invoke({"input": user_query})
print(response["output"])


Sample Output:

Violation Detected 🚨
According to Section 5 of the National Policy on Software Products (2019),
explicit user consent is required for data sharing with third parties.

→ Recommendation: Update vendor agreement clauses to include user-consent flow.

📊 Folder Structure
guardian-ai/
│
├── Guardian_AI_Colab_Notebook.ipynb
├── policies/
│   ├── national_policy_on_software_products-2019.pdf
│   └── <other_policy_docs>.pdf
├── README.md
└── .env

🧠 Future Improvements

🧩 Support for multi-document reasoning and citation ranking

🔒 Integration with company compliance dashboards

📜 Automatic policy-section extraction and summarization

🌐 Optional Gemini 2.0 Vision OCR for scanned policies
