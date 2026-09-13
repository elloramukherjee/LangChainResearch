"""
Agentic AI Research Assistant
RAG + ReAct Agent in a single LangChain workflow.
The agent decides on its own whether to search the PDF, the web, or both.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

# ========================================
# CONCEPT 1: ChatOpenAI (Chat Model)
# ========================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ========================================
# CONCEPT 2: PyPDFLoader (Document Loader)
# ========================================
pdf_path = input("Enter PDF path: ").strip()
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ========================================
# CONCEPT 3: RecursiveCharacterTextSplitter
# ========================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"✅ PDF loaded: {len(documents)} pages, {len(chunks)} chunks")

# ========================================
# CONCEPT 4: OpenAIEmbeddings
# ========================================
embeddings = OpenAIEmbeddings()

# ========================================
# CONCEPT 5: FAISS (Vector Store)
# ========================================
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Vector store ready (FAISS)")

# ========================================
# CONCEPT 6: as_retriever()
# ========================================
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ========================================
# CONCEPT 7: create_retriever_tool (RAG as an agent tool)
# ========================================
pdf_tool = create_retriever_tool(
    retriever,
    name="pdf_knowledge_base",
    description=(
        "Search the uploaded PDF document for information. "
        "Use this when the question is about the document's content."
    ),
)

# ========================================
# CONCEPT 8: TavilySearchResults (Web Search Tool)
# ========================================
web_tool = TavilySearchResults(max_results=3, name="web_search")

# ========================================
# CONCEPT 9: @tool decorator (custom tool)
# ========================================
@tool
def save_answer(answer: str) -> str:
    """Save the final answer text to answers.txt. Pass the answer as a string."""
    with open("answers.txt", "a", encoding="utf-8") as f:
        f.write(answer + "\n---\n")
    return "Answer saved to answers.txt"


tools = [pdf_tool, web_tool, save_answer]

# ========================================
# CONCEPT 10: hub.pull() — Load ReAct prompt
# ========================================
prompt = hub.pull("hwchase17/react")

# ========================================
# CONCEPT 11: create_react_agent
# ========================================
agent = create_react_agent(llm, tools, prompt)

# ========================================
# CONCEPT 12: AgentExecutor (Think -> Act -> Observe loop)
# ========================================
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
)

print(
    f"✅ Agent ready with {len(tools)} tools: "
    "[pdf_knowledge_base, web_search, save_answer]\n"
)

# ========================================
# Interactive question loop
# ========================================
print("Ask anything (type 'quit' to exit):\n")
while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("👋 Goodbye!")
        break
    if not question:
        continue
    result = agent_executor.invoke({"input": question})
    print(f"\n✅ Answer: {result['output']}\n")
