# Agentic AI Research Assistant

A single ReAct agent that combines **RAG** (search your PDF) with
**live web search**, deciding on its own which tool(s) to use.

## Setup

> **Windows users:** if `pip install` or running the script fails with a
> `tiktoken` / `_tiktoken.pyd` load error, it's Windows **Smart App Control**
> blocking the native module (Code Integrity event 3077), not a bug here. Run
> this project inside **WSL** instead — see below.

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows (native, no Smart App Control): .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY and TAVILY_API_KEY
python main.py
```

Requirements are pinned to the pre-1.0 LangChain line (`langchain>=0.3,<0.4`)
so the classic `create_react_agent` / `AgentExecutor` / `hub.pull` imports
used in `main.py` work as-is. LangChain 1.x moved these into a separate
`langchain_classic` package.

## Sample Interaction

```
$ python main.py

Enter PDF path: company_report.pdf
✅ PDF loaded: 24 pages, 47 chunks
✅ Vector store ready (FAISS)
✅ Agent ready with 3 tools: [pdf_knowledge_base, web_search, save_answer]

Ask anything (type 'quit' to exit):

You: What does the document say about revenue growth?
🤔 Thinking: This is about the document → use pdf_knowledge_base
⚡ Action: pdf_knowledge_base("revenue growth")
👁 Observation: "Revenue grew 34% YoY to $12.4M..."
✅ Answer: According to the document, revenue grew 34%
   year-over-year reaching $12.4M...

You: How does that compare to industry average?
🤔 Thinking: I need current industry data → use web_search
⚡ Action: web_search("SaaS industry average revenue growth 2025")
👁 Observation: "Average SaaS growth rate is 20-25%..."
✅ Answer: Your company's 34% growth significantly outperforms
   the industry average of 20-25%...

You: quit
👋 Goodbye!
```
