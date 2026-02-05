GraphScribe ⭐

Research-aware technical blog generation using LangGraph

GraphScribe is an agentic technical blog writer built with LangGraph, LangChain, and Streamlit.

It plans, researches, and writes publish-ready technical blogs end-to-end using a structured multi-agent pipeline.

Unlike simple LLM wrappers, GraphScribe separates decision-making, planning, research, and writing into explicit graph nodes—making the system transparent, extensible, and production-oriented.



✨ Key Features

🧭 Intelligent Routing (Router Node)

Automatically decides whether web research is required

Classifies topics into:

Closed-book (evergreen fundamentals)

Hybrid (mostly evergreen + up-to-date examples)

Open-book (volatile, time-sensitive topics)

Generates high-signal search queries only when needed


🔍 Research & Evidence Synthesis

Integrates Tavily Search for live web results

Deduplicates and normalizes sources into structured evidence

Prioritizes authoritative sources (docs, company blogs, reputable outlets)

Preserves publication dates when available

Enforces URL-only citations (no hallucinated sources)


🧠 Orchestrator-Driven Blog Planning

Generates a strictly validated blog plan using Pydantic schemas

Produces 5–7 structured sections, each with:

Clear learning goal

3–5 concrete, non-overlapping bullets

Target word count

Section type (intro, core, examples, common mistakes, checklist, conclusion)


Guarantees:

Exactly one “Common Mistakes” section

At least one section covering:

Code examples

Edge cases / failure modes

Performance or cost trade-offs

Debugging / observability



⚙️ Fan-Out Worker Architecture

Each section is written by an independent worker agent

Workers:

Follow the plan strictly

Stay within word limits

Include code snippets, examples, or checklists when relevant

Explain why best practices matter

Enables parallel section generation (scales cleanly)



🧩 Deterministic State Management (LangGraph)

Explicit state transitions:

Router → Research → Orchestrator → Workers → Reducer


Typed global state ensures:

Predictable execution

Easy debugging

Safe extensibility



📝 Clean Markdown Output

Final reducer:

Combines all sections

Generates a safe filename

Writes a publish-ready Markdown file

Perfect for:

Blogs

Documentation

Technical tutorials

Developer advocacy content



🎨 Streamlit Frontend

Minimal, professional UI

Sidebar-driven topic input

Tabs for:

📝 Final Blog

🧠 Blog Plan

🔍 Research & Evidence

📜 Execution Logs

One-click Markdown download

🏗️ Architecture Overview
User Topic

   ↓
   
Router (research decision)

   ↓
   
Research (optional)

   ↓
   
Orchestrator (blog plan)

   ↓
   
Fan-out Workers (sections)

   ↓
   
Reducer (final markdown)


This architecture mirrors real production agent systems, not toy pipelines.


🛠️ Tech Stack

LangGraph – Agent orchestration & state machine

LangChain – Structured prompting & tools

Ollama (Qwen 2.5) – Local LLM inference

Tavily Search API – Live research

Pydantic – Schema enforcement

Streamlit – Frontend UI

🚀 Getting Started

1. Clone the repository
2. 
git clone https://github.com/Satyam-Singh-x/GraphScribe-Technical-Blog-Writer
.git

cd GraphScribe-Technical-Blog-Writer

4. Install dependencies
5. 
pip install -r requirements.txt

6. Set environment variables
   
TAVILY_API_KEY=your_api_key_here


8. Run Ollama
   
ollama run qwen2.5

10. Launch the app
streamlit run frontend.py


🎯 Use Cases

Technical blogging at scale

Developer documentation

AI / ML explainers

System design articles

Research-backed tutorials

Developer advocacy content




🔮 Future Improvements

Section-level citation rendering

Pluggable LLM backends

Async / distributed workers

SEO optimization passes

PDF / HTML export

Versioned blog regeneration

📄 License

MIT License

By Satyam
