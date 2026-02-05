from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated, Literal,Optional

from langchain_community.tools import TavilySearchResults
from ollama import ListResponse
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()


llm = ChatOllama(
    model="qwen2.5:latest",
    base_url="http://localhost:11434",
    temperature=0
)

#--------------SCHEMAS-----------------------------
class Task(BaseModel):
    id: int
    title: str = Field(..., description="Suitable Task title")
    goal: str = Field(...,
                      description="One sentence describing what the reader should be able to understand/do after this section.")
    requires_research: bool
    requires_citations: bool
    requires_code:bool

    bullets: List[str] = Field(...,
                               min_length=3,
                               max_length=5,
                               description="3-5 concrete non overlapping subpoints to cover in this section")
    target_words: int = Field(..., description="Target word count for this section (120-450)")

    section_type: Literal["Intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(...,
                                                                                                             description="Use 'common_mistakes' exactly once in the plan")


class Plan(BaseModel):
    blog_title: str = Field(..., description="Suitable Blog title")
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical,crisp)")
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    URL:str
    published_at:Optional[str]=None
    snippet: Optional[str]=None
    source: Optional[str]=None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book","hybrid","open_book"]
    queries:List[str]=Field(default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem]=Field(default_factory=list)


class State(TypedDict):
    topic: str
    #routing/researcher
    mode:str
    needs_research:bool
    queries:List[str]
    evidence:List[EvidenceItem]
    plan: Plan

    #workers
    sections: Annotated[List[Plan], operator.add]
    final: str


#ROUTER NODE:
ROUTER_SYSTEM="""
You are a routing module for a technical blog planned.
Decide whether web research is needed BEFORE planning.

Modes:
 -Closed book (needs_research=false)
  Evergreen topics where correctness does not depend on recent facts (concepts,  fundamentals).
-hybrid (needs_research=true)
  Mostly evergreen but needs up-to-date examples/tools/models to be useful
- open_book (needs_research=true)
  Mostly volatile: weekly roundups,"this week","latest",rankings,pricing,policy/regulation.
  
If needs_research=True:
 -Output 3-10 high signal queries.
 - Queries should not be scoped and specific (avoid generic queries like just "AI" or "LLM")
 - If user asked for "last week/this week/latest", reflect that constraint in the QUERIES.
"""

def router_node(state:State) -> dict:
    topic=state["topic"]
    decider= llm.with_structured_output(RouterDecision)
    decision=decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"topic: {topic}"),
        ]


    )
    return {"needs_research": decision.needs_research, "mode": decision.mode,"queries": decision.queries}

def route_next(state:State)->str:
    return "research" if state["needs_research"] else "orchestrator"

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results:
        normalized.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "source": r.get("source", ""),
            "published_at": r.get("published_date"),
        })

    return normalized



RESEARCH_SYSTEM="""
You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
 - Only include items with a non-empty url.
 - prefer relevant + authoritative sources (company blogs,docs,reputable outlets).
 - If a published date is explicitly mentioned in the result payload, keep it as YYYY-MM-DD.
   If missing or unclear, set published_at=null. Do not guess.
-Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state:State)->dict:
    queries=(state.get("queries",[]) or [])
    max_results=6
    raw_results:List[dict]=[]
    for q in queries:
        raw_results.extend(_tavily_search(q,max_results=max_results))
    if not raw_results:
        return {"evidence": []}

    extractor= llm.with_structured_output(EvidencePack)
    pack =extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw Results: {raw_results}"),
    ]
    )
    dedup={}
    for e in pack.evidence:
        if e.URL:
            dedup[e.URL]=e

    return {"evidence": list(dedup.values())}












def orchestrator(state: State) -> dict:
    mode= state.get("mode","closed_book")
    evidence=state.get("evidence",[])
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=("""
                -You are a senior technical writer and developer advocate Your job is to produce a 
                highly actionable outline for a technical blog post.
                **Hard Requirements:
                -Create 5-7 sections (tasks) that fit a technical blog.
                -Each section(task) must include:
                   1) goal (1 sentencee: what the reader can do/understand after this section)
                   2) 3-5 bullets that are concrete specific , non overlapping subpoints to cover in this section.
                   3) target word count (120-450)
                -Include exactly one section with section_type='common_mistakes'.
                -Make it technical(not generic)
                -Assume the reader is a developer , use correct terminology.
                -Prefer design/engineering structures: problem -> Intuition -> approach -> Implementation ->
                trade-offs -> testing/observability -> conclusions.
                --Bullets must be actionable and testable (e.g. 'Show a minimal code snippet for X','Explain why Y fails under Z condition','Add a checklist for production readiness')
                -Explicitly include at least ONE of the following somewhere in the plan (as bullets):
                * A minimal working example or code sketch
                * edge cases / failure modes
                * performance/cost considerations 
                * security/privacy considerations (if relevant) 
                * Debugging tips / observability(logs, metrics, traces)
                - Avoid vague bullets like "Explain X" or "Discuss Y". Every bullet should state what to build/compare/measure/verify.
                - Ordering guidance:
                 -Start with a crisp intro and problem facing.
                 - Build core concepts before advanced details.
                 - Include one section common mistakes and how to avoid them.
                 - End with a practical summary/checklist and next steps.
                 Output must strictly match the plan Schema.





                """)

            ), HumanMessage(
            content=(f'Topic: {state["topic"]}\n'
                     f'Mode: {mode}\n'
                     f'Evidence: only use for freshn claims; may be empty'
                     f'{[e.model_dump() for e in evidence][:16]})'


                     ))

        ]
    )
    return {"plan": plan}


def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"],"mode":state["mode"], "plan": state["plan"],"evidence":[e.model_dump() for e in state.get("evidence",[])]}) for task in
            state["plan"].tasks]


def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    blog_title = plan.blog_title
    evidence=[EvidenceItem(**e) for e in payload.get("evidence",[])]

    bullets_text = "\n-" + "\n-".join(task.bullets)

    evidence_text=""
    if evidence:
        evidence_text += "\n".join(
            f"- {e.title} | {e.URL}  | {e.published_at or "date: unknown"}".strip() for e in evidence[:20]
        )

    section_msg = llm.invoke(
        [
            SystemMessage(content="""
            You are a senior technical writer and developer advocate. Write one section of a technical blog post in MARKDOWN.
            **HARD CONSTRAINTS:
            -Follow the provided goal and cover ALL Bullets in order (do not skip or merge bullets.)
            -Stay Close to the target words.
            - Output only the section content in Markdown (no blog title H1,no extra commentary).
            ** TECHNICAL QUALITY BAR
            -Be precise and implementation-oriented (developers should be able to apply it.)
            - Prefer concrete details over abstractions: APIs, data structures, protocols , and exact terms.
            -When relevant , include at least one of:
              * a small code snipper(minimal, correct and idiomatic)
              * a tiny example input/output
              * a checklist of steps.
              * a diagram described in text(e.g. flow: a-> b-> c)
            -Explain trade-offs briefly (performance, cost , complexity, reliability)
            -Call out edge cases / failure modes and what to do about them.
            - If you mention a best practice, add a 'why' in one sentence.
            *MARKDOWN STYLE:*
            - Start with a '## <Section Title>' heading.
            -use short paragraphs, bullet lists where helpful and code fences for code.
            - Avoid fluff. Avoid marketing language.
            -If you include code, keep it focused on bullet being addressed.

            """),
            HumanMessage(
                content=(
                    f"Blog title: {blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Topic: {topic}\n"
                    f"Section: {task.title}\n"
                    f"Section type: {task.section_type}\n"
                    f"Goal: {task.goal}\n"
                    f"target words:{task.target_words}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets: {bullets_text}\n"
                    f"Evidence (ONLY use these URLS when citing):{evidence_text}\n"
                    "Return only the section content in markdown."
                )
            )
        ]
    )

    section_md = section_msg.content.strip()

    return {"sections": [section_md]}


from pathlib import Path


from pathlib import Path
import re

def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"

    # SAFE filename generation
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    slug = slug[:80]  # prevent Windows path errors

    path = Path(f"{slug}.md")
    path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}


g = StateGraph(State)
g.add_node("router",router_node)
g.add_node("research",research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "router")
g.add_conditional_edges("router",route_next,{"research":"research","orchestrator":"orchestrator"})
g.add_edge("research", "orchestrator")


g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

print(app.get_graph().draw_mermaid())


