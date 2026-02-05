import streamlit as st
from pathlib import Path
import re
import json

# Import your LangGraph app
from research_blog_writer import app


# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="GraphScribe",
    page_icon="⭐",
    layout="wide",
)


# =====================================================
# Sidebar (Branding + Input)
# =====================================================
st.sidebar.markdown(
    """
    <h1 style="margin-bottom:0.2rem;">GraphScribe ⭐</h1>
    <p style="color:#6b7280; margin-top:0;">
    Plan smarter. Write better.
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

topic = st.sidebar.text_input(
    "Blog topic",
    placeholder="e.g. State of multimodal LLMs in 2026",
)

generate_btn = st.sidebar.button(
    "Generate Blog",
    type="primary",
    use_container_width=True,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    """
    **What GraphScribe does**
    - Plans a technical blog with structure
    - Performs research when needed
    - Writes each section via agents
    - Outputs clean, publish-ready Markdown
    """
)


# =====================================================
# Session state
# =====================================================
if "result" not in st.session_state:
    st.session_state.result = None


# =====================================================
# Run graph
# =====================================================
if generate_btn:
    if not topic.strip():
        st.sidebar.warning("Please enter a blog topic.")
    else:
        with st.spinner("GraphScribe is generating your blog…"):
            result = app.invoke({"topic": topic})
            st.session_state.result = result


# =====================================================
# Main content
# =====================================================
st.markdown(
    """
    <h2 style="margin-bottom:0.2rem;">📘 Generated Technical Blog</h2>
    <p style="color:#6b7280; margin-top:0;">
    Auto-generated using a research-aware LangGraph pipeline
    </p>
    """,
    unsafe_allow_html=True,
)

if st.session_state.result is None:
    st.info("Enter a topic in the sidebar and click **Generate Blog**.")
    st.stop()

result = st.session_state.result
plan = result["plan"]
final_md = result["final"]


# =====================================================
# Tabs
# =====================================================
tab_blog, tab_plan, tab_research, tab_logs = st.tabs(
    [
        "📝 Blog",
        "🧠 Plan",
        "🔍 Research",
        "📜 Execution Logs",
    ]
)


# =====================================================
# Blog tab
# =====================================================
with tab_blog:
    st.markdown(final_md)

    # Safe filename
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", plan.blog_title).strip("_").lower()[:80]

    st.download_button(
        label="⬇️ Download Markdown",
        data=final_md,
        file_name=f"{slug}.md",
        mime="text/markdown",
    )


# =====================================================
# Plan tab
# =====================================================
with tab_plan:
    st.subheader("Blog Plan")

    col1, col2, col3 = st.columns(3)
    col1.metric("Audience", plan.audience)
    col2.metric("Tone", plan.tone)
    col3.metric("Sections", len(plan.tasks))

    st.markdown("---")

    for idx, task in enumerate(plan.tasks, start=1):
        with st.expander(f"Section {idx}: {task.title}", expanded=False):
            st.markdown(f"**Goal**  \n{task.goal}")
            st.markdown(f"**Section type:** `{task.section_type}`")
            st.markdown(f"**Target words:** `{task.target_words}`")

            st.markdown("**Coverage bullets:**")
            for bullet in task.bullets:
                st.markdown(f"- {bullet}")


# =====================================================
# Research tab
# =====================================================
with tab_research:
    st.subheader("Research & Evidence")

    st.markdown(
        f"""
        **Mode:** `{result.get("mode")}`  
        **Needs research:** `{result.get("needs_research")}`
        """
    )

    st.markdown("### Generated Queries")
    queries = result.get("queries", [])
    if queries:
        for q in queries:
            st.code(q, language="text")
    else:
        st.caption("No research queries were generated.")

    st.markdown("### Evidence Used")
    evidence = result.get("evidence", [])
    if evidence:
        for e in evidence:
            st.markdown(
                f"""
                **[{e.title}]({e.URL})**  
                Source: `{e.source or 'unknown'}` · Date: `{e.published_at or 'unknown'}`
                """
            )
    else:
        st.caption("No external sources were required.")


# =====================================================
# Logs tab
# =====================================================
with tab_logs:
    st.subheader("Execution Trace")

    st.caption("Raw structured output from the LangGraph pipeline.")

    st.json(
        {
            "mode": result.get("mode"),
            "needs_research": result.get("needs_research"),
            "queries": result.get("queries"),
            "sections_generated": len(result.get("sections", [])),
        }
    )
