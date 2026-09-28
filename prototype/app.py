from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import load_rag_components, run_rag


st.set_page_config(
    page_title="Research Attention RAG",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def get_rag_components():
    # Load DPR model, FAISS index, and chunks only once
    return load_rag_components()


def display_sources(sources):
    st.subheader("Supporting Sources")

    for source in sources:
        with st.expander(
            f"[Source {source['source_number']}] "
            f"{source['paper_id']} — Page {source['page']}"
        ):
            st.write(f"**Title:** {source['title']}")
            st.write(f"**Chunk ID:** {source['chunk_id']}")
            st.write(f"**Page:** {source['page']}")
            st.write(f"**DOI:** {source['doi']}")

            if source.get("score") is not None:
                st.write(
                    f"**Retrieval Score:** "
                    f"{source['score']:.4f}"
                )


def main():
    st.title("Research Evidence Assistant")

    st.write(
        "Ask a question about attention, alertness, "
        "working memory, or cognitive performance."
    )

    st.caption(
        "Answers are generated only from retrieved "
        "research evidence in the project knowledge base."
    )

    # Load the RAG system once per Streamlit session.
    with st.spinner("Loading RAG system..."):
        components = get_rag_components()

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: Does a small amount of caffeine "
            "improve sustained attention?"
        ),
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner(
                "Retrieving research evidence and "
                "generating answer..."
            ):
                result = run_rag(
                    question,
                    components=components,
                )

            st.divider()

            st.subheader("Answer")
            st.markdown(result["answer"])

            st.divider()

            display_sources(
                result["sources"]
            )

        except Exception as error:
            st.error(
                f"An error occurred: {error}"
            )


if __name__ == "__main__":
    main()