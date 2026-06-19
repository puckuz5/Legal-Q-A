import streamlit as st
import tempfile
import os
from rag import extract_text, create_chunks, build_index, get_answer

# ──────────────────────────────────────────────
# PAGE CONFIG
# Must be the very first Streamlit command
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Legal Document Q&A",
    page_icon="⚖️",
    layout="centered"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# Makes the app look clean and professional
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; }
    .stAlert { border-radius: 10px; }
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.7;
    }
    .source-box {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        font-size: 13px;
        color: #555;
        margin-bottom: 8px;
    }
    .header-tag {
        background: #1f77b4;
        color: white;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("⚖️ Indian Legal Document Q&A")
st.markdown("Upload any Indian court judgment, contract, or FIR — then ask questions in plain English.")
st.divider()

# ──────────────────────────────────────────────
# SESSION STATE
# Streamlit reruns the entire script on every
# interaction. session_state lets us remember
# things across reruns — like the built index.
# Without this, it would re-process the PDF
# on every single question.
# ──────────────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = None       # FAISS index
if "chunks" not in st.session_state:
    st.session_state.chunks = None      # text chunks
if "filename" not in st.session_state:
    st.session_state.filename = None    # uploaded file name
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of Q&A pairs

# ──────────────────────────────────────────────
# STEP 1: FILE UPLOAD
# ──────────────────────────────────────────────
st.subheader("Step 1 — Upload your document")

uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["pdf"],
    help="Supports Indian court judgments, contracts, FIRs, and any legal PDF"
)

if uploaded_file is not None:
    # Only re-process if a NEW file is uploaded
    # If same file, use the cached index from session_state
    if uploaded_file.name != st.session_state.filename:

        with st.spinner("Reading and indexing your document... please wait"):
            try:
                # Save uploaded file to a temp location
                # Streamlit gives us a file-like object, not a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Run the full RAG pipeline from rag.py
                with open(tmp_path, "rb") as f:
                    text = extract_text(f)

                if not text.strip():
                    st.error("Could not extract text from this PDF. It may be a scanned image. Try a text-based PDF.")
                    st.stop()

                chunks = create_chunks(text)
                index, chunks = build_index(chunks)

                # Save to session state so it persists across questions
                st.session_state.index    = index
                st.session_state.chunks   = chunks
                st.session_state.filename = uploaded_file.name
                st.session_state.chat_history = []  # reset chat for new doc

                # Clean up temp file
                os.unlink(tmp_path)

                st.success(f"✅ Document processed — {len(chunks)} sections indexed. Ready for questions!")

            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.stop()
    else:
        st.success(f"✅ '{uploaded_file.name}' is loaded and ready.")

# ──────────────────────────────────────────────
# STEP 2: ASK QUESTIONS
# Only show this section after a doc is uploaded
# ──────────────────────────────────────────────
if st.session_state.index is not None:
    st.divider()
    st.subheader("Step 2 — Ask your question")

    # Example questions to help users get started
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Who are the parties?"):
            st.session_state.example_q = "Who are the parties involved in this case?"
    with col2:
        if st.button("What is the verdict?"):
            st.session_state.example_q = "What was the final verdict or order?"
    with col3:
        if st.button("What is the case about?"):
            st.session_state.example_q = "What is this case about? Give a brief summary."

    # Pre-fill input if example button was clicked
    default_q = st.session_state.get("example_q", "")

    question = st.text_input(
        "Type your question here",
        value=default_q,
        placeholder="e.g. What was the final order of the court?",
    )

    # Clear example after use
    if "example_q" in st.session_state:
        del st.session_state.example_q

    ask_clicked = st.button("Ask ⚖️", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Searching document and generating answer..."):
            try:
                answer, sources = get_answer(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )

                # Save to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error generating answer: {e}")

    # ──────────────────────────────────────────────
    # DISPLAY CHAT HISTORY
    # Show all previous Q&A pairs, newest first
    # ──────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Answers")

        # Reverse so newest answer appears at top
        for i, qa in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q: {qa['question']}**")

            # Answer in a styled box
            st.markdown(
                f'<div class="answer-box">{qa["answer"]}</div>',
                unsafe_allow_html=True
            )

            # Source excerpts in a collapsible section
            # Users can verify the answer themselves
            with st.expander(f"View source excerpts used ({len(qa['sources'])} found)"):
                for j, source in enumerate(qa["sources"]):
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Excerpt {j+1}</strong> (Section #{source["chunk_number"]})<br><br>'
                        f'{source["text"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            if i < len(st.session_state.chat_history) - 1:
                st.divider()

        # Button to clear chat history
        if st.button("Clear all answers"):
            st.session_state.chat_history = []
            st.rerun()

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built with RAG pipeline · PyPDF2 · FAISS · LLaMA 3 via Groq · Streamlit</small></center>",
    unsafe_allow_html=True
)
