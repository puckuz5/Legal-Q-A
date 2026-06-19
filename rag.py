import os
import faiss
import numpy as np
import PyPDF2

from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your API key from the .env file
load_dotenv()

# ──────────────────────────────────────────────
# STEP 1: Load the free embedding model
# This runs locally on your computer — no API cost
# Downloads ~90MB the first time, then cached forever
# ──────────────────────────────────────────────
print("Loading embedding model... (slow only first time)")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")


# ──────────────────────────────────────────────
# FUNCTION 1: Extract text from a PDF file
# Input  → a PDF file object (from Streamlit uploader)
# Output → one big string of all the text in the PDF
# ──────────────────────────────────────────────
def extract_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)

    # Loop through every page and grab its text
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_text = page.extract_text()

        # Some pages have no extractable text (scanned images)
        # We skip those quietly
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page_text

    return text


# ──────────────────────────────────────────────
# FUNCTION 2: Split text into overlapping chunks
# Why chunks? Because we can't send a 100-page
# document to the LLM — it's too many tokens.
# We split it into small pieces and only send
# the 3 most relevant pieces per question.
#
# chunk_size=500  → each chunk is ~500 characters
# chunk_overlap=50 → 50 characters shared between
#                    chunks so context isn't lost
#                    at the boundary
# ──────────────────────────────────────────────
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Document split into {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────
# FUNCTION 3: Build a FAISS vector index
# This converts each chunk into a vector (a list
# of 384 numbers that capture its meaning).
# FAISS stores all vectors so we can search them
# instantly by meaning — not just keyword match.
#
# Input  → list of text chunks
# Output → (faiss_index, chunks_list)
#           We return chunks_list too because FAISS
#           only stores vectors, not the original text.
#           We need chunks_list to retrieve text later.
# ──────────────────────────────────────────────
def build_index(chunks):
    print("Building vector index...")

    # Convert all chunks to vectors at once
    # Shape: (number_of_chunks, 384)
    embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)

    # Convert to float32 — FAISS requires this format
    embeddings = np.array(embeddings).astype("float32")

    # Get the size of each vector (384 for this model)
    dimension = embeddings.shape[1]

    # Create a flat L2 index — best for small documents
    # L2 = Euclidean distance (smaller = more similar)
    index = faiss.IndexFlatL2(dimension)

    # Add all vectors to the index
    index.add(embeddings)

    print(f"Index built with {index.ntotal} vectors")
    return index, chunks


# ──────────────────────────────────────────────
# FUNCTION 4: Search the index for relevant chunks
# Input  → question (string), faiss index, chunks list
# Output → list of the top 3 most relevant chunks
# ──────────────────────────────────────────────
def search_chunks(question, index, chunks, top_k=3):
    # Convert the question to a vector using the same model
    question_vector = EMBEDDING_MODEL.encode([question])
    question_vector = np.array(question_vector).astype("float32")

    # Search the index — returns distances and positions
    # distances: how similar each result is (lower = better)
    # positions: which chunk numbers were found
    distances, positions = index.search(question_vector, top_k)

    # Retrieve the actual text of the top matching chunks
    relevant_chunks = []
    for i, pos in enumerate(positions[0]):
        if pos != -1:  # -1 means no result found
            relevant_chunks.append({
                "text": chunks[pos],
                "score": float(distances[0][i]),
                "chunk_number": int(pos)
            })

    return relevant_chunks


# ──────────────────────────────────────────────
# FUNCTION 5: Get answer from Groq (FREE)
# Groq runs LLaMA 3 for free with very fast speeds
# We send it:
#   1. A system message defining its role
#   2. The 3 relevant chunks as context
#   3. The user's question
# It answers ONLY from the document chunks.
# ──────────────────────────────────────────────
def get_answer(question, index, chunks):
    # First find the relevant chunks
    relevant_chunks = search_chunks(question, index, chunks)

    if not relevant_chunks:
        return "Could not find relevant sections in the document.", []

    # Build context string from the retrieved chunks
    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"\n--- Excerpt {i+1} ---\n"
        context += chunk["text"]
        context += "\n"

    # System message — tells Groq its role and rules
    system_message = """You are a legal document assistant helping users understand Indian legal documents — court judgments, contracts, FIRs, and other legal texts.

You will be given relevant excerpts from a legal document. You must:
1. Answer ONLY using information from the provided excerpts
2. Never use your general knowledge about law
3. Never make up information
4. If the answer is not in the excerpts, say: "This information is not clearly stated in the provided document sections."
5. Keep answers clear and easy to understand for non-lawyers"""

    # User message — the context + question
    user_message = f"""DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:"""

    # Call Groq API — free, fast, no credit card needed
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free model on Groq
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message}
        ],
        max_tokens=1024,
        temperature=0.1  # Low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content
    return answer, relevant_chunks


# ──────────────────────────────────────────────
# QUICK TEST — run this file directly to check
# everything is working before building the UI
#
# Usage: python rag.py
# It will ask you for a PDF path and a question
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Legal QA System — Quick Test ===\n")

    pdf_path = input("Enter path to a PDF file (e.g. test.pdf): ").strip()

    try:
        with open(pdf_path, "rb") as f:
            print("\nExtracting text...")
            text = extract_text(f)
            print(f"Extracted {len(text)} characters of text")

            print("\nCreating chunks...")
            chunks = create_chunks(text)

            print("\nBuilding index...")
            index, chunks = build_index(chunks)

            print("\nReady! Type your questions below.")
            print("Type 'quit' to exit\n")

            while True:
                question = input("Your question: ").strip()
                if question.lower() == "quit":
                    break
                if not question:
                    continue

                print("\nSearching document and generating answer...\n")
                answer, sources = get_answer(question, index, chunks)

                print("ANSWER:")
                print(answer)
                print("\nSOURCE EXCERPTS USED:")
                for i, s in enumerate(sources):
                    print(f"\nExcerpt {i+1} (chunk #{s['chunk_number']}):")
                    print(s["text"][:200] + "...")
                print("\n" + "="*50 + "\n")

    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
    except Exception as e:
        print(f"Error: {e}")
