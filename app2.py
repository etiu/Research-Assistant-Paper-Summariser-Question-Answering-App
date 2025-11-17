import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("RAG powered Research Assistant")

# ---------- LOAD PDFs + Create Chunks ----------
@st.cache_data
def load_papers(files):
    all_chunks = []
    chunk_locations = []  # NEW: store metadata

    for file in files:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    words = text.split()

                    # chunk at 500 words steps
                    for i in range(0, len(words), 500):
                        chunk = " ".join(words[i:i+500])

                        # store raw chunk
                        all_chunks.append(chunk)

                        # store (filename, page number)
                        chunk_locations.append((file.name, page_num + 1))

    # Build embeddings once
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks, convert_to_tensor=True)

    return all_chunks, embeddings, model, chunk_locations



# ---------- UI ----------
files = st.file_uploader("Upload one or more research papers", type="pdf", accept_multiple_files=True)

if files:

    all_chunks, embeddings, model, chunk_locations = load_papers(files)

    # Initialize chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show conversation history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # User Input
    query = st.chat_input("Ask something about the uploaded papers...")

    if query:

        # ---------- Semantic Search ----------
        query_emb = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, embeddings, top_k=5)

        # Build context with top chunks
        retrieved_text = "\n\n".join([f"[Source {h['corpus_id']}]: {all_chunks[h['corpus_id']]}" for h in hits[0]])

        # Conversation history to maintain chatbot memory
        conversation_context = "\n".join(
            f"User: {m['content']}" if m["role"]=="user" else f"Assistant: {m['content']}"
            for m in st.session_state.chat_history
        )

        # ---------- RAG Prompt ----------
        prompt = f"""
You are a helpful research assistant. 
You MUST cite relevant text chunks using format (Source X) where X is the chunk number. 
Do not invent citations. If the answer cannot be supported by the sources, say: "Not in the uploaded papers."


Conversation History:
{conversation_context}

Relevant Extracts:
{retrieved_text}

Question: {query}

Answer clearly and cite source numbers like (Source 2).
"""

        # ---------- LLM Response ----------
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()

        # Display new messages
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)

        # ---------- Extract citations used in the model answer ----------
        import re

        # Find all "Source X" patterns in the LLM response
        source_ids = set(re.findall(r"Source (\d+)", answer))

        citation_details = []
        for sid in source_ids:
            sid = int(sid)
            filename, page = chunk_locations[sid]
            chunk_text = all_chunks[sid]

            # Make excerpt short
            short_excerpt = chunk_text[:300].replace("\n", " ") + "..."

            citation_details.append(f"Source {sid} - {filename}, page {page}\n> {short_excerpt}")

        # Display citations under the answer
        if citation_details:
            with st.expander("View referenced sources"):
                for c in citation_details:
                    st.write(c)
        # Store in session state for memory
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
        # ---------- Download Conversation ----------
        if st.session_state.chat_history:
            chat_text = ""
            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_text += f"{role}: {msg['content']}\n\n"

            st.download_button(
                label="Download Conversation",
                data=chat_text,
                file_name="conversation.txt",
                mime="text/plain"
            )

        
        
