import os
import re
import tempfile
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="EduGenie - Intelligent PDF Study Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Outfit', sans-serif; }
    .banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        padding: 2.5rem; border-radius: 16px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .banner h1 { font-size: 2.8rem; font-weight: 700; margin: 0; color: white !important; }
    .banner p { font-size: 1.15rem; opacity: 0.9; margin-top: 0.5rem; margin-bottom: 0; }
    .summary-card {
        background: linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(236,72,153,0.08) 100%);
        border: 1px solid rgba(124,58,237,0.3); border-radius: 12px;
        padding: 1.75rem; margin-top: 1rem;
    }
    .summary-card h3 { color: #7C3AED; font-size: 1.4rem; margin-top: 0; }
    .topic-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white; padding: 0.4rem 0.8rem; border-radius: 20px;
        font-size: 0.9rem; font-weight: 500; margin: 0.25rem;
    }
    .custom-card {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# GROQ CLIENT
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_groq_client():
    token = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    return Groq(api_key=token)

def generate_response(prompt, max_tokens=500):
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# =====================================================================
# EMBEDDINGS & PDF
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def process_pdf(uploaded_file, embedding_model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        os.remove(tmp_path)
        return vectorstore, len(docs), len(split_docs)
    except Exception as e:
        st.error(f"PDF processing failed: {str(e)}")
        return None, 0, 0

def get_context(vectorstore, query, k=5):
    if not vectorstore:
        return ""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def parse_mcqs(text):
    blocks = re.split(r'\bQ\d+[:\.]', text)
    mcqs = []
    for block in blocks[1:]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) < 5:
            continue
        question = lines[0]
        options = {}
        answer = None
        for line in lines[1:]:
            opt = re.match(r'^([A-D])[\)\.]\s*(.*)', line, re.IGNORECASE)
            ans = re.match(r'^(?:Answer|Ans)[:\.\s]*([A-D])', line, re.IGNORECASE)
            if opt:
                options[opt.group(1).upper()] = opt.group(2)
            elif ans:
                answer = ans.group(1).upper()
        if len(options) >= 4 and answer:
            mcqs.append({
                "question": question,
                "options": [options.get('A',''), options.get('B',''), options.get('C',''), options.get('D','')],
                "answer": answer
            })
    return mcqs[:5]

# =====================================================================
# SESSION STATE
# =====================================================================
defaults = {
    "vectorstore": None, "chat_history": [], "pdf_processed": False,
    "mcqs": [], "summary": "", "topics": [], "pdf_name": "",
    "explanation": "", "last_explained": "", "pages": 0, "chunks": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🧠 EduGenie</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>Your Smart Study Companion</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📥 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Loading embedding model..."):
            embeddings = load_embeddings()
        with st.spinner("Processing PDF..."):
            vs, pages, chunks = process_pdf(uploaded_file, embeddings)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.chat_history = []
                st.session_state.mcqs = []
                st.session_state.summary = ""
                st.session_state.topics = []
                st.session_state.explanation = ""
                st.rerun()

    if st.session_state.pdf_processed:
        st.success(f"✅ {st.session_state.pdf_name}")
        st.markdown(f"**Pages:** {st.session_state.pages}")
        st.markdown(f"**Chunks:** {st.session_state.chunks}")
        if st.button("Reset", type="secondary"):
            st.session_state.clear()
            st.rerun()

# =====================================================================
# MAIN
# =====================================================================
st.markdown("""
<div class="banner">
    <h1>🧠 EduGenie</h1>
    <p>Upload a PDF to chat, summarize, quiz, and study smarter.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="custom-card" style="text-align:center;padding:3rem;">
        <h3>👋 Welcome! Upload a PDF in the sidebar to begin.</h3>
        <div style="display:flex;justify-content:center;gap:2rem;margin-top:2rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:130px;padding:1rem;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
                <h4>💬 Chat</h4><p style="color:#888;font-size:0.9rem;">Ask questions from your PDF</p>
            </div>
            <div style="flex:1;min-width:130px;padding:1rem;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
                <h4>📄 Summary</h4><p style="color:#888;font-size:0.9rem;">Bullet point summaries</p>
            </div>
            <div style="flex:1;min-width:130px;padding:1rem;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
                <h4>📝 MCQ Quiz</h4><p style="color:#888;font-size:0.9rem;">Auto-generated quizzes</p>
            </div>
            <div style="flex:1;min-width:130px;padding:1rem;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
                <h4>💡 Study Guide</h4><p style="color:#888;font-size:0.9rem;">Simple explanations</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    tab_chat, tab_summary, tab_mcq, tab_study = st.tabs([
        "💬 Chat", "📄 Summary", "📝 MCQ Quiz", "💡 Study Guide"
    ])

    # CHAT TAB
    with tab_chat:
        st.markdown("### 💬 Chat with your PDF")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask something about the document..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = get_context(st.session_state.vectorstore, prompt)
                    full_prompt = f"""You are EduGenie, a helpful study assistant.
Answer ONLY using the context below. If not found, say "I could not find this in the document."

Context:
{context}

Question: {prompt}"""
                    response = generate_response(full_prompt, 300)
                    st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # SUMMARY TAB
    with tab_summary:
        st.markdown("### 📄 Smart Summary")
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarizing..."):
                context = get_context(st.session_state.vectorstore, "main points key ideas overview", k=10)
                full_prompt = f"""Summarize the text below in 6-8 clear bullet points.
Start directly with bullet points, no intro.

Text:
{context}"""
                st.session_state.summary = generate_response(full_prompt, 400)
                st.rerun()
        if st.session_state.summary:
            formatted = st.session_state.summary.replace("\n", "<br>")
            st.markdown(f'<div class="summary-card"><h3>📋 Summary</h3><div style="line-height:1.6;">{formatted}</div></div>', unsafe_allow_html=True)

    # MCQ TAB
    with tab_mcq:
        st.markdown("### 📝 MCQ Quiz")
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Creating quiz..."):
                context = get_context(st.session_state.vectorstore, "key concepts facts definitions", k=10)
                full_prompt = f"""Generate exactly 5 multiple choice questions from the text below.
Use this exact format:

Q1: [Question here]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: A

Q2: [Question here]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: B

Text:
{context}"""
                raw = generate_response(full_prompt, 700)
                parsed = parse_mcqs(raw)
                st.session_state.mcqs = parsed if parsed else [{"fallback": raw}]
                st.rerun()

        if st.session_state.mcqs:
            if "fallback" in st.session_state.mcqs[0]:
                st.warning("Could not parse quiz. Raw output:")
                st.text(st.session_state.mcqs[0]["fallback"])
            else:
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, mcq in enumerate(st.session_state.mcqs):
                        st.markdown(f"**Q{i+1}. {mcq['question']}**")
                        user_answers[i] = st.radio(
                            "Answer:",
                            [f"A) {mcq['options'][0]}", f"B) {mcq['options'][1]}",
                             f"C) {mcq['options'][2]}", f"D) {mcq['options'][3]}"],
                            key=f"q{i}", index=None
                        )
                        st.markdown("---")
                    if st.form_submit_button("✅ Submit"):
                        score = 0
                        for i, mcq in enumerate(st.session_state.mcqs):
                            sel = user_answers[i]
                            correct = mcq['answer']
                            correct_text = f"{correct}) {mcq['options'][ord(correct)-65]}"
                            if sel:
                                if sel[0] == correct:
                                    score += 1
                                    st.success(f"Q{i+1}: ✅ Correct!")
                                else:
                                    st.error(f"Q{i+1}: ❌ Wrong. Correct: {correct_text}")
                            else:
                                st.warning(f"Q{i+1}: Unanswered. Correct: {correct_text}")
                        total = len(st.session_state.mcqs)
                        st.metric("Score", f"{score}/{total}", f"{int(score/total*100)}%")

    # STUDY GUIDE TAB
    with tab_study:
        st.markdown("### 💡 Study Guide")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔍 Key Topics")
            if st.button("Extract Topics"):
                with st.spinner("Extracting..."):
                    context = get_context(st.session_state.vectorstore, "main topics concepts", k=8)
                    full_prompt = f"""List exactly 5 key topics from the text below.
One topic per line only. No numbers, no bullets, no extra text.

Text:
{context}"""
                    raw = generate_response(full_prompt, 150)
                    topics = [t.strip().lstrip('-*•12345. ') for t in raw.split('\n') if t.strip()]
                    st.session_state.topics = [t for t in topics if t][:5]
                    st.rerun()
            if st.session_state.topics:
                for t in st.session_state.topics:
                    st.markdown(f"<span class='topic-badge'>{t}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 💡 Explain Simply")
            explain_input = ""
            if st.session_state.topics:
                sel = st.selectbox("Pick a topic:", ["Custom"] + st.session_state.topics)
                if sel != "Custom":
                    explain_input = sel
            custom = st.text_input("Or type a topic:")
            final = custom if custom else explain_input
            if st.button("Explain") and final:
                with st.spinner("Explaining..."):
                    context = get_context(st.session_state.vectorstore, final, k=5)
                    full_prompt = f"""Explain '{final}' in very simple, beginner-friendly language using the context below. Use analogies.

Context:
{context}"""
                    st.session_state.explanation = generate_response(full_prompt, 300)
                    st.session_state.last_explained = final
                    st.rerun()
            if st.session_state.explanation:
                st.markdown(f'<div class="summary-card"><h3>💡 {st.session_state.last_explained}</h3><p style="line-height:1.6;">{st.session_state.explanation}</p></div>', unsafe_allow_html=True)
