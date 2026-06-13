import os
import re
import tempfile
import streamlit as st
from huggingface_hub import InferenceClient

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
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
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
# HF INFERENCE CLIENT
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_hf_client():
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    return InferenceClient(token=token)

def generate_response(prompt, max_tokens=400):
    try:
        client = get_hf_client()
        response = client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=max_tokens,
            temperature=0.3,
            stop_sequences=["</s>", "[INST]", "[/INST]"],
        )
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# =====================================================================
# EMBEDDINGS & VECTOR STORE
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

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
        st.error(f"Failed to process PDF: {str(e)}")
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
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 5:
            continue
        question = lines[0]
        options = {}
        answer = None
        for line in lines[1:]:
            opt_match = re.match(r'^([A-D])[\)\.]\s*(.*)', line, re.IGNORECASE)
            ans_match = re.match(r'^(?:Answer|Ans)[:\.\s]*([A-D])', line, re.IGNORECASE)
            if opt_match:
                options[opt_match.group(1).upper()] = opt_match.group(2)
            elif ans_match:
                answer = ans_match.group(1).upper()
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
for key, val in {
    "vectorstore": None, "chat_history": [], "pdf_processed": False,
    "mcqs": [], "summary": "", "topics": [], "pdf_name": "",
    "explanation": "", "last_explained": "", "pages": 0, "chunks": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🧠 EduGenie</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Your Smart Study Companion</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📥 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        if not st.session_state.pdf_processed:
            with st.spinner("Loading embedding model..."):
                embeddings = load_embeddings()
            with st.spinner("Analyzing and indexing document..."):
                vectorstore, pages, chunks = process_pdf(uploaded_file, embeddings)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
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
        st.success(f"✅ Loaded: {st.session_state.pdf_name}")
        st.markdown(f"**Pages:** {st.session_state.pages}")
        st.markdown(f"**Text Chunks:** {st.session_state.chunks}")
        if st.button("Reset App", type="secondary"):
            st.session_state.clear()
            st.rerun()

# =====================================================================
# MAIN UI
# =====================================================================
st.markdown("""
<div class="banner">
    <h1>EduGenie</h1>
    <p>Upload a PDF to generate summaries, chat with your document, and practice with quizzes.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="custom-card" style="text-align:center; padding:3rem;">
        <h3 style="margin-top:0;">👋 Welcome to EduGenie!</h3>
        <p style="font-size:1.1rem; color:#888;">Upload a PDF in the sidebar to get started.</p>
        <div style="display:flex; justify-content:center; gap:2rem; margin-top:2rem; flex-wrap:wrap;">
            <div style="flex:1; min-width:150px; padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                <h4>💬 RAG Chat</h4><p style="font-size:0.9rem; color:#888;">Chat with your document context.</p>
            </div>
            <div style="flex:1; min-width:150px; padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                <h4>📄 Summary</h4><p style="font-size:0.9rem; color:#888;">Bullet point summaries.</p>
            </div>
            <div style="flex:1; min-width:150px; padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                <h4>📝 MCQ Quiz</h4><p style="font-size:0.9rem; color:#888;">Auto-generated quizzes.</p>
            </div>
            <div style="flex:1; min-width:150px; padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                <h4>💡 Study Guide</h4><p style="font-size:0.9rem; color:#888;">Simple explanations.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    tab_chat, tab_summary, tab_mcq, tab_study = st.tabs([
        "💬 Chat Assistant", "📄 Smart Summary", "📝 MCQ Practice Quiz", "💡 Study Guide"
    ])

    # TAB 1: CHAT
    with tab_chat:
        st.markdown("### Chat with your PDF")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask a question about the document..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = get_context(st.session_state.vectorstore, prompt, k=5)
                    formatted_prompt = f"""[INST] You are EduGenie, a helpful educational assistant.
Answer the question ONLY using the context below. If the answer is not in the context, say "I could not find the answer in the document."

Context:
{context}

Question: {prompt} [/INST]"""
                    response = generate_response(formatted_prompt, max_tokens=300)
                    st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # TAB 2: SUMMARY
    with tab_summary:
        st.markdown("### Smart Summary Generator")
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                context = get_context(st.session_state.vectorstore, "summary key overview main points", k=10)
                formatted_prompt = f"""[INST] Summarize the following text in 5-8 clear bullet points. Start directly with the bullet points, no intro text.

Context:
{context} [/INST]"""
                st.session_state.summary = generate_response(formatted_prompt, max_tokens=400)
                st.rerun()

        if st.session_state.summary:
            formatted = st.session_state.summary.replace("\n", "<br>")
            st.markdown(f"""
            <div class="summary-card">
                <h3>📋 Summary</h3>
                <div style="line-height:1.6;">{formatted}</div>
            </div>""", unsafe_allow_html=True)

    # TAB 3: MCQ
    with tab_mcq:
        st.markdown("### Practice Quiz Generator")
        if st.button("Generate MCQs", type="primary"):
            with st.spinner("Generating quiz..."):
                context = get_context(st.session_state.vectorstore, "key concepts definitions facts", k=10)
                formatted_prompt = f"""[INST] Generate exactly 5 multiple choice questions from the context below.
Format each question exactly like this:
Q1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Letter]

Context:
{context} [/INST]"""
                raw_mcqs = generate_response(formatted_prompt, max_tokens=600)
                parsed = parse_mcqs(raw_mcqs)
                if parsed:
                    st.session_state.mcqs = parsed
                else:
                    st.session_state.mcqs = [{"fallback": raw_mcqs}]
                st.rerun()

        if st.session_state.mcqs:
            if "fallback" in st.session_state.mcqs[0]:
                st.warning("Could not parse quiz. Showing raw output:")
                st.text(st.session_state.mcqs[0]["fallback"])
            else:
                user_answers = {}
                with st.form("quiz_form"):
                    for idx, mcq in enumerate(st.session_state.mcqs):
                        st.markdown(f"**Q{idx+1}. {mcq['question']}**")
                        options_labels = [
                            f"A) {mcq['options'][0]}", f"B) {mcq['options'][1]}",
                            f"C) {mcq['options'][2]}", f"D) {mcq['options'][3]}"
                        ]
                        user_answers[idx] = st.radio("Select answer:", options_labels, key=f"mcq_{idx}", index=None)
                        st.markdown("<hr style='margin:0.5rem 0; opacity:0.1;'/>", unsafe_allow_html=True)

                    if st.form_submit_button("Submit Answers"):
                        score = 0
                        st.markdown("### 📊 Results")
                        for idx, mcq in enumerate(st.session_state.mcqs):
                            selected = user_answers[idx]
                            correct_letter = mcq['answer']
                            correct_text = f"{correct_letter}) {mcq['options'][ord(correct_letter)-65]}"
                            if selected:
                                if selected[0] == correct_letter:
                                    score += 1
                                    st.success(f"**Q{idx+1}: Correct!** {selected}")
                                else:
                                    st.error(f"**Q{idx+1}: Wrong.** You chose: {selected} | Correct: {correct_text}")
                            else:
                                st.warning(f"**Q{idx+1}: Unanswered.** Correct: {correct_text}")
                        total = len(st.session_state.mcqs)
                        st.metric("Score", f"{score}/{total}", f"{int(score/total*100)}%")

    # TAB 4: STUDY GUIDE
    with tab_study:
        st.markdown("### Study Guide & Explanation Companion")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔍 Topic Extractor")
            if st.button("Extract Key Topics"):
                with st.spinner("Extracting topics..."):
                    context = get_context(st.session_state.vectorstore, "core concepts topics", k=8)
                    formatted_prompt = f"""[INST] List the top 5 key topics from the context below.
One topic per line, no numbering, no extra text.

Context:
{context} [/INST]"""
                    raw_topics = generate_response(formatted_prompt, max_tokens=150)
                    parsed_topics = [t.strip().lstrip('-*•12345. ') for t in raw_topics.split('\n') if t.strip()]
                    st.session_state.topics = [t for t in parsed_topics if t][:5]
                    st.rerun()

            if st.session_state.topics:
                st.markdown("##### Key Topics:")
                for topic in st.session_state.topics:
                    st.markdown(f"<span class='topic-badge'>{topic}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 💡 Simple Explainer")
            explain_input = ""
            if st.session_state.topics:
                selected = st.selectbox("Select topic:", ["Custom"] + st.session_state.topics)
                if selected != "Custom":
                    explain_input = selected
            custom = st.text_input("Or type a topic:", value="")
            final_topic = custom if custom else explain_input

            if st.button("Explain Simply") and final_topic:
                with st.spinner("Generating explanation..."):
                    context = get_context(st.session_state.vectorstore, final_topic, k=5)
                    formatted_prompt = f"""[INST] Explain '{final_topic}' in simple, easy-to-understand language using the context below. Use analogies. Keep it beginner-friendly.

Context:
{context} [/INST]"""
                    st.session_state.explanation = generate_response(formatted_prompt, max_tokens=300)
                    st.session_state.last_explained = final_topic
                    st.rerun()

            if st.session_state.explanation:
                st.markdown(f"""
                <div class="summary-card">
                    <h3>💡 {st.session_state.last_explained}</h3>
                    <p style="line-height:1.6;">{st.session_state.explanation}</p>
                </div>""", unsafe_allow_html=True)
