import os
import re
import tempfile
import torch
import streamlit as st

# =====================================================================
# STREAMLIT APP CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="EduGenie - Intelligent PDF Study Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glowing Title Banner */
    .banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
    }
    
    .banner h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
        letter-spacing: -0.5px;
    }
    
    .banner p {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Custom Card Containers */
    .custom-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs custom styling */
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Interactive Results Styling */
    .summary-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(236, 72, 153, 0.08) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 1rem;
        box-shadow: 0 6px 18px rgba(124, 58, 237, 0.08);
    }
    
    .summary-card h3 {
        color: #7C3AED;
        font-size: 1.4rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .topic-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CACHED MODEL LOADERS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    from transformers import pipeline
    if torch.cuda.is_available():
        return pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=0,
            torch_dtype=torch.float16
        )
    else:
        return pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=-1,
            torch_dtype=torch.float32
        )

# =====================================================================
# INITIALIZE SESSION STATE
# =====================================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "mcqs" not in st.session_state:
    st.session_state.mcqs = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "topics" not in st.session_state:
    st.session_state.topics = []
if "embeddings_loaded" not in st.session_state:
    st.session_state.embeddings_loaded = False
if "llm_loaded" not in st.session_state:
    st.session_state.llm_loaded = False

# Lazy loader helper functions
def get_embeddings():
    model = load_embeddings()
    st.session_state.embeddings_loaded = True
    return model

def get_llm():
    model = load_llm()
    st.session_state.llm_loaded = True
    return model
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "last_explained" not in st.session_state:
    st.session_state.last_explained = ""

# =====================================================================
# CORE UTILITY FUNCTIONS
# =====================================================================
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
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        
        # Cleanup
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

def generate_response(generator, prompt, max_new_tokens=250, temperature=0.3, do_sample=True):
    try:
        response = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        output = response[0]["generated_text"]
        
        # Extract the assistant's final response cleanly
        parts = output.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip().replace("</s>", "")
        
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip().replace("</s>", "")
            
        return output.replace(prompt, "").strip().replace("</s>", "")
    except Exception as e:
        return f"Error during generation: {str(e)}"

def parse_mcqs(text):
    # Split text into question blocks
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
                opt_letter = opt_match.group(1).upper()
                options[opt_letter] = opt_match.group(2)
            elif ans_match:
                answer = ans_match.group(1).upper()
                
        if len(options) >= 4 and answer:
            mcqs.append({
                "question": question,
                "options": [options.get('A', ''), options.get('B', ''), options.get('C', ''), options.get('D', '')],
                "answer": answer
            })
            
    return mcqs[:5]

# =====================================================================
# APPLICATION INITIALIZATION (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 EduGenie</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Your Smart Study Companion</p>", unsafe_allow_html=True)
    st.divider()
    
    # Model Loading Status
    st.markdown("### 🤖 AI Model Status")
    if st.session_state.embeddings_loaded:
        st.success("🟢 Embeddings: Ready")
    else:
        st.info("⚪ Embeddings: Standby")
        
    if st.session_state.llm_loaded:
        st.success("🟢 AI Generator: Ready")
    else:
        st.info("⚪ AI Generator: Standby")
        
    st.divider()
    st.markdown("### 📥 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        if not st.session_state.pdf_processed:
            with st.spinner("Loading Embedding Model..."):
                embeddings = get_embeddings()
            with st.spinner("Analyzing and indexing document..."):
                vectorstore, pages, chunks = process_pdf(uploaded_file, embeddings)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.pages = pages
                    st.session_state.chunks = chunks
                    # Clear previous state on new PDF upload
                    st.session_state.chat_history = []
                    st.session_state.mcqs = []
                    st.session_state.summary = ""
                    st.session_state.topics = []
                    st.session_state.explanation = ""
                    st.rerun()
                    
    if st.session_state.pdf_processed:
        st.success(f"✅ Loaded: {uploaded_file.name}")
        st.markdown(f"**Pages:** {st.session_state.pages}")
        st.markdown(f"**Text Chunks:** {st.session_state.chunks}")
        
        if st.button("Reset App", type="secondary"):
            st.session_state.clear()
            st.rerun()

# =====================================================================
# MAIN USER INTERFACE
# =====================================================================
st.markdown("""
<div class="banner">
    <h1>EduGenie</h1>
    <p>Upload a PDF study guide, textbook, or notes to generate summaries, chat with topics, and practice with generated quizzes.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="custom-card" style="text-align: center; padding: 3rem;">
        <h3 style="margin-top: 0;">👋 Welcome to EduGenie!</h3>
        <p style="font-size: 1.1rem; color: #888;">To start using the app, please upload a PDF document in the sidebar.</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02)">
                <h4>💬 RAG Chat</h4>
                <p style="font-size: 0.9rem; color: #888;">Chat directly with the context of your file.</p>
            </div>
            <div style="flex: 1; min-width: 150px; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02)">
                <h4>📄 Summary</h4>
                <p style="font-size: 0.9rem; color: #888;">Get bullet points outlining the main document context.</p>
            </div>
            <div style="flex: 1; min-width: 150px; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02)">
                <h4>📝 MCQ Quizzes</h4>
                <p style="font-size: 0.9rem; color: #888;">Generate and take custom quizzes from the text.</p>
            </div>
            <div style="flex: 1; min-width: 150px; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02)">
                <h4>💡 Simple Explainer</h4>
                <p style="font-size: 0.9rem; color: #888;">Get simplified explanations for complex topics.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # PDF is loaded, display standard tabs
    tab_chat, tab_summary, tab_mcq, tab_study = st.tabs([
        "💬 Chat Assistant", 
        "📄 Smart Summary", 
        "📝 MCQ Practice Quiz", 
        "💡 Study Guide"
    ])
    
    # -----------------------------------------------------------------
    # TAB 1: CHAT ASSISTANT
    # -----------------------------------------------------------------
    with tab_chat:
        st.markdown("### Chat with your PDF")
        
        # Display chat history using Streamlit chat elements
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Ask input
        if prompt := st.chat_input("Ask a question about the uploaded document..."):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Retrieve context and answer
            with st.chat_message("assistant"):
                with st.spinner("Searching document & formulating answer..."):
                    context = get_context(st.session_state.vectorstore, prompt, k=5)
                    formatted_prompt = f"""<|system|>
You are a helpful educational AI assistant called EduGenie.
Answer the user's question ONLY using the provided Context.
Do NOT make up information.
If the answer is not found in the Context, say: "I could not find the answer in the document."

Context:
{context}</s>
<|user|>
{prompt}</s>
<|assistant|>
"""
                    response = generate_response(get_llm(), formatted_prompt, max_new_tokens=200, temperature=0.3)
                    st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # -----------------------------------------------------------------
    # TAB 2: SMART SUMMARY
    # -----------------------------------------------------------------
    with tab_summary:
        st.markdown("### Smart Summary Generator")
        st.write("Generate a brief and informative summary consisting of the main takeaways of the document.")
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Extracting text and compiling summary..."):
                context = get_context(st.session_state.vectorstore, "summary and key overview", k=10)
                formatted_prompt = f"""<|system|>
You are an expert summarizer. Summarize the provided context in 5 to 10 clear, high-quality bullet points. Do not include introductory text, start directly with the bullet points.

Context:
{context}</s>
<|user|>
Summarize the document.</s>
<|assistant|>
"""
                st.session_state.summary = generate_response(get_llm(), formatted_prompt, max_new_tokens=300, temperature=0.1, do_sample=False)
                st.rerun()
                
        if st.session_state.summary:
            formatted_summary = st.session_state.summary.replace("\n", "<br>")
            st.markdown(f"""
            <div class="summary-card">
                <h3>📋 Bullet Point Summary</h3>
                <div style="line-height: 1.6;">
                    {formatted_summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 3: MCQ PRACTICE QUIZ
    # -----------------------------------------------------------------
    with tab_mcq:
        st.markdown("### Practice Quiz Generator")
        st.write("Generate a multiple-choice practice quiz with 5 questions based on your uploaded document.")
        
        if st.button("Generate MCQs", type="primary"):
            with st.spinner("Crafting quiz questions..."):
                context = get_context(st.session_state.vectorstore, "key concepts, definitions, and facts", k=10)
                formatted_prompt = f"""<|system|>
You are an educational test generator. Generate exactly 5 multiple choice questions (MCQs) from the provided Context.
Format each question exactly as follows:
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Correct Option Letter, e.g., A]

Do not include any extra text. Generate exactly 5 questions.

Context:
{context}</s>
<|user|>
Generate 5 MCQs.</s>
<|assistant|>
"""
                raw_mcqs = generate_response(get_llm(), formatted_prompt, max_new_tokens=500, temperature=0.2, do_sample=False)
                parsed = parse_mcqs(raw_mcqs)
                
                if parsed:
                    st.session_state.mcqs = parsed
                else:
                    # In case parsing fails, save raw text for fallback display
                    st.session_state.mcqs = [{"fallback": raw_mcqs}]
                st.rerun()
                
        if st.session_state.mcqs:
            if "fallback" in st.session_state.mcqs[0]:
                st.warning("Could not automatically parse the quiz structure. Showing generated output:")
                st.text(st.session_state.mcqs[0]["fallback"])
            else:
                user_answers = {}
                with st.form("quiz_form"):
                    for idx, mcq in enumerate(st.session_state.mcqs):
                        st.markdown(f"**Q{idx+1}. {mcq['question']}**")
                        options_labels = [
                            f"A) {mcq['options'][0]}", 
                            f"B) {mcq['options'][1]}", 
                            f"C) {mcq['options'][2]}", 
                            f"D) {mcq['options'][3]}"
                        ]
                        user_answers[idx] = st.radio(
                            "Select answer:",
                            options_labels,
                            key=f"mcq_{idx}",
                            index=None
                        )
                        st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.1;'/>", unsafe_allow_html=True)
                        
                    submitted = st.form_submit_button("Submit Quiz Answers")
                    
                    if submitted:
                        score = 0
                        total = len(st.session_state.mcqs)
                        
                        st.markdown("### 📊 Results & Explanations")
                        
                        for idx, mcq in enumerate(st.session_state.mcqs):
                            selected = user_answers[idx]
                            correct_letter = mcq['answer']
                            correct_text = f"{correct_letter}) {mcq['options'][ord(correct_letter) - 65]}"
                            
                            if selected:
                                selected_letter = selected[0]
                                if selected_letter == correct_letter:
                                    score += 1
                                    st.success(f"**Q{idx+1}: Correct!** You selected: *{selected}*")
                                else:
                                    st.error(f"**Q{idx+1}: Incorrect.** You selected: *{selected}*. Correct answer: *{correct_text}*")
                            else:
                                st.warning(f"**Q{idx+1}: Unanswered.** Correct answer: *{correct_text}*")
                        
                        st.metric("Final Score", f"{score}/{total}", f"{int(score/total*100)}% Correct" if total > 0 else "0%")

    # -----------------------------------------------------------------
    # TAB 4: STUDY GUIDE
    # -----------------------------------------------------------------
    with tab_study:
        st.markdown("### Study Guide & Explanation Companion")
        st.write("Extract the main topics in your document and generate simple, plain-English explanations.")
        
        col_topics, col_explain = st.columns([1, 1])
        
        with col_topics:
            st.markdown("#### 🔍 Topic Extractor")
            if st.button("Extract Key Topics"):
                with st.spinner("Scanning document for key topics..."):
                    context = get_context(st.session_state.vectorstore, "core concepts and topics", k=8)
                    formatted_prompt = f"""<|system|>
You are an expert educational tutor. Extract the top 5 key topics or concepts discussed in the provided Context.
Format the output as a simple list of topics, one per line, with no extra text or numbering.

Context:
{context}</s>
<|user|>
Extract 5 key topics.</s>
<|assistant|>
"""
                    raw_topics = generate_response(get_llm(), formatted_prompt, max_new_tokens=150, temperature=0.2, do_sample=False)
                    parsed_topics = [t.strip().lstrip('-*•12345. ') for t in raw_topics.split('\n') if t.strip()]
                    st.session_state.topics = [t for t in parsed_topics if t][:5]
                    st.rerun()
            
            if st.session_state.topics:
                st.markdown("##### Extracted Key Topics:")
                for topic in st.session_state.topics:
                    st.markdown(f"<span class='topic-badge'>{topic}</span>", unsafe_allow_html=True)
                    
        with col_explain:
            st.markdown("#### 💡 Explain in Simple Language")
            
            # Populate options for selection if topics have been extracted
            explain_input = ""
            if st.session_state.topics:
                selected_topic = st.selectbox("Select an extracted topic or choose 'Custom Topic'", ["Custom Topic"] + st.session_state.topics)
                if selected_topic != "Custom Topic":
                    explain_input = selected_topic
                    
            # Let the user type custom text if they want
            custom_input = st.text_input("Or enter a concept/topic to explain:", value="" if explain_input else "")
            
            final_topic = custom_input if custom_input else explain_input
            
            if st.button("Explain Simply") and final_topic:
                with st.spinner(f"Analyzing '{final_topic}' and building simple explanation..."):
                    context = get_context(st.session_state.vectorstore, final_topic, k=5)
                    formatted_prompt = f"""<|system|>
You are an expert teacher who explains complex concepts in simple, easy-to-understand language. Explain the following query based on the Context, using simple analogies and formatting it for a beginner (e.g., explain like I'm 10 years old).

Context:
{context}</s>
<|user|>
Explain in simple terms: {final_topic}</s>
<|assistant|>
"""
                    st.session_state.explanation = generate_response(get_llm(), formatted_prompt, max_new_tokens=300, temperature=0.4)
                    st.session_state.last_explained = final_topic
                    st.rerun()
                    
            if st.session_state.explanation and st.session_state.last_explained:
                st.markdown(f"""
                <div class="summary-card">
                    <h3>💡 Simplified Explanation: {st.session_state.last_explained}</h3>
                    <p style="line-height: 1.6;">{st.session_state.explanation}</p>
                </div>
                """, unsafe_allow_html=True)
