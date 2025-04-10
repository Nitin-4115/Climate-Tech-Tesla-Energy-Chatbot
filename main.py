import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚫 Gemini API key is missing. Please set it in your .env file.")
    st.stop()

# Maintenance Mode Check (via Streamlit Secret)
APP_ONLINE = st.secrets.get("APP_ONLINE", "false").lower() == "true"

if not APP_ONLINE:
    st.set_page_config(page_title="Chatbot Offline", page_icon="💤")
    st.image("./assets/error_icon.png", width=120)
    st.markdown("## 🛠️ The Climate Tech Chatbot is currently under maintenance.")
    st.info("This app is temporarily offline. Please check back later. 🚧")
    st.stop()

# Configuration
class Config:
    def __init__(self):
        base_path = Path(__file__).resolve().parent
        self.document_path = base_path / "corpus.txt"
        self.index_path = base_path / "faiss_index"
        self.embedding_model = "models/embedding-001"
        self.llm_model = "gemini-2.0-flash-lite"
        self.llm_temperature = 0.3
        self.search_kwargs = {"k": 3}
        self.chunk_size = 1000
        self.chunk_overlap = 200

CONFIG = Config()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONFIG.chunk_size,
    chunk_overlap=CONFIG.chunk_overlap
)

@st.cache_data(show_spinner=False)
def load_documents() -> List[Document]:
    try:
        if not CONFIG.document_path.exists():
            st.error("Corpus file not found. Please upload a document first.")
            return []

        with open(CONFIG.document_path, "r", encoding="utf-8") as f:
            text = f.read()
            if not text.strip():
                st.error("The corpus file is empty.")
                return []

            documents = text_splitter.create_documents([text])
            for doc in documents:
                doc.metadata = {"source": "tesla_energy_corpus"}
            return documents

    except Exception as e:
        st.error(f"Failed to load documents: {str(e)}")
        return []

@st.cache_resource(show_spinner="Building search index...")
def create_and_save_index() -> bool:
    try:
        documents = load_documents()
        if not documents:
            return False

        embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG.embedding_model,
            google_api_key=GEMINI_API_KEY
        )

        db = FAISS.from_documents(documents, embeddings)
        db.save_local(str(CONFIG.index_path))
        return True

    except Exception as e:
        st.error(f"Index creation failed: {str(e)}")
        return False

@st.cache_resource(show_spinner="Loading search index...")
def load_faiss_index() -> Optional[FAISS]:
    try:
        required_files = ["index.faiss", "index.pkl"]
        index_exists = all((CONFIG.index_path / fname).exists() for fname in required_files)

        if not index_exists:
            return None

        embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG.embedding_model,
            google_api_key=GEMINI_API_KEY
        )

        return FAISS.load_local(
            str(CONFIG.index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        return None

def initialize_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CONFIG.llm_model,
        google_api_key=GEMINI_API_KEY,
        temperature=CONFIG.llm_temperature
    )

def handle_user_query(prompt: str, vector_store: FAISS, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs=CONFIG.search_kwargs),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return qa_chain({"question": prompt})

def display_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)

def display_sources(source_documents: List[Document]):
    with st.expander("📚 Source References"):
        for i, doc in enumerate(source_documents, 1):
            st.markdown(f"**Source {i}**")
            st.caption("Document snippet")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            st.divider()

def setup_ui():
    st.set_page_config(page_title="Climate Tech (Tesla Energy) Chatbot", page_icon="./assets/logo.png", layout="wide")

    st.markdown("""
        <style>
        /* Import Manrope font */
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Manrope', sans-serif !important;
        }

        /* Title */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Manrope', sans-serif !important;
            font-weight: 700 !important;
            color: white !important;
        }

        /* Markdown paragraphs and chat text */
        .stMarkdown p, .stText, .stChatMessageContent, .stChatMessage {
            font-family: 'Manrope', sans-serif !important;
            font-size: 1rem !important;
            color: #E0E0E0;
        }

        /* Chat input box */
        [data-testid="stChatInput"] textarea {
            font-family: 'Manrope', sans-serif !important;
            font-size: 1rem !important;
            color: white !important;
            background-color: #1F222E;
            border-radius: 8px;
        }

        [data-testid="stChatInput"] button {
            font-family: 'Manrope', sans-serif !important;
            background-color: #00FFF7;
            color: black;
            border-radius: 6px;
        }

        /* Sidebar */
        .stSidebar, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
            font-family: 'Manrope', sans-serif !important;
        }

        .stSidebar [data-testid="stFileUploaderDropzone"] {
            background-color: #1F222E;
            border-radius: 10px;
            border: 1px dashed #00FFF7;
        }

        /* General input fields and buttons */
        .stButton>button, .stDownloadButton>button,
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div {
            font-family: 'Manrope', sans-serif !important;
        }

        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="multi-title">
            <h1><span class="emoji">🤖</span> <span class="gradient-text">Climate Tech (Tesla Energy) Chatbot</span></h1>
            <p class="glow-subtitle">Your intelligent assistant for Tesla Energy products and climate policy</p>
        </div>
    
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
    
        .multi-title {
            font-family: 'Manrope', sans-serif;
            margin-bottom: 1.5rem;
        }
    
        .multi-title h1 {
            font-size: 2.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.3rem;
            white-space: nowrap;
            overflow: hidden;
        }
    
        .emoji {
            font-size: 2.3rem;
            color: white;
            flex-shrink: 0;
        }
    
        .gradient-text {
            background: linear-gradient(90deg, #00FFF7, #39FF14, #FF5ACD, #00FFF7);
            background-size: 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: scroll-gradient 10s linear infinite;
            display: inline-block;
        }
    
        .glow-subtitle {
            font-size: 1rem;
            color: #AFAFAF;
            font-weight: 400;
            margin-top: -8px;
        }
    
        @keyframes scroll-gradient {
            0%   { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.first_run = True

def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload new corpus document", type=["txt"])

        if uploaded_file:
            with st.spinner("Updating knowledge base..."):
                with open(CONFIG.document_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if create_and_save_index():
                    st.success("Knowledge base updated successfully!")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.messages = []
                    st.session_state.first_run = True
                    st.rerun()
                else:
                    st.error("Failed to update knowledge base")

        st.divider()
        if st.button("🔄 Rebuild Search Index", help="Force rebuild the search index"):
            with st.spinner("Rebuilding search index..."):
                st.cache_data.clear()
                st.cache_resource.clear()
                success = create_and_save_index()
            if success:
                st.success("✅ Search index rebuilt successfully! You can now start chatting.")
                st.session_state.messages = []
                time.sleep(2.5)
                st.rerun()
            else:
                st.error("❌ Failed to rebuild search index. Please check the corpus file and try again.")

        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("""
        **About this chatbot:**
        - Answers questions about Tesla Energy products
        - Provides climate policy information
        - Uses Gemini 2.0 Flash Lite model

        **Tips:**
        - Ask about Solar Panels, Powerwall, or Megapack
        - Ask about clean energy or carbon offset policies
        """)

def use_sample_corpus():
    sample_text = """
    Tesla Energy is a division of Tesla, Inc. that focuses on renewable energy products and services, including solar panels, Solar Roof, Powerwall, Powerpack, and Megapack. These products help individuals and businesses reduce reliance on fossil fuels.

    The Tesla Powerwall is a rechargeable home battery system designed to store energy from solar or the grid, providing backup power and enabling time-of-use load shifting.

    The Tesla Megapack is a large-scale energy storage system used by utilities and large businesses to store energy, reduce reliance on peaker plants, and stabilize the grid.

    Tesla's mission includes accelerating the world’s transition to sustainable energy through affordable electric vehicles, solar energy, and integrated renewable systems.
    """
    with open(CONFIG.document_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    return create_and_save_index()

def main():
    setup_ui()
    sidebar_controls()

    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    vector_store = load_faiss_index()
    llm = initialize_llm()

    if st.session_state.get("first_run", True):
        if vector_store is None:
            st.warning("⚠️ Search index not found.")
            if st.button("Use Sample Tesla Corpus"):
                if use_sample_corpus():
                    st.success("Sample corpus loaded successfully!")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.first_run = True
                    st.rerun()
                else:
                    st.error("Failed to load sample corpus.")
        else:
            st.success("✅ System ready! Ask me about Tesla Energy or climate policy.")
        st.session_state.first_run = False

    if prompt := st.chat_input("Ask about Tesla Energy or climate policy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        if vector_store is None:
            st.error("System not ready. Please upload a document or load the sample corpus.")
            return

        with st.spinner("Analyzing your question..."):
            try:
                start_time = time.time()
                response = handle_user_query(prompt, vector_store, llm)
                processing_time = time.time() - start_time
                answer = f"{response['answer']}\n\n*⏱️ Processed in {processing_time:.2f}s*"
                display_chat_message("assistant", answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if response.get("source_documents"):
                    display_sources(response["source_documents"])

            except Exception as e:
                error_msg = f"⚠️ Sorry, I encountered an error processing your request:\n\n```\n{str(e)}\n```"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index")
    args = parser.parse_args()

    if args.rebuild:
        if create_and_save_index():
            print("Index rebuilt successfully")
        else:
            print("Index rebuild failed")
    else:
        main()
