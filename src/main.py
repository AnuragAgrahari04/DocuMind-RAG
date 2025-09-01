# import os
# from os.path import dirname
#
# import streamlit as st
#
# from doc_chat_utility import get_answer
#
# working_dir = os.path.dirname(os.path.abspath(__file__))
#
# st.set_page_config(
#     page_title = "Chat with Doc",
#     page_icon = "🖷",
#     layout = "centered"
# )
#
# st.title("Document Q&a - Llama 3 - Ollama")
#
# uploaded_file = st.file_uploader(label = "Upload you file", type = "pdf")
#
# user_query = st.text_input("Ask you questions")
#
# if st.button("Run"):
#     bytes_data = uploaded_file.read()
#     file_name = uploaded_file.name
#
#     #save the file to the working directory
#     file_path = os.path.join(working_dir,file_name)
#     with open(file_path,"wb") as f:
#         f.write(bytes_data)
#     answer = get_answer(file_name,user_query)
#
#     st.success(answer)


import os
import streamlit as st
from doc_chat_utility import create_vector_store, create_conversational_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    /* Target all chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        background-color: #f0f2f6 !important;
    }
    /* Target the text paragraph inside the chat message for visibility */
    .stChatMessage p {
        color: #1E1E1E !important;
    }
    /* Target info boxes (like the 'Currently chatting with...' one) */
    .stAlert p {
        color: #262730 !important;
    }
    /* Custom class for the RAG info text in the sidebar */
    .rag-info-text {
        color: white !important;
        background-color: #1F456E;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .st-emotion-cache-janbn0 {
        background-color: #e0e0e0;
    }
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🧠 DocuMind RAG")
    st.markdown("---")
    st.header("Configuration")

    # Feature: Model Selection
    selected_model = st.selectbox(
        "Choose your model",
        ("gemma:2b", "llama3:8b", "mistral"),
        index=0
    )

    # Feature: Multi-file Uploader
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                # Setup directory for uploads
                working_dir = os.path.dirname(os.path.abspath(__file__))
                uploads_dir = os.path.join(working_dir, "uploads")
                os.makedirs(uploads_dir, exist_ok=True)

                # Save uploaded files and get their paths
                file_paths = []
                file_names = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(uploads_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    file_paths.append(file_path)
                    file_names.append(uploaded_file.name)

                # Create vector store and conversational chain
                vector_store = create_vector_store(file_paths)
                st.session_state.conversation = create_conversational_chain(vector_store, selected_model)
                st.session_state.processed_files = file_names
                st.session_state.messages = [{"role": "assistant", "content": "Documents processed! Ask me anything."}]
                st.rerun()
        else:
            st.warning("Please upload at least one document.")

    st.markdown("---")
    # Apply the custom class to the info text
    st.markdown(
        '<div class="rag-info-text">This app uses Retrieval-Augmented Generation (RAG) to answer questions from your documents.</div>',
        unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Clear Chat & Files"):
        st.session_state.clear()
        st.rerun()

# --- Main Chat Interface ---
st.header("RAG-Powered Document Chat")

# Display list of processed documents
if "processed_files" in st.session_state and st.session_state.processed_files:
    st.info(f"Currently chatting with: **{', '.join(st.session_state.processed_files)}**")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Welcome! Please upload your documents and click 'Process Documents' to begin."}]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(
                        f"{os.path.basename(source.metadata.get('source', ''))} (Page {source.metadata.get('page', 'N/A')})")
                    st.text(f"\"...{source.page_content[:250]}...\"")

# Chat Input
if user_query := st.chat_input("Ask a question..."):
    if "conversation" in st.session_state and st.session_state.conversation:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_query})
            answer = response['answer']
            sources = response['source_documents']
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()
    else:
        st.warning("Please upload and process documents before asking questions.")







