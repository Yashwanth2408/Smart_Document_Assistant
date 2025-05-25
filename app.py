import streamlit as st
import os
from file_processing import process_uploaded_file, answer_query_with_gemini, reset_chroma_db

st.set_page_config(page_title="📄 AI Document Chatbot", layout="wide")

# Ensure 'uploaded_files' directory exists
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Title
st.title("📄 AI-Powered Document Chatbot")

# File Upload
uploaded_file = st.file_uploader(
    "📂 Upload a document (PDF, DOCX, TXT, XLSX, PPTX)",
    type=["pdf", "docx", "txt", "xlsx", "pptx"]
)

if uploaded_file:
    # Save file locally
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # Prevent duplicate processing
    if "processed_file" not in st.session_state or st.session_state.processed_file != file_path:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the uploaded document
        with st.spinner("⏳ Processing document..."):
            process_uploaded_file(file_path)

        st.session_state.processed_file = file_path
        st.success(f"**{uploaded_file.name}** uploaded & stored in ChromaDB!")

# Restart Button
if st.button("Restart"):
    reset_chroma_db()
    st.session_state.clear()
    st.success("ChromaDB reset. Please upload a new file.")

# Chat Interface (Only show if a document is processed)
if "processed_file" in st.session_state:
    st.subheader("💬 Ask Questions About Your Document")

    # Preserve conversation history
    st.session_state.setdefault("messages", [])

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.text_input("🔍 Ask your question here:")

    if st.button("Ask"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("🤖 AI is thinking..."):
                response = answer_query_with_gemini(user_input)

            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

# Footer
st.markdown("---")
st.markdown("🚀 *Built with Streamlit & LangChain*")
