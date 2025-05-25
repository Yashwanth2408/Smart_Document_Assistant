# Smart Document Assistant

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Yashwanth2408/Smart_Document_Assistant)
[![Working Video](https://img.shields.io/badge/Working-Video-green)](https://drive.google.com/drive/folders/1QK3T55u8egJfaf_IWesIMhnm7VG41Lcy?usp=sharing)

---

## 1. Project Overview

The **Smart Document Assistant** enables users to upload documents and interact with their content using natural language queries. By leveraging AI, vector similarity search, and modern NLP, it extracts highly relevant answers from large files—ideal for academic, legal, and enterprise use cases.

---

## 2. Problem Statement

Manually searching through lengthy documents is inefficient. This project allows users to query documents conversationally and receive accurate, context-aware responses instantly.

---

## 3. Objectives

- Automate extraction of document content
- Index content using vector embeddings
- Enable semantic querying via LLMs
- Provide a simple, intuitive UI for uploads and chat

---

## 4. Technology Stack

| Component          | Library / Tool                                         |
|--------------------|--------------------------------------------------------|
| Programming Lang   | Python 3.11                                            |
| LLM                | Gemini 1.5 Flash (`langchain_google_genai`)            |
| Embedding Model    | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector DB          | ChromaDB                                               |
| Text Splitter      | RecursiveCharacterTextSplitter (LangChain)             |
| File Parsing       | fitz, docx, pandas, pptx                               |
| UI                 | Streamlit                                              |
| Config             | python-dotenv                                          |

---

## 5. System Architecture

[User Uploads Document]

|

v

[File Parsing & Text Extraction]

|

v

[Text Chunking & Embedding]

|

v

[ChromaDB Vector Storage]

|

v

[User Query] --> [Similarity Search] --> [Gemini LLM] --> [Response]

---

## 6. Modules and Workflow

### File Handling

- Supports: `.pdf`, `.docx`, `.txt`, `.pptx`, `.xlsx`
- Specialized extractors for each format

### Preprocessing

- Splits text into 500-character chunks (100-character overlap)
- Embeds chunks using HuggingFace transformers

### Vector Storage

- Uses ChromaDB for persistent, efficient similarity search
- Collection: `"documents"`
- Resets DB on new session to avoid duplicates

### Query & Retrieval

- User query → vector search (top 3) → Gemini LLM prompt
- Gemini returns a document-based answer

---

## 7. Features

| Feature                | Description                                      |
|------------------------|--------------------------------------------------|
| Multi-format Support   | PDF, DOCX, TXT, PPTX, XLSX                       |
| Smart Text Extraction  | Format-specific extractors                       |
| Vector Search Engine   | ChromaDB for fast lookup                         |
| HuggingFace Embeddings | Converts chunks to vectors                       |
| Gemini AI Integration  | Contextual answers using latest LLM              |
| Chat History           | Session-based user and AI messages               |
| UI Frontend            | Streamlit-based                                  |
| Reset Functionality    | Wipes ChromaDB for fresh usage                   |

---

## 8. Code Structure & Examples

### `main.py` (CLI Interface)
from file_processing import process_uploaded_file, answer_query_with_gemini

file_path = "sample.pdf"
process_uploaded_file(file_path)

while True:
query = input("Ask your document: ")
if query.lower() == "exit":
break
response = answer_query_with_gemini(query)
print("AI:", response)


### `file_processing.py` (Core Logic)
def process_uploaded_file(file_path):
# Extract text based on file type
if file_path.endswith(".pdf"):
text = extract_text_from_pdf(file_path)
elif file_path.endswith(".docx"):
text = extract_text_from_docx(file_path)
# ... handle other formats
# Chunk, embed, and store in ChromaDB

def answer_query_with_gemini(query):
# Retrieve top 3 similar chunks from ChromaDB
# Compose prompt and query Gemini LLM
# Return Gemini's answer


### Utility Functions
def extract_text_from_pdf(file_path):
import fitz
doc = fitz.open(file_path)
text = ""
for page in doc:
text += page.get_text()
return text

def reset_chroma_db():
import shutil
shutil.rmtree("chroma_db_path")
# Reinitialize ChromaDB


### `app.py` (Streamlit UI)
import streamlit as st
from file_processing import process_uploaded_file, answer_query_with_gemini

st.title("AI Document Chatbot")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "pptx", "xlsx"])
if uploaded_file:
process_uploaded_file(uploaded_file)
st.success("File processed!")

user_query = st.text_input("Ask a question about your document:")
if user_query:
answer = answer_query_with_gemini(user_query)
st.write("AI:", answer)


---

## 9. Error Handling & Logging

- Checks for missing API keys
- Exception handling for file parsing and ChromaDB
- Detects empty/unsupported files with user feedback

---

## 10. Testing & Results

- Tested with research papers, company reports, presentations, etc.
- Consistently returns relevant answers within 3 seconds
- ChromaDB and Gemini integration verified

---

## 11. Challenges

- ChromaDB state caching: solved with `shutil.rmtree()`
- Secure Gemini API management: solved with `.env`
- XLSX/PPTX formatting: custom preprocessing for clean answers

---

## 12. Future Improvements

- Multi-document support with tagging
- PDF OCR for scanned files (Tesseract)
- Downloadable chat history
- File summarization
- Docker/Hugging Face Spaces deployment

---

## 13. Screenshots

> *(Add screenshots of each supported file type as shown in the original document)*

---

## 14. Conclusion

This chatbot combines modern NLP and LLMs for seamless, conversational document understanding. It removes the pain of manual searching and delivers fast, relevant answers for academic, business, or data-heavy workflows.

---

## Links

- **[GitHub Repository](https://github.com/praveenaperi/ai-doc-chatbot)**
- **[Working Demo Video](https://drive.google.com/drive/folders/1QK3T55u8egJfaf_IWesIMhnm7VG41Lcy?usp=sharing)**

---

## License

MIT License
