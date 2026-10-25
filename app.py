import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Campus Rulebook QA Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .stChatInput {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# -----------------------------------------------------------------
# Sidebar Configuration & Ingestion
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key to power embeddings and generation."
    )
    
    st.markdown("---")
    st.markdown("### 📚 Policy Upload")
    uploaded_file = st.file_uploader(
        "Upload Academic Handbook",
        type=["pdf", "txt"],
        help="Upload the official campus policy PDF or text document."
    )
    
    if uploaded_file and api_key_input:
        if not st.session_state.doc_processed:
            with st.spinner("Processing policy document..."):
                try:
                    # Save uploaded file to a temporary file
                    file_extension = uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Load document based on extension
                    if file_extension.lower() == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding="utf-8")
                    
                    raw_docs = loader.load()
                    os.unlink(tmp_path) # Clean up temp file

                    # Split text: 500-character chunks with 80-character overlap
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=80,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(raw_docs)

                    # Initialize Embeddings and ChromaDB vector store
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key_input)
                    
                    # Create Chroma vector store in-memory
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.doc_processed = True
                    st.success(f"Successfully indexed {len(chunks)} chunks from {uploaded_file.name}!")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    elif uploaded_file and not api_key_input:
        st.warning("Please enter your OpenAI API key above to process the document.")

    if st.session_state.doc_processed:
        if st.button("🔄 Reset / Clear Handbook", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.messages = []
            st.session_state.doc_processed = False
            st.rerun()

# -----------------------------------------------------------------
# Main Chat Interface
# -----------------------------------------------------------------
st.markdown('<p class="main-header">🎓 Campus Rulebook QA Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about academic policies, codes of conduct, grading scales, and campus guidelines.</p>', unsafe_allow_html=True)
st.markdown("---")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Source Quotes"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {idx}** (Page {src.metadata.get('page', 'N/A') if 'page' in src.metadata else 'Text'}):")
                    st.quote(src.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about the academic policy (e.g., What is the attendance requirement?)"):
    if not api_key_input:
        st.error("Please enter your OpenAI API key in the sidebar to ask questions.")
    elif not st.session_state.vector_store:
        st.warning("Please upload and process an academic handbook in the sidebar first.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response with strict grounding
        with st.chat_message("assistant"):
            with st.spinner("Searching policy guidelines..."):
                try:
                    # Setup retriever and LLM
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        openai_api_key=api_key_input
                    )

                    system_prompt = (
                        "You are an official Campus Rulebook QA Assistant for college students and staff. "
                        "Answer the user's question strictly using ONLY the provided context below. "
                        "If the answer cannot be found in the context, state clearly: "
                        "'I cannot find the answer to this question in the uploaded academic policy handbook.' "
                        "Do not assume or extrapolate beyond the provided text.\n\n"
                        "Context:\n{context}"
                    )

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    # Execute query
                    response = rag_chain.invoke({"input": prompt})
                    answer_text = response["answer"]
                    source_docs = response.get("context", [])

                    st.markdown(answer_text)
                    
                    if source_docs:
                        with st.expander("🔍 View Source Quotes"):
                            for idx, doc in enumerate(source_docs, 1):
                                page_info = f"Page {doc.metadata.get('page', 'N/A') + 1}" if 'page' in doc.metadata else "Document Text"
                                st.markdown(f"**Source {idx}** ({page_info}):")
                                st.quote(doc.page_content)

                    # Save assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": source_docs
                    })

                except Exception as e:
                    error_msg = f"An error occurred while generating the response: {str(e)}"
                    st.error(error_msg)