# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG Chatbot'
load_dotenv()  # expects google_API_KEY in .env

PDF_PATH = "Oliver Wyman DNA India Decode 2026 Round 1 Brief.pdf"  # <-- change to your PDF filename
INDEX_PATH = Path(".faiss_index")

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index (reuse the index so every run does not consume quota)
emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-002")
if INDEX_PATH.exists():
    vs = FAISS.load_local(
        str(INDEX_PATH),
        emb,
        allow_dangerous_deserialization=True,
    )
else:
    for attempt in range(3):
        try:
            vs = FAISS.from_documents(splits, emb)
            vs.save_local(str(INDEX_PATH))
            break
        except Exception as exc:
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            if attempt == 2:
                raise RuntimeError(
                    "Gemini embedding quota is still exhausted. Wait for the "
                    "quota window to reset, then run this script again."
                ) from exc
            wait_seconds = 20 * (attempt + 1)
            print(f"Embedding quota reached; retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
