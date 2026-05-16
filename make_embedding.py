import os
import time
import chromadb
from google import genai
from pypdf import PdfReader
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent

DOCS_PATH = BASE_DIR / "docs" / "001"
CHROMA_PATH = BASE_DIR / "vector_db"
COLLECTION_NAME = "admission_docs"

model = SentenceTransformer("BAAI/bge-m3")

# =========================
# PDF 읽기
# =========================
def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    print("PDF 전체 페이지 수:", len(reader.pages))
 
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        print(page_no, "페이지 텍스트 길이:", len(text) if text else 0)
        #print(page_no, text)

        if text and text.strip():
            pages.append({
                "page": page_no,
                "text": text.strip()
            })
    return pages


# =========================
# 텍스트 쪼개기
# =========================
def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================
# Gemini 임베딩
# =========================
def embed_text(text):
    return model.encode(text).tolist()


# =========================
# 실행
# =========================
def main():
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"docs 폴더가 없습니다: {DOCS_PATH}")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None
    )

    pdf_files = list(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        print("docs 폴더에 PDF 파일이 없습니다.")
        return

    for pdf_file in pdf_files:
        print(f"\nPDF 처리 시작: {pdf_file.name}")

        pages = read_pdf(pdf_file)
        print("읽은 페이지 수:", len(pages))

        for page in pages:
            print("텍스트 길이:", len(page["text"]))

            chunks = chunk_text(page["text"])

            #print("청크 개수:", len(chunks))
            #print(chunks[:1])

            for idx, chunk in enumerate(chunks):
                doc_id = f"{pdf_file.parent.name}_{pdf_file.stem}_p{page['page']}_c{idx}"                
                #print("doc_id:", doc_id)

                try:
                    embedding = embed_text(chunk)
                    print(len(embedding))
                    #print("임베딩 완료")

                except Exception as e:
                    print("임베딩 오류:", e)
                    continue

                try:
                    collection.upsert(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "file": pdf_file.name,
                            "page": page["page"],
                            "chunk": idx
                        }]
                    )
                    print(f"저장 완료: {doc_id}")

                except Exception as e:
                    print("Chroma 저장 오류:", e)
                    continue
                    
                time.sleep(0.3)

    print("\n전체 임베딩 생성 완료")


if __name__ == "__main__":
    main()