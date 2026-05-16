import os
import streamlit as st
import chromadb
from google import genai
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from dotenv import load_dotenv


load_dotenv()

# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = str(BASE_DIR / "vector_db")
COLLECTION_NAME = "admission_docs"

print("API KEY:", os.getenv("GEMINI_API_KEY"))

# =========================
# Gemini 설정
# =========================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


@st.cache_resource
def get_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    return chroma_client.get_collection(
        name=COLLECTION_NAME
    )

collection = get_chroma_collection()

# =========================
# BGE-M3 임베딩 모델
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")

embedding_model = load_embedding_model()


def embed_text(text):
    return embedding_model.encode(text).tolist()


# =========================
# 문서 검색
# =========================
def search_docs(query, n_results=6):
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""

    for doc, meta in zip(docs, metas):
        context += f"""
[파일]
{meta.get("file", "")}

[페이지]
{meta.get("page", "")}

[내용]
{doc}

"""

    return context, metas


# =========================
# 프롬프트 생성
# =========================
def make_admission_prompt(question, context):
    return f"""
너는 대학 모집요강 분석 AI이다.

반드시 제공된 문서 내용만 근거로 답변한다.

규칙:
1. 문서에 없는 내용은 절대 추측하지 않는다.
2. 수능최저, 반영비율, 전형방법, 모집인원, 지원자격은 정확하게 설명한다.
3. 여러 대학 정보가 섞이면 대학별로 구분해서 설명한다.
4. 불확실하거나 문서에서 찾을 수 없으면 "제공된 문서에서는 확인되지 않습니다."라고 답변한다.
5. 답변 마지막에 반드시 참고한 파일명과 페이지를 정리한다.
6. 표로 정리할 수 있으면 표를 사용한다.
7. 학생이 이해하기 쉽게 설명한다.

[참고문서]
{context}

[질문]
{question}

[답변 형식]
1. 핵심 답변
2. 세부 설명
3. 주의사항
4. 참고 문서
"""


def make_consulting_prompt(question, context, student_info):
    return f"""
너는 대한민국 대입 수시모집 전문 입시 컨설턴트 AI이다.

학생의 정보와 제공된 모집요강, 입시결과 문서를 바탕으로 학생부종합전형 중심의 지원 전략을 분석한다.

중요 규칙:
1. 제공된 문서에 없는 사실은 만들어내지 않는다.
2. 모집요강의 전형방법, 수능최저, 면접 여부, 반영비율을 우선 확인한다.
3. 2026 입시결과가 문서에 있으면 참고하되, 2027 지원 가능성을 단정하지 않는다.
4. 지원 전략은 상향 / 적정 / 안정으로 구분한다.
5. 학생에게 유리한 점과 불리한 점을 함께 설명한다.
6. 불확실하면 "추가 확인이 필요합니다."라고 말한다.
7. 답변 마지막에 참고한 파일명과 페이지를 정리한다.
8. 학생 눈높이에서 쉽게 설명한다.

[학생 정보]
{student_info}

[참고문서]
{context}

[질문]
{question}

[답변 형식]
1. 핵심 요약
2. 지원 가능성 분석
3. 추천 전략
4. 주의해야 할 점
5. 추가로 확인하면 좋은 정보
6. 참고 문서
"""


# =========================
# Gemini 답변 생성
# =========================
def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# =========================
# 화면 구성
# =========================
st.set_page_config(
    page_title="대입 모집요강 AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 모집요강 & 입시 컨설팅 AI")

st.caption("CSIA 친구들아, 우리 끝까지 해내자🔥 19기 화이팅🔥🔥🔥")

mode = st.radio(
    "사용 모드 선택",
    ["모집요강 검색", "입시 컨설팅"],
    horizontal=True
)

st.divider()

# =========================
# 입시 컨설팅용 학생 정보
# =========================
student_info = ""

if mode == "입시 컨설팅":
    st.subheader("학생 정보 입력")

    col1, col2 = st.columns(2)

    with col1:
        grade = st.text_input("내신 등급", placeholder="예: 2.8")
        region = st.text_input("희망 지역", placeholder="예: 수도권, 충청권")

    with col2:
        major = st.text_input("희망 학과/계열", placeholder="예: 약학과, 생명과학과")

    student_info = f"""
내신 등급: {grade}
희망 지역: {region}
희망 학과/계열: {major}
"""

st.subheader("질문 입력")

if mode == "모집요강 검색":
    st.markdown("""
예시)
- 약학과 모집인원 알려줘
- 학생부종합 전형방법 알려줘
- 수능최저 있는 전형 알려줘
- 면접 반영비율 알려줘
- 지원자격 알려줘
""")
else:
    st.markdown("""
예시)
- 내신 2.8인데 생명과학과 학생부종합으로 어디가 유리할까?
- 수도권 약학과 지원 전략 짜줘
- 내신 3.1이면 상향/적정/안정으로 나눠줘
- 면접 있는 전형이 나한테 유리할까?
- 수능최저 없는 대학 중심으로 추천해줘
""")

question = st.text_input("질문", placeholder="궁금한 내용을 자유롭게 입력하세요.")

search_button = st.button("AI에게 물어보기", type="primary")

# =========================
# 실행
# =========================
if search_button and question:

    with st.spinner("문서를 검색하고 답변을 생성하는 중입니다..."):

        if mode == "입시 컨설팅":
            search_query = f"{student_info}\n{question}"
        else:
            search_query = question

        context, metas = search_docs(search_query, n_results=6)

        if mode == "모집요강 검색":
            prompt = make_admission_prompt(question, context)
        else:
            prompt = make_consulting_prompt(question, context, student_info)

        answer = ask_gemini(prompt)

    st.divider()

    st.subheader("AI 답변")
    st.write(answer)

    st.subheader("참고 문서")

    shown = set()

    for meta in metas:
        key = (meta.get("file"), meta.get("page"))

        if key in shown:
            continue

        shown.add(key)

        st.write(
            f"- {meta.get('file', '')} / {meta.get('page', '')}페이지"
        )

    with st.expander("검색된 원문 보기"):
        st.text(context)

elif search_button and not question:
    st.warning("질문을 입력해주세요.")