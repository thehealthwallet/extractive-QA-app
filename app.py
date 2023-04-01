import pinecone
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

PINECONE_KEY = "0d6943e0-7c3c-40b0-af68-972b6916413b"  # app.pinecone.io

@st.cache_resource
def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="eu-west1-gcp")  # get a free api key from app.pinecone.io
    return pinecone.Index("extractive-question-answering")
    
@st.cache_resource
def init_models():
    retriever = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    model_name = 'deepset/electra-base-squad2'
    reader = pipeline(tokenizer=model_name, model=model_name, task='question-answering')
    return retriever, reader

st.session_state.index = init_pinecone()
retriever, reader = init_models()


def card(title, context, score):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
             <div  class="col-md-12 col-sm-12">
                 <b>{title}</b>
                 <br>
                 <span style="color: #808080;">
                     <small>{context}</small>
                     [<b>Score: </b>{score}]
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

st.title("")

st.write("""
# Extractive QA Bot
Ask a question!
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

def run_query(query):
    xq = retriever.encode([query]).tolist()
    try:
        xc = st.session_state.index.query(xq, top_k=3, include_metadata=True)
    except:
        # force reload
        pinecone.init(api_key=PINECONE_KEY, environment="us-west1-gcp")
        st.session_state.index = pinecone.Index("extractive-question-answering")
        xc = st.session_state.index.query(xq, top_k=3, include_metadata=True)

    results = []
    for match in xc['matches']:
        answer = reader(question=query, context=match["metadata"]['context'])
        answer["title"] = match["metadata"]['title']
        answer["context"] = match["metadata"]['context']
        results.append(answer)

    sorted_result = sorted(results, key=lambda x: x['score'], reverse=True)

    for r in sorted_result:
        answer = r["answer"]
        context = r["context"].replace(answer, f"<mark>{answer}</mark>")
        title = r["title"].replace("_", " ")
        score = round(r["score"], 4)
        card(title, context, score)

query = st.text_input("Search!", "")

if query != "":
    run_query(query)