import os
import math
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/dataset", StaticFiles(directory="C:/Users/PBS/Desktop/Information Retrieval/dataset"), name="dataset")

templates = Jinja2Templates(directory="templates")

# Utility functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    return [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]

def term_frequency(term, document):
    return document.count(term) / len(document) 

def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (num_docs_containing_term)) if num_docs_containing_term > 0 else 0

def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0

def extract_meta_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    return meta_tag['content'].strip() if meta_tag and 'content' in meta_tag.attrs else ""

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else ""
    body = soup.body.get_text(separator=" ") if soup.body else ""
    return title, body

def load_html_files(folder_path):
    titles = []
    descriptions = []
    documents = []
    filenames = []  

    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                title, body = extract_text_from_html(content)
                description = extract_meta_description(content)
                
                titles.append(title)
                descriptions.append(description)
                documents.append(clean_text(title + " " + body))
                filenames.append(filename)  # Store the actual filename

    return titles, descriptions, documents, filenames

# Route to handle search
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse("404.html", {"request": request})
    return await request.app.default_exception_handler(request, exc)

# Route to handle search
@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, query: str):
    folder_path = './dataset/'
    titles, descriptions, documents, filenames = load_html_files(folder_path)

    # Clean the query
    cleaned_query = clean_text(query)

    # Get unique vocabulary
    vocab = sorted(set(word for doc in documents + [cleaned_query] for word in doc))

    # Compute TF-IDF vectors
    query_vector = compute_tfidf(cleaned_query, documents, vocab)
    doc_vectors = [compute_tfidf(doc, documents, vocab) for doc in documents]

    # Calculate cosine similarity
    similarities = [(titles[i], descriptions[i], filenames[i], cosine_similarity(query_vector, doc_vector)) for i, doc_vector in enumerate(doc_vectors)]
    
    # Sort similarities
    similarities = sorted(similarities, key=lambda x: x[3], reverse=True)

    # Check if all similarities are zero
    if all(similarity[3] == 0 for similarity in similarities):
        raise HTTPException(status_code=404, detail="No similar documents found")

    # Pass results to the template
    return templates.TemplateResponse("search_results.html", {
        "request": request,
        "query": query,
        "results": similarities,
    })

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})