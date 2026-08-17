import re
import os
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

_DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents_dataset")

def load_documents_from_folder(folder=_DATASET_FOLDER):
    docs = {}
    if os.path.isdir(folder):
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".txt"):
                name = os.path.splitext(fname)[0].replace("_", " ")
                fpath = os.path.join(folder, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        docs[name] = text
                except Exception:
                    pass
    return docs

documents = {
    "Doc 1":  "Artificial intelligence refers to computer systems that can perform tasks normally requiring human intelligence, including visual perception, speech recognition, decision making, and natural language translation.",
    "Doc 2":  "Cloud computing delivers on-demand computing resources over the internet, such as servers, storage, databases, networking, software, and analytics, without direct active management by the user.",
    "Doc 3":  "Climate change describes long-term shifts in global temperatures and weather patterns, mainly driven by human activities like burning fossil fuels, deforestation, and industrial greenhouse gas emissions.",
    "Doc 4":  "Machine learning is a subset of artificial intelligence that enables systems to learn patterns from data and improve their performance on a specific task without being explicitly programmed.",
    "Doc 5":  "Renewable energy comes from naturally replenishing sources such as sunlight, wind, rain, tides, waves, and geothermal heat, offering a cleaner alternative to fossil fuels like coal and oil.",
    "Doc 6":  "The Internet of Things connects everyday physical objects embedded with sensors, software, and network connectivity, enabling them to collect and exchange data across smart homes and cities.",
    "Doc 7":  "Deforestation involves the permanent removal of trees to make room for agriculture, grazing, or urban expansion, threatening biodiversity and accelerating global carbon dioxide levels.",
    "Doc 8":  "Cybersecurity protects computer systems, networks, programs, and data from digital attacks, damage, or unauthorized access through encryption, firewalls, and intrusion detection systems.",
    "Doc 9":  "Solar power generates electricity by converting sunlight using photovoltaic panels or concentrated solar power, providing a sustainable and increasingly affordable source of renewable energy.",
    "Doc 10": "Online learning delivers education through digital platforms, video lectures, interactive quizzes, and virtual classrooms, enabling students to study from anywhere at their own pace.",
    "Doc 11": "Blockchain is a decentralized and distributed digital ledger that records transactions across many computers so that no single participant can alter the record retroactively without changing all subsequent blocks.",
    "Doc 12": "Water pollution occurs when harmful substances like chemicals, plastics, or sewage contaminate rivers, lakes, oceans, and groundwater, damaging aquatic ecosystems and human drinking water supplies.",
    "Doc 13": "Robotics combines mechanical engineering, electronic engineering, and computer science to design, construct, operate, and use robots that can assist humans in manufacturing, healthcare, and exploration.",
    "Doc 14": "Wind energy uses large turbines mounted on towers to capture the kinetic energy of moving air and convert it into electrical power, producing zero fuel cost or greenhouse gas emissions.",
    "Doc 15": "Natural language processing is a branch of artificial intelligence that helps computers understand, interpret, and generate human language, powering chatbots, translation tools, and voice assistants.",
    "Doc 16": "Electric vehicles use rechargeable battery packs to power electric motors instead of internal combustion engines, reducing transportation emissions and dependence on petroleum-based gasoline.",
    "Doc 17": "Computer vision enables machines to extract meaningful information from digital images and videos, enabling applications like facial recognition, autonomous vehicles, and medical image analysis.",
    "Doc 18": "Biodiversity refers to the variety of all living species on Earth, including plants, animals, bacteria, and fungi, and it supports ecosystem stability, food security, and natural medicine discovery.",
    "Doc 19": "Big data analytics examines extremely large and varied datasets to uncover hidden patterns, market trends, customer preferences, and other useful business insights using specialized software tools.",
    "Doc 20": "Digital libraries organize and provide access to vast collections of electronic books, journals, research papers, and multimedia resources, enabling remote learning and global knowledge sharing."
}

loaded_from_folder = load_documents_from_folder()
if len(loaded_from_folder) >= 10:
    documents = loaded_from_folder

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    cleaned = []
    for word in tokens:
        if word not in stop_words and len(word) > 1:
            cleaned.append(stemmer.stem(word))
    return " ".join(cleaned)

def raw_term_counts(doc_text, query_terms):
    doc_tokens = re.sub(r'[^a-zA-Z\s]', '', doc_text.lower()).split()
    stemmed_doc_tokens = [stemmer.stem(t) for t in doc_tokens if len(t) > 1]
    counts = {}
    for qt in query_terms:
        counts[qt] = stemmed_doc_tokens.count(qt)
    return counts, sum(counts.values())

processed_corpus = {doc_id: preprocess_text(text) for doc_id, text in documents.items()}

print("=" * 78)
print("  TF-IDF INFORMATION RETRIEVAL PROJECT")
print("  Dataset: 20 Documents on Technology, Environment & Education")
print("=" * 78)

print("\n--- SECTION 1: SAMPLE PREPROCESSING OUTPUT ---")
for i, (doc_id, cleaned) in enumerate(list(processed_corpus.items())[:4], 1):
    print(f"\n{doc_id} (Original): {documents[doc_id][:110]}...")
    print(f"{doc_id} (Cleaned) : {cleaned}")

vectorizer = TfidfVectorizer()
corpus_list = list(processed_corpus.values())
doc_ids = list(processed_corpus.keys())
tfidf_matrix = vectorizer.fit_transform(corpus_list)
feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=doc_ids,
    columns=feature_names
)

print("\n--- SECTION 2: TF-IDF FEATURE MATRIX SUMMARY ---")
print(f"  Total documents (corpus size) : {len(doc_ids)}")
print(f"  Unique vocabulary terms (V)   : {len(feature_names)}")
print(f"  TF-IDF matrix dimensions      : {tfidf_matrix.shape[0]} rows x {tfidf_matrix.shape[1]} columns")
print(f"  Non-zero entries in matrix    : {tfidf_matrix.nnz}")
print(f"  Matrix sparsity               : {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f} %")

def search_and_rank(user_query, top_k=10):
    processed_q = preprocess_text(user_query)
    query_tokens = processed_q.split()
    query_vector = vectorizer.transform([processed_q])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked = np.argsort(scores)[::-1]

    print("\n" + "=" * 78)
    print(f"  USER QUERY           : {user_query}")
    print(f"  PREPROCESSED QUERY   : {processed_q}")
    print(f"  QUERY TERMS (STEMMED): {query_tokens}")
    print("=" * 78)

    header = f"  {'Rank':<5}{'Doc ID':<8}{'Total':<7}{'Term Counts':<35}{'TF-IDF Score':<12}"
    print(header)
    print("  " + "-" * 74)

    results_data = []
    for rank, idx in enumerate(ranked[:top_k], 1):
        doc_id = doc_ids[idx]
        score = scores[idx]
        per_term, total_count = raw_term_counts(documents[doc_id], query_tokens)
        count_str = ", ".join([f"{t}:{c}" for t, c in per_term.items()])
        snippet = documents[doc_id][:50] + "..." if len(documents[doc_id]) > 50 else documents[doc_id]
        print(f"  {rank:<5}{doc_id:<8}{total_count:<7}{count_str:<35}{score:<12.4f}")
        results_data.append({
            "Rank": rank, "Doc ID": doc_id,
            "Total Matches": total_count,
            "Term Counts": count_str,
            "TF-IDF Score": round(score, 4),
            "Document Text": documents[doc_id]
        })

    print("\n  * Ranking is in DESCENDING order of TF-IDF Score (highest = Rank 1)")
    print("  * 'Total' column shows how many times all query terms together occurred.")
    print("  * 'Term Counts' shows per-stemmed-term occurrence in each document.")
    return pd.DataFrame(results_data), ranked, scores, query_tokens

print("\n\n--- SECTION 3: TEST QUERY RESULTS ---")
search_and_rank("artificial intelligence machine learning data", top_k=10)
search_and_rank("renewable energy solar wind power environment", top_k=10)
search_and_rank("cloud computing cybersecurity data network", top_k=10)
search_and_rank("education online learning digital libraries books", top_k=10)

def evaluate_pipeline():
    raw_words = re.sub(r'[^a-zA-Z\s]', '', " ".join(documents.values()).lower()).split()
    processed_words = " ".join(processed_corpus.values()).split()
    unique_raw = set(raw_words)
    processed_vocab = set(feature_names)
    reduction = ((len(unique_raw) - len(processed_vocab)) / len(unique_raw)) * 100
    print("\n\n--- SECTION 4: PIPELINE EVALUATION METRICS ---")
    print(f"  Total raw word count (before cleaning) : {len(raw_words)}")
    print(f"  Total stemmed tokens (after cleaning)  : {len(processed_words)}")
    print(f"  Unique raw vocabulary (before)         : {len(unique_raw)}")
    print(f"  Unique TF-IDF features (after)         : {len(processed_vocab)}")
    print(f"  Net vocabulary reduction               : {reduction:.2f} %")

evaluate_pipeline()

class IRSystem:
    def __init__(self, doc_dict):
        self.raw_documents = doc_dict
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.processed_corpus = {k: self._preprocess(v) for k, v in self.raw_documents.items()}
        self.doc_ids = list(self.processed_corpus.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(list(self.processed_corpus.values()))
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _preprocess(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        cleaned = []
        for w in text.split():
            if w not in self.stop_words and len(w) > 1:
                cleaned.append(self.stemmer.stem(w))
        return " ".join(cleaned)

    def _raw_counts(self, doc_text, q_tokens):
        tokens = re.sub(r'[^a-zA-Z\s]', '', doc_text.lower()).split()
        stemmed = [self.stemmer.stem(t) for t in tokens if len(t) > 1]
        per_term = {qt: stemmed.count(qt) for qt in q_tokens}
        return per_term, sum(per_term.values())

    def search(self, query, top_k=5):
        pv = self._preprocess(query)
        q_tokens = pv.split()
        q_vec = self.vectorizer.transform([pv])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            per_term, total = self._raw_counts(self.raw_documents[self.doc_ids[idx]], q_tokens)
            results.append({
                "Rank": rank,
                "Doc ID": self.doc_ids[idx],
                "Total Matches": total,
                "Term Counts": ", ".join([f"{t}:{c}" for t, c in per_term.items()]),
                "TF-IDF Score": round(scores[idx], 4),
                "Document": self.raw_documents[self.doc_ids[idx]]
            })
        return pd.DataFrame(results)

print("\n\n--- SECTION 5: IRSystem CLASS DEMONSTRATION ---")
ir = IRSystem(documents)
print("  Example: ir.search('blockchain decentralized ledger transactions', top_k=3)")
demo_result = ir.search("blockchain decentralized ledger transactions", top_k=3)
with pd.option_context('display.width', 200, 'display.max_columns', None, 'display.max_colwidth', 60):
    print(demo_result[['Rank', 'Doc ID', 'Total Matches', 'Term Counts', 'TF-IDF Score']].to_string(index=False))

print("\n\n--- SECTION 6: INTERACTIVE SEARCH MODE ---")
print("  Type any query and press ENTER. Type 'exit' or blank line to quit.\n")
while True:
    try:
        q = input("  Search query> ").strip()
    except EOFError:
        break
    if q.lower() in ('exit', 'quit', ''):
        print("  Exiting.")
        break
    res = ir.search(q, top_k=10)
    with pd.option_context('display.width', 220, 'display.max_columns', None, 'display.max_colwidth', 90):
        print(res[['Rank', 'Doc ID', 'Total Matches', 'Term Counts', 'TF-IDF Score', 'Document']].to_string(index=False))
    print()
