import re
import math
from collections import Counter
import pandas as pd

def preprocess_simple(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return [w for w in text.split() if len(w) > 1]

def compute_tf(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return {w: counts[w] / total for w in counts}

def compute_idf(docs_tokens):
    N = len(docs_tokens)
    all_words = set()
    for tokens in docs_tokens:
        all_words.update(tokens)
    idf = {}
    for word in all_words:
        doc_freq = sum(1 for tokens in docs_tokens if word in tokens)
        idf[word] = math.log(N / doc_freq) + 1
    return idf

def compute_tfidf_matrix(docs_tokens, vocab):
    idf = compute_idf(docs_tokens)
    rows = []
    for i, tokens in enumerate(docs_tokens):
        tf = compute_tf(tokens)
        row = {w: round(tf.get(w, 0) * idf.get(w, 0), 6) for w in vocab}
        rows.append(row)
    return pd.DataFrame(rows, index=[f"Doc {i+1}" for i in range(len(docs_tokens))]), idf

def cosine_similarity_manual(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

sample_docs = {
    "Doc 1": "artificial intelligence computer systems visual perception speech recognition decision making",
    "Doc 2": "cloud computing on demand servers storage databases networking software analytics",
    "Doc 3": "climate change global temperatures fossil fuels deforestation greenhouse gas emissions",
    "Doc 4": "machine learning artificial intelligence patterns data performance explicitly programmed",
}

sample_queries = [
    "artificial intelligence machine learning data",
    "cloud computing servers storage networking",
    "climate change deforestation greenhouse emissions",
]

print("=" * 72)
print("  STEP-BY-STEP TF-IDF NUMERICAL CALCULATOR")
print("  (For Exam Practice and Formula Verification)")
print("=" * 72)

print("\n[1] CORPUS DOCUMENTS:")
for k, v in sample_docs.items():
    print(f"     {k}: {v}")

docs_tokens = {k: preprocess_simple(v) for k, v in sample_docs.items()}
print("\n[2] TOKENS AFTER CLEANING:")
for k, v in docs_tokens.items():
    print(f"     {k}: {v}")

vocab = sorted(set(t for tokens in docs_tokens.values() for t in tokens))
print(f"\n[3] VOCABULARY (V = {len(vocab)} terms):")
print(f"     {vocab}")

docs_tokens_list = list(docs_tokens.values())
tfidf_df, idf_vals = compute_tfidf_matrix(docs_tokens_list, vocab)

N = len(docs_tokens_list)
print(f"\n[4] TERM FREQUENCY (TF) PER DOCUMENT (tf = count(t,d)/|d|):")
for i, (doc, tokens) in enumerate(docs_tokens.items()):
    tf = compute_tf(tokens)
    print(f"\n     {doc}  (total tokens = {len(tokens)}):")
    for w in vocab:
        raw = Counter(tokens).get(w, 0)
        print(f"       TF({w:<14}) = {raw}/{len(tokens)} = {tf.get(w, 0):.4f}")

print(f"\n[5] INVERSE DOCUMENT FREQUENCY (IDF) PER TERM (N = {N}):")
print(f"     Formula: IDF(t) = ln(N / df(t)) + 1")
for w in sorted(idf_vals.keys()):
    doc_freq = sum(1 for tokens in docs_tokens_list if w in tokens)
    print(f"     IDF({w:<14}) = ln({N}/{doc_freq}) + 1 = {idf_vals[w]:.6f}")

print("\n[6] TF-IDF MATRIX (elementwise TF * IDF):")
with pd.option_context('display.width', 220, 'display.max_columns', None,
                       'display.float_format', lambda x: f"{x:.4f}"):
    print(tfidf_df.to_string())

print("\n" + "=" * 72)
print("  QUERY PROCESSING + DESCENDING RANKING")
print("  (shows term occurrence counts + TF-IDF score)")
print("=" * 72)

for qi, query in enumerate(sample_queries, 1):
    q_tokens = preprocess_simple(query)
    print(f"\n--- Query {qi}: '{query}' ---")
    print(f"     Cleaned tokens: {q_tokens}")

    q_tf = compute_tf(q_tokens)
    q_vec = [round(q_tf.get(w, 0) * idf_vals.get(w, 0), 6) for w in vocab]
    print(f"\n     Query TF-IDF vector (per vocab term):")
    for w, v in zip(vocab, q_vec):
        if v > 0:
            print(f"       {w:>16s} : {v:.6f}")

    print(f"\n     Per-document occurrence counts + cosine score:")
    ranked = []
    for doc_name in sample_docs.keys():
        doc_vec = tfidf_df.loc[doc_name].tolist()
        doc_tokens = docs_tokens[doc_name]
        per_term_counts = {qt: Counter(doc_tokens).get(qt, 0) for qt in q_tokens}
        total_matches = sum(per_term_counts.values())
        score = cosine_similarity_manual(doc_vec, q_vec)
        ranked.append((doc_name, score, total_matches, per_term_counts, sample_docs[doc_name]))
        print(f"       {doc_name} | Total={total_matches} | {per_term_counts}")
        print(f"              cosine(Q, {doc_name}) = {score:.6f}")

    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\n     FINAL RANKING (DESCENDING BY TF-IDF SCORE, Rank 1 = Highest):")
    print(f"     {'Rank':<5}{'Doc':<8}{'Total':<8}{'Per-Term Counts':<42}{'Score':<10}")
    print(f"     {'-'*72}")
    for rank, (d, s, t, pc, txt) in enumerate(ranked, 1):
        print(f"     {rank:<5}{d:<8}{t:<8}{str(pc):<42}{s:<10.4f}")

print("\n" + "=" * 72)
print("  INTERACTIVE MODE - Enter your own docs and query")
print("=" * 72)

try:
    print("\n  How many documents? (2-5 recommended)")
    try:
        n = int(input("  > ").strip() or "3")
    except EOFError:
        raise KeyboardInterrupt
    if not (2 <= n <= 10):
        n = 3
        print(f"  (using default n={n})")

    custom_docs = {}
    for i in range(1, n + 1):
        print(f"  Doc {i} text: ", end="", flush=True)
        try:
            text = input().strip()
        except EOFError:
            break
        if not text:
            text = f"default document about topic number {i} for testing"
        custom_docs[f"Doc {i}"] = text

    print("  Query text: ", end="", flush=True)
    try:
        custom_query = input().strip() or "example query"
    except EOFError:
        custom_query = "example query"

    c_tokens = {k: preprocess_simple(v) for k, v in custom_docs.items()}
    c_vocab = sorted(set(t for tokens in c_tokens.values() for t in tokens))
    c_tfidf, c_idf = compute_tfidf_matrix(list(c_tokens.values()), c_vocab)
    q_clean = preprocess_simple(custom_query)
    q_tf = compute_tf(q_clean)
    q_vec = [round(q_tf.get(w, 0) * c_idf.get(w, 0), 6) for w in c_vocab]

    print("\n  CUSTOM TF-IDF MATRIX:")
    with pd.option_context('display.width', 220, 'display.max_columns', None,
                           'display.float_format', lambda x: f"{x:.4f}"):
        print(c_tfidf.to_string())

    ranked = []
    for dn in custom_docs.keys():
        dv = c_tfidf.loc[dn].tolist()
        per = {qt: Counter(c_tokens[dn]).get(qt, 0) for qt in q_clean}
        ranked.append((dn, cosine_similarity_manual(dv, q_vec), sum(per.values()), per))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  RANKED RESULTS FOR QUERY '{custom_query}':")
    print(f"  {'Rank':<5}{'Doc ID':<8}{'Total Matches':<15}{'Per-Term Counts':<38}{'TF-IDF Score':<10}")
    print(f"  {'-'*75}")
    for rank, (d, s, t, pc) in enumerate(ranked, 1):
        print(f"  {rank:<5}{d:<8}{t:<15}{str(pc):<38}{s:<10.4f}")
except (EOFError, KeyboardInterrupt):
    print("\nGoodbye!")
