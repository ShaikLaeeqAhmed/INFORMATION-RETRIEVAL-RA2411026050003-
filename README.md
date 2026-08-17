# INFORMATION-RETRIEVAL-RA2411026050003-

# 🔎 TF-IDF Document Search Engine

### Information Retrieval System using TF-IDF & Cosine Similarity

A lightweight and explainable **Information Retrieval (IR) system** that searches a collection of text documents using **TF-IDF vectorization** and **Cosine Similarity**. The system preprocesses documents and user queries, builds a TF-IDF feature matrix, calculates similarity scores, and returns documents ranked by relevance.

---

## 📌 Overview

This project implements a document search engine as part of an **Information Retrieval (IR)** assignment.

The system accepts:

* 📄 A corpus of text documents
* 🔍 A user search query
* 🔢 An optional number of top results

It then performs text preprocessing, TF-IDF feature extraction, cosine similarity calculation, and relevance-based ranking.

The final output provides:

* Rank
* Document ID
* Total query-term matches
* Individual term counts
* TF-IDF similarity score
* Retrieved document content

---

## ✨ Key Features

* 🔎 **Keyword-based document search**
* 🧹 **Six-step text preprocessing pipeline**
* 📊 **TF-IDF feature matrix generation**
* 📐 **Cosine similarity-based ranking**
* 🏆 **Descending relevance ranking**
* 🔢 **Per-term occurrence counting**
* 📈 **Pipeline vocabulary evaluation**
* 🧩 **Reusable `IRSystem` Python class**
* 📂 **Dynamic document loading from `.txt` files**
* 💬 **Interactive search mode**
* 📋 **Structured results using Pandas DataFrame**
* ⚡ **Efficient repeated querying without rebuilding the TF-IDF matrix**

---

## 🧠 System Architecture

```text
                ┌──────────────────────┐
                │   Text Documents     │
                │  documents_dataset/  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Document Loading    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Text Preprocessing   │
                │                      │
                │ • Lowercase          │
                │ • Remove punctuation │
                │ • Tokenization       │
                │ • Stop-word removal  │
                │ • Remove short words │
                │ • Porter stemming    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ TF-IDF Vectorization │
                └──────────┬───────────┘
                           │
                           ▼
┌──────────────┐  ┌──────────────────────┐
│ User Query   │─►│ Query Preprocessing  │
└──────────────┘  └──────────┬───────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Query Vectorization │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Cosine Similarity   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Relevance Ranking   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Ranked Search       │
                  │ Results             │
                  └─────────────────────┘
```

---

## ⚙️ How It Works

### 1. Document Loading

The system automatically scans the `documents_dataset/` directory and loads all `.txt` files.

This allows the document collection to be modified without changing the source code.

```text
documents_dataset/
├── Doc_1.txt
├── Doc_2.txt
├── Doc_3.txt
├── ...
└── Doc_20.txt
```

The current dataset contains **20 text documents** covering topics such as:

* Artificial Intelligence
* Machine Learning
* Cloud Computing
* Blockchain
* Cybersecurity
* Renewable Energy
* Solar Power
* Robotics
* Natural Language Processing
* Computer Vision
* Digital Libraries
* Online Learning
* Climate Change
* Biodiversity
* Internet of Things

---

## 🧹 Text Preprocessing

Both documents and queries pass through the same six-step preprocessing pipeline.

### Step 1 — Lowercasing

Converts all text to lowercase.

```text
"Artificial Intelligence"
          ↓
"artificial intelligence"
```

### Step 2 — Punctuation Removal

Non-alphabetic characters are removed using regular expressions.

### Step 3 — Tokenization

Text is split into individual tokens.

### Step 4 — Stop-word Removal

Common English words such as:

```text
the, is, and, for, of
```

are removed using NLTK.

### Step 5 — Short-token Removal

Tokens with one or fewer characters are discarded.

### Step 6 — Porter Stemming

Words are reduced to their stem/root form.

```text
computing
computed
computer

        ↓

     comput
```

---

## 📊 TF-IDF

The system uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to represent documents numerically.

### Term Frequency

Measures how frequently a term occurs within a document.

### Inverse Document Frequency

Assigns higher importance to terms that occur in fewer documents.

The project uses:

```text
IDF(t) = ln(N / df(t)) + 1
```

where:

* `N` = total number of documents
* `df(t)` = number of documents containing term `t`

### TF-IDF Weight

```text
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

Terms that are relatively frequent in a document but rare across the corpus receive higher weights.

---

## 📐 Cosine Similarity

After converting the documents and query into TF-IDF vectors, the system calculates their similarity using **Cosine Similarity**.

```text
                 d · q
Cosine = ───────────────────
          ||d|| × ||q||
```

The resulting score ranges from **0 to 1**.

* `1.0` → highly similar
* `0.0` → no similarity

The documents are then sorted in descending order of similarity.

```text
Highest Score → Rank 1
      ↓
Lower Score  → Rank 2
      ↓
Lower Score  → Rank 3
      ↓
      ...
```

---

## 🔍 Query Processing

The user query follows the same preprocessing pipeline as the documents.

For example:

```text
User Query:
artificial intelligence machine learning data

             ↓

Preprocessed Query:
artifici intellig machin learn data
```

The processed query is converted into a TF-IDF vector using the already-fitted vectorizer.

> The query uses `transform()` instead of `fit_transform()` so that the existing document vocabulary and feature-space alignment are preserved.

---

## 🏆 Example Search

### Query

```text
artificial intelligence machine learning data
```

### Example Result

```text
Rank   Doc ID   Total   TF-IDF Score
------------------------------------
1      Doc 4      6        0.5769
2      Doc 1      3        0.2543
3      Doc 15     2        0.1746
4      Doc 17     1        0.0987
5      Doc 10     1        0.0907
```

`Doc 4` receives the highest score because it contains strong matches for terms related to artificial intelligence and machine learning.

---

## 🧪 Example Queries

The project evaluates several search scenarios, including:

```text
1. Artificial Intelligence & Machine Learning

2. Renewable Energy

3. Cloud Computing & Cybersecurity

4. Education & Digital Libraries

5. Blockchain & Distributed Ledger
```

For the blockchain demonstration, `Doc 11` is ranked first because it directly discusses blockchain as a decentralized distributed ledger.

---

## 📈 Pipeline Evaluation

The project also evaluates the effect of preprocessing on the vocabulary.

The evaluation compares:

```text
Raw Vocabulary
      ↓
Preprocessed Vocabulary
      ↓
TF-IDF Feature Vocabulary
```

Stop-word removal and stemming reduce the vocabulary size and consequently reduce the dimensionality of the TF-IDF representation.

The report observes approximately **20% vocabulary reduction** after preprocessing.

---

## 🧩 IRSystem Class

The complete search engine is encapsulated in a reusable Python class:

```python
class IRSystem:
```

### Main Components

| Component       | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `__init__()`    | Initializes the search engine and TF-IDF vectorizer |
| `_preprocess()` | Performs text preprocessing                         |
| `_raw_counts()` | Calculates query-term occurrences                   |
| `search()`      | Searches and ranks documents                        |
| `tfidf_matrix`  | Stores the fitted TF-IDF matrix                     |
| `feature_names` | Stores vocabulary features                          |

The TF-IDF vectorizer is fitted once when the class is initialized, allowing multiple queries to be processed efficiently.

---

## 🛠️ Technologies Used

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Core implementation                        |
| **NLTK**         | Stop-word removal and Porter stemming      |
| **Scikit-learn** | TF-IDF vectorization and cosine similarity |
| **Pandas**       | Structured search-result display           |
| **NumPy**        | Numerical operations and ranking           |
| **ReportLab**    | Project report generation                  |

---

## 📁 Project Structure

```text
TF-IDF-Document-Search-Engine/
│
├── documents_dataset/
│   ├── Doc_1.txt
│   ├── Doc_2.txt
│   ├── Doc_3.txt
│   ├── ...
│   └── Doc_20.txt
│
├── <IR search engine Python file>
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TF-IDF-Document-Search-Engine.git
```

### 2. Navigate to the Project

```bash
cd TF-IDF-Document-Search-Engine
```

### 3. Install Dependencies

```bash
pip install nltk scikit-learn pandas numpy reportlab
```

### 4. Download NLTK Stop Words

Run Python and execute:

```python
import nltk
nltk.download('stopwords')
```

---

## ▶️ Running the Project

Place the required `.txt` documents inside:

```text
documents_dataset/
```

Then execute the Python search-engine script.

The program loads the documents, preprocesses the corpus, builds the TF-IDF matrix, and allows the user to enter search queries.

Example:

```text
USER QUERY:
blockchain decentralized ledger transactions

PREPROCESSED QUERY:
blockchain decentral ledger transact

RANK   DOC ID   TOTAL MATCHES   TF-IDF SCORE
1      Doc 11        4             0.4592
2      Doc 9         0             0.0000
3      Doc 7         0             0.0000
```

---

## 📋 Output

The search results provide an explainable ranking table:

```text
Rank
Doc ID
Total Matches
Term Counts
TF-IDF Score
Document
```

This makes it possible to understand not only **which document was retrieved**, but also **why it received its ranking**.

---

## 💡 Advantages

* Simple and lightweight implementation
* No external database required
* Easy-to-modify document corpus
* Explainable ranking mechanism
* Efficient repeated searches
* Uses standard Information Retrieval techniques
* Suitable for educational and academic applications
* Demonstrates practical TF-IDF implementation

---

## 🔮 Future Enhancements

The current system can be extended with:

* 🌐 Web-based search interface
* 🗄️ Database-backed document storage
* 🔤 Improved lemmatization
* 🧠 Semantic search using word embeddings
* 🤖 Transformer-based retrieval
* 🔎 Fuzzy query matching
* 📊 Search-result visualization
* ⚡ Large-scale corpus optimization
* 📄 PDF and DOCX document support
* 🏷️ Document categorization
* 📈 Advanced IR evaluation metrics such as Precision, Recall, MAP, and NDCG

---

## 🎯 Learning Outcomes

This project demonstrates practical understanding of:

* Information Retrieval
* Text preprocessing
* Tokenization
* Stop-word removal
* Stemming
* Term Frequency
* Inverse Document Frequency
* TF-IDF representation
* Vector-space information retrieval
* Cosine similarity
* Document ranking
* Query processing
* Python-based IR system design

---

## 📚 Project Reference

**Project:** TF-IDF based Document Search Engine with Cosine Similarity Ranking
**Course:** Information Retrieval (IR)
**Assignment:** TF-IDF Search Engine Implementation
**Author:** Shaik Laeeq Ahmed
**Date:** August 2026

---

## 📄 License

This project is developed for **academic and educational purposes**.

You are free to study, modify, and extend the implementation for learning and research purposes.

---

## ⭐ Acknowledgement

This project was developed as part of an **Information Retrieval course assignment**, demonstrating the implementation of a classical vector-space document retrieval system using TF-IDF and cosine similarity.
