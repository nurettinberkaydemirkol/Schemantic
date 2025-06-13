# Schemantic

**A Python Library to Search Smarter Across Documents**

Schemantic is a fast, lightweight vector-based document clustering and retrieval library for Python — ideal for AI agents, RAG (Retrieval-Augmented Generation) systems, and semantic search pipelines.

---

## 🚀 Why Schemantic?

- 🧠 **AI Agent & RAG-Ready**  
  Easily integrates with LLM-based agents and retrieval-augmented generation workflows.

- 🔍 **Vector-Based Smart Clustering**  
  Automatically organizes documents into meaningful clusters using embedding vectors and different strategies.

- 🧩 **Flexible Clustering Algorithms**  
  Supports multiple clustering methods including `mean`, `knn`, and `l2`-based approaches for adaptable performance.

---

## 💡 Use Cases

- Semantic document search and filtering  
- Lightweight RAG prototypes and pipelines  
- Context retrieval for LLMs and chatbot agents  
- Fast information triage in document-heavy environments
## Build

Build this library by **Maturin**, first install library.
```bash
  pip install maturin
```
Then build Rust project
```bash
  cargo build
```
And build the library
```bash
  maturin develop
```
## Usage

```bash
from schemantic import VectorCube

data = [
    (0, [0.1] * 32, "first document"),
    (1, [0.9] * 32, "second document"),
    (2, [0.2, 0.1] * 16, "third document"),
]

cube = VectorCube(data, cluster_type="knn")
results = cube.query([0.87] * 32)

#['second document']
print(results)  
```
