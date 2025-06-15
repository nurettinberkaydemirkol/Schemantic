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
import openai
from schemantic import VectorCube, same_search
from pprint import pprint
import time
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embed(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

records = [
    # Cats
    ("1",  "The cat sits by the fire."),
    ("6",  "A kitten curls up next to the furnace."),
    ("7",  "The feline lounges beside the hearth."),
    ("8",  "A cat naps on the warm rug in front of the fireplace."),
    ("9",  "The pet cat dozes near the living room heater."),
    # Dogs
    ("2",  "A dog sleeps on the sofa."),
    ("10", "The hound slumbers on the couch."),
    ("11", "My puppy is dozing on the living room chair."),
    ("12", "The dog lies sprawled across the couch cushions."),
    ("13", "A canine rests on the comfortable sofa."),
    # Weather
    ("3",  "Weather in Izmit will be sunny."),
    ("4",  "Izmit weather: 20 degrees and sunny."),
    ("14", "Tomorrow Izmit will see clear skies with highs of 19°C."),
    ("15", "Izmit is expected to be bright and warm today."),
    ("16", "Sunny weather and 19 degrees are forecast for Izmit."),
    # Other
    ("5",  "The dog chases the cat."),
    ("17", "She writes code at her desk all night."),
    ("18", "Machine learning models detect patterns in data."),
    ("19", "Deep neural networks are used for image recognition."),
    ("20", "He enjoys painting landscapes in his free time."),
]

print("🔄 Creating embeddings...")
data = [
    (i, get_openai_embed(text), label)
    for i, (label, text) in enumerate(records)
]

# Same Search

start_time = time.time()

matches = same_search(data, threshold=0.50, brute_force=False)

end_time = time.time()
elapsed = end_time - start_time

print(f"⏱️ same_search completed in {elapsed:.4f} seconds.")

# Query Search

cube = VectorCube(data, cluster_type="l1")

query_text = "weather"
query_embed = get_openai_embed(query_text)

start = time.time()
results = cube.query(query_embed, query_type="cosine")
end = time.time()

print("Query results:")
pprint(results)
print(f"\n⏱️ Query completed in {end - start + elapsed:.5f} seconds.")
```
