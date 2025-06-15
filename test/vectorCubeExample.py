import openai
from schemantic import VectorCube
from pprint import pprint
import time
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embed(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

records = [
    ("1", "The cat quietly sits on the warm mat near the fireplace."),
    ("2", "A quick brown fox effortlessly jumps over a sleepy dog."),
    ("3", "Deep learning models are revolutionizing artificial intelligence."),
    ("4", "Today’s weather is 19 degrees with mild winds and cloudy skies."),
    ("5", "Neural networks are at the core of many deep learning models."),
    ("6", "The temperature today will reach 19 degrees with sunny intervals."),
    ("7", "She has been writing a science fiction book for over a year."),
    ("8", "A clever fox darted past the lazy dog lying in the sun."),
    ("9", "Machine learning has numerous applications across industries."),
    ("10", "Computer vision is one of the key areas of deep learning."),
    ("11", "Reinforcement learning enables agents to learn from feedback."),
    ("12", "Natural language processing helps computers understand text.")
]

print("🔄 Creating embeddings...")
data = [
    (i, get_openai_embed(text), label)
    for i, (label, text) in enumerate(records)
]

cube = VectorCube(data, cluster_type="l2")

query_text = "whats the weather"
query_embed = get_openai_embed(query_text)

start = time.time()
results = cube.query(query_embed, query_type="l2")
end = time.time()

print("Query results:")
pprint(results)
print(f"\n⏱️ Query completed in {end - start:.4f} seconds.")