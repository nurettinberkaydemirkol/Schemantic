import os
from dotenv import load_dotenv
import openai
from schemantic import same_search
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embed(text, model="text-embedding-3-large"):
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

data = [
    (i, get_openai_embed(text), label)
    for i, (label, text) in enumerate(records)
]

start_time = time.time()

matches = same_search(data, threshold=0.50, brute_force=False)

end_time = time.time()
elapsed = end_time - start_time

for a, b in matches:
    print(f"{a[0]} ↔ {b[0]}")
    print(f"• {a[2]}")
    print(f"• {b[2]}\n")

print(f"⏱️ same_search completed in {elapsed:.4f} seconds.")