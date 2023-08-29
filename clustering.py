# Import libraries
import openai
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('~').expanduser() / '.env'
load_dotenv(dotenv_path=env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize GPT-4
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")  # Replace with the actual model name if it's different
model = AutoModel.from_pretrained("gpt2-large")

# gpt = GPT(engine="text-davinci-004", temperature=0.5)
words = ["happy", "sad", "angry"]

# Functions
def get_sentiment(word):
    prompt = f"What is the sentiment of the word '{word}'?"
    response = openai.Completion.create(
      engine="text-davinci-004",
      prompt=prompt,
      max_tokens=1
    )
    return response.choices[0].text.strip()

def get_gpt_vector(word):
    input_ids = tokenizer.encode(word, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids).last_hidden_state
    return output[0][-1].detach().numpy()

# Get Sentiments
words_sentiments = {}
for word in words:
    sentiment = get_sentiment(word)
    words_sentiments[word] = sentiment

# Get Vectors
word_vectors = {}
for word in words:
    vector = get_gpt_vector(word)
    word_vectors[word] = vector

# Clustering
words = list(words_sentiments.keys())
X = np.array(list(words_sentiments.values())).reshape(-1, 1)
kmeans = KMeans(n_clusters=3).fit(X)

# Initialize Graph
G = nx.Graph()

# Add Nodes with Cluster Labels
for word, cluster in zip(words, kmeans.labels_):
    G.add_node(word, cluster=cluster)

# Add Edges Based on Cosine Similarity
for word1 in words:
    for word2 in words:
        if word1 != word2:
            similarity = cosine_similarity([word_vectors[word1]], [word_vectors[word2]])[0][0]
            if similarity > 0.7:
                G.add_edge(word1, word2, weight=similarity)

# Draw Graph
nx.draw(G, with_labels=True)
plt.show()
