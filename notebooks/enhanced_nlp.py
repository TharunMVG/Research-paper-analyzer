import re
import spacy
import nltk
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import networkx as nx
import subprocess
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading SpaCy model 'en_core_web_md'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

try:
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error initializing sentence transformer: {e}")
    sentence_transformer = None

def clean_text(text: str) -> str:
    """
    Enhanced text cleaning with more sophisticated preprocessing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,;:!?()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Enhanced tokenization with stopword removal and lemmatization.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def get_pos_tags(text: str) -> List[Tuple[str, str, str]]:
    """
    Enhanced POS tagging with dependency parsing.
    Returns (word, POS tag, dependency role).
    """
    doc = nlp(text)
    return [(token.text, token.pos_, token.dep_) for token in doc]

def extract_named_entities(text: str) -> List[Tuple[str, str, str]]:
    """
    Enhanced named entity recognition with entity linking.
    Returns (entity text, label, knowledge base ID if available).
    """
    doc = nlp(text)
    return [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents]

def get_text_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence transformer.
    Falls back to TF-IDF if sentence transformer is not available.
    """
    if sentence_transformer is not None:
        return sentence_transformer.encode(texts)
    else:
        # Fallback to TF-IDF embeddings
        vectorizer = TfidfVectorizer(max_features=1000)
        return vectorizer.fit_transform(texts).toarray()

def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
    """
    Extract key phrases using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Get top scoring phrases
    top_indices = np.argsort(scores)[-top_n:]
    return [feature_names[i] for i in top_indices]

def build_knowledge_graph(text: str) -> nx.Graph:
    """
    Build a knowledge graph from the text using named entities and relationships.
    """
    doc = nlp(text)
    graph = nx.Graph()
    
    # Add entities as nodes
    for ent in doc.ents:
        graph.add_node(ent.text, type=ent.label_)
    
    # Add relationships as edges
    for token in doc:
        if token.dep_ in ["nsubj", "dobj", "pobj"]:
            head = token.head.text
            tail = token.text
            if head in graph and tail in graph:
                graph.add_edge(head, tail, relation=token.dep_)
    
    return graph

def visualize_embeddings(embeddings: np.ndarray, labels: List[str], title: str = "t-SNE Visualization"):
    """
    Enhanced visualization with clustering and better annotations.
    """
    if len(embeddings) == 0:
        print("No embeddings to visualize")
        return
        
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c='skyblue', edgecolors='k', alpha=0.7)
    
    # Annotate points with labels
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            xy=(reduced[i, 0], reduced[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
        )
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_research_sections(text: str) -> Dict[str, List[str]]:
    """
    Analyze research paper sections and extract key information.
    """
    sections = {
        "abstract": [],
        "introduction": [],
        "methodology": [],
        "results": [],
        "discussion": [],
        "conclusion": [],
        "limitations": [],
        "future_work": []
    }
    
  
    current_section = None
    for line in text.split('\n'):
        line = line.strip().lower()
        if any(section in line for section in sections.keys()):
            current_section = line
        elif current_section:
            sections[current_section].append(line)
    
    return sections


if __name__ == "__main__":
    sample_text = """
    This research paper discusses the application of deep learning in automated lip-reading.
    The methodology involves using convolutional neural networks to process video frames.
    Limitations include low accuracy in noisy environments and limited training data.
    Future work suggests exploring multimodal inputs and larger datasets.
    """
    
    print("Original Text:", sample_text)
    
    cleaned = clean_text(sample_text)
    print("\nCleaned Text:", cleaned)
    
    tokens = tokenize_text(cleaned)
    print("\nTokens:", tokens)
    
    pos_tags = get_pos_tags(sample_text)
    print("\nPOS Tags:", pos_tags)
    
    entities = extract_named_entities(sample_text)
    print("\nNamed Entities:", entities)
    
    key_phrases = extract_key_phrases(sample_text)
    print("\nKey Phrases:", key_phrases)
    
    sections = analyze_research_sections(sample_text)
    print("\nResearch Sections:", sections)
    
    texts = [sample_text, "Another research paper about machine learning."]
    embeddings = get_text_embeddings(texts)
    visualize_embeddings(embeddings, ["Paper 1", "Paper 2"])
