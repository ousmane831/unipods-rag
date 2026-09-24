from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Chargement du modèle...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Modèle chargé !")

texts = [
    "La réunion de vendredi commence à 15h.",
    "La deadline du projet est jeudi à 18h."
]

print("Création des embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=2,
    show_progress_bar=False
)

print("Embeddings créés !")
print("Nombre de textes :", len(embeddings))
print("Dimension :", len(embeddings[0]))
print("Premier vecteur :", embeddings[0][:5])