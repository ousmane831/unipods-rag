from unipods_rag.config import Settings
from unipods_rag.generator import get_generator


settings = Settings.from_env()

print("Backend :", settings.llm_backend)
print("Modèle  :", settings.local_model)

generator = get_generator(settings)

print("Generator :", generator.name)

response = generator._client.chat(
    model=generator.model,
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant qui répond clairement en français.",
        },
        {
            "role": "user",
            "content": "Explique en deux phrases ce qu'est un système RAG.",
        },
    ],
)

print("\n--- RÉPONSE BRUTE ---")
print(response)

print("\n--- CONTENT ---")
print(repr(response["message"].get("content")))

print("\n--- THINKING ---")
print(repr(response["message"].get("thinking")))