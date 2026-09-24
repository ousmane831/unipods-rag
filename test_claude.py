from unipods_rag.config import Settings
from unipods_rag.generator import get_generator

settings = Settings.from_env()

print("Backend :", settings.llm_backend)
print("Modèle :", settings.llm_model)
print("Clé présente :", bool(settings.anthropic_api_key))

generator = get_generator(settings)

print("Generator :", generator.name)

response = generator._call(
    "You are a helpful assistant.",
    "Say only: Claude is connected."
)

print("\nRÉPONSE:")
print(response)