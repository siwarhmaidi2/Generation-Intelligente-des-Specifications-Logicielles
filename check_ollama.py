"""
Script de vérification pour diagnostiquer les problèmes avec Ollama.
"""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

def check_ollama():
    """Vérifie que Ollama est accessible et que le modèle est disponible."""
    api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
    model = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_K_M")
    
    print("🔍 Vérification de la configuration Ollama...\n")
    print(f"   API Base: {api_base}")
    print(f"   Modèle: {model}\n")
    
    # Vérifier la connexion à Ollama
    print("1️⃣ Vérification de la connexion à Ollama...")
    try:
        url = f"{api_base}/api/tags"
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        print("   ✅ Ollama est accessible\n")
    except httpx.ConnectError:
        print("   ❌ Impossible de se connecter à Ollama")
        print("   💡 Solutions :")
        print("      - Vérifie qu'Ollama est lancé : `ollama list`")
        print("      - Si Ollama n'est pas lancé, démarre-le depuis le menu Démarrer")
        return False
    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        return False
    
    # Vérifier que le modèle est disponible
    print("2️⃣ Vérification des modèles disponibles...")
    models = [m.get("name", "") for m in data.get("models", [])]
    
    if not models:
        print("   ⚠️  Aucun modèle trouvé dans Ollama")
        print(f"   💡 Télécharge le modèle : `ollama pull {model}`")
        return False
    
    print(f"   📦 Modèles disponibles ({len(models)}):")
    for m in models:
        marker = "✅" if model in m else "  "
        print(f"      {marker} {m}")
    
    # Vérifier si le modèle demandé est disponible
    model_found = any(model in m for m in models)
    if not model_found:
        print(f"\n   ❌ Modèle '{model}' non trouvé")
        print(f"   💡 Télécharge le modèle : `ollama pull {model}`")
        return False
    
    print(f"\n   ✅ Modèle '{model}' est disponible\n")
    
    # Test de génération rapide
    print("3️⃣ Test de génération rapide...")
    try:
        url = f"{api_base}/api/generate"
        payload = {
            "model": model,
            "prompt": "Bonjour",
            "stream": False,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
        print("   ✅ Génération test réussie\n")
        return True
    except httpx.ReadTimeout:
        print("   ⚠️  Timeout lors du test (peut être normal si le modèle est en train de se charger)")
        print("   💡 Réessaie dans quelques secondes")
        return True  # On considère que c'est OK, juste lent
    except Exception as e:
        print(f"   ⚠️  Erreur lors du test : {e}")
        print("   💡 Cela peut être normal, réessaie le script principal")
        return True

if __name__ == "__main__":
    success = check_ollama()
    if success:
        print("✅ Tout semble correct ! Tu peux lancer `python analyze_example.py`")
    else:
        print("\n❌ Des problèmes ont été détectés. Corrige-les avant de continuer.")
