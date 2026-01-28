# Guide rapide - Ollama avec Mistral quantifié

## 🚀 Installation en 3 étapes

### 1. Installer Ollama

Télécharge et installe depuis : https://ollama.com/download

Ollama démarre automatiquement après l'installation.

### 2. Télécharger Mistral 7B quantifié

```powershell
ollama pull mistral:7b-instruct-q4_K_M
```

**Temps estimé** : 10-20 minutes (modèle ~4GB)

### 3. Configurer le projet

```powershell
# Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt

# Créer le fichier .env (déjà configuré pour Ollama)
Copy-Item env.example .env
```

**C'est tout !** Le fichier `.env` est déjà configuré avec :
```env
LLM_PROVIDER=ollama
LLM_API_BASE=http://localhost:11434
LLM_MODEL=mistral:7b-instruct-q4_K_M
LLM_API_KEY=
```

## ✅ Vérification

```powershell
# Vérifier qu'Ollama fonctionne
ollama list

# Tu devrais voir : mistral:7b-instruct-q4_K_M

# Tester l'agent
python analyze_example.py
```

## 📊 Avantages de Mistral quantifié

- **RAM nécessaire** : ~5GB (au lieu de 16GB pour le modèle complet)
- **Qualité** : ⭐⭐⭐⭐⭐ (presque identique au modèle complet)
- **Taille** : ~4GB (au lieu de 13GB)
- **Vitesse** : Rapide, surtout avec GPU NVIDIA

## 🔧 Dépannage

### Ollama ne démarre pas
- Relance l'application Ollama depuis le menu Démarrer
- Vérifie qu'aucun autre processus n'utilise le port 11434

### Modèle non trouvé
```powershell
ollama pull mistral:7b-instruct-q4_K_M
```

### Erreur "Connection refused"
- Vérifie qu'Ollama est lancé : `ollama list`
- Vérifie que `LLM_API_BASE=http://localhost:11434` dans `.env`

## 💡 Autres modèles disponibles

Si tu veux essayer d'autres modèles :

```powershell
# Phi-3 Mini (très léger, ~2GB RAM)
ollama pull phi3:mini

# Mistral complet (non quantifié, ~16GB RAM)
ollama pull mistral:7b-instruct
```

Puis change `LLM_MODEL` dans `.env` en conséquence.
