## Agent Analyst – Mistral 7B quantifié (Ollama) + OCR + Chunking

### Objectif
Analyser des **PDF** (dans `docs/`) et des **images** (dans `images/`) en :
- extraction texte (PDF texte) + **OCR** si nécessaire
- **découpage en morceaux (chunking)** pour les longs documents
- analyse par **Mistral 7B quantifié** (`mistral:7b-instruct-q4_K_M`) via **Ollama**
- génération d’un **JSON** structuré dans `results/` (même nom que le fichier source)

### Prérequis
- **Ollama** installé et lancé
- Modèle :

```powershell
ollama pull mistral:7b-instruct-q4_K_M
```

- **Tesseract OCR** (si tu veux l’OCR sur images / PDF scannés)

### Installation Python

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item env.example .env
```

### Traitement dynamique (batch)

#### PDF (et images éventuellement dans `docs/`)

```powershell
python analyze_docs.py --lang fra --chunk-words 450 --overlap-words 60 --min-words-to-chunk 600
```

#### Images dans `images/`

```powershell
python analyze_images.py --lang fra
```

#### Options
- `--overwrite` : écraser les JSON existants
- `--recursive` : parcourir les sous-dossiers







