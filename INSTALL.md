# Installation (Windows) – Ollama + Mistral quantifié + OCR

### 1) Ollama + modèle

Installe Ollama puis :

```powershell
ollama pull mistral:7b-instruct-q4_K_M
ollama list
```

### 2) Python (venv)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item env.example .env
```

### 3) OCR (Tesseract)

Option `winget` :

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

Si `tesseract.exe` n’est pas dans le PATH, ajoute dans `.env` :

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 4) Lancer

```powershell
python analyze_docs.py --lang fra
python analyze_images.py --lang fra
```


