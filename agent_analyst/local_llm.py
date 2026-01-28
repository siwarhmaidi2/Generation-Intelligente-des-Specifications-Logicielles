"""
Client LLM local utilisant transformers pour charger Mistral 7B directement en mémoire.
Pas besoin d'Ollama ni d'API externe.
"""
import os
from typing import List, Dict, Any, Optional
import asyncio
from threading import Lock

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LocalMistralClient:
    """
    Client local pour Mistral 7B Instruct chargé directement en mémoire.
    Utilise transformers de Hugging Face.
    """

    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _model_lock = Lock()
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name: str | None = None):
        """
        Initialise le client local.
        
        Args:
            model_name: Nom du modèle Hugging Face. 
                       Par défaut, utilise un modèle léger selon la RAM disponible.
                       Options recommandées :
                       - "microsoft/Phi-3-mini-4k-instruct" (~2GB RAM) - très léger
                       - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (~1GB RAM) - ultra léger
                       - "mistralai/Mistral-7B-Instruct-v0.2" (~16GB RAM) - complet mais lourd
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers et torch sont requis pour le mode local. "
                "Installe-les avec: pip install transformers torch accelerate"
            )
        
        # Si aucun modèle spécifié, choisir automatiquement selon la RAM
        if model_name is None:
            model_name = os.getenv("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Charge le modèle et le tokenizer (une seule fois, partagé entre instances)."""
        with self._model_lock:
            if self._model is None or self._tokenizer is None:
                print(f"🔄 Chargement du modèle {self.model_name} sur {self._device}...")
                print("   (Premier chargement peut prendre quelques minutes)")
                
                # Charger le tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Charger le modèle avec optimisations pour économiser la RAM
                # Utiliser float16 sur GPU, float32 sur CPU
                dtype = torch.float16 if self._device == "cuda" else torch.float32
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self._device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                if self._device == "cpu":
                    self._model = self._model.to(self._device)
                
                print(f"✅ Modèle chargé avec succès sur {self._device}")

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Formate les messages selon le format du modèle.
        Supporte Mistral, Phi-3, TinyLlama et autres formats ChatML.
        """
        system_prompt = ""
        user_prompt = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
        
        # Détecter le format selon le modèle
        model_lower = self.model_name.lower()
        
        # Format Phi-3
        if "phi" in model_lower:
            if system_prompt:
                return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
            return f"<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
        
        # Format TinyLlama (ChatML)
        if "tinylama" in model_lower:
            if system_prompt:
                return f"<|system|>\n{system_prompt}<|user|>\n{user_prompt}<|assistant|>\n"
            return f"<|user|>\n{user_prompt}<|assistant|>\n"
        
        # Format Mistral Instruct (par défaut)
        if system_prompt:
            formatted = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        else:
            formatted = f"<s>[INST] {user_prompt} [/INST]"
        
        return formatted

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Génère une réponse à partir des messages.
        
        Args:
            messages: Liste de messages au format [{"role": "user", "content": "..."}, ...]
            temperature: Température pour la génération (0.1 = déterministe)
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            Texte généré
        """
        # Formater les messages pour Mistral
        prompt = self._format_messages(messages)
        
        # Tokeniser
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        
        # Générer
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Décoder la réponse (en excluant le prompt)
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Version asynchrone de generate (exécute dans un thread pool).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            messages,
            temperature,
            max_tokens
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Interface compatible avec LLMClient.
        Retourne un format similaire à OpenAI API.
        """
        content = self.generate(messages, temperature, max_tokens)
        return {
            "choices": [{
                "message": {
                    "content": content,
                    "role": "assistant"
                }
            }]
        }

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Version asynchrone de complete.
        """
        content = await self.agenerate(messages, temperature, max_tokens)
        return {
            "choices": [{
                "message": {
                    "content": content,
                    "role": "assistant"
                }
            }]
        }



