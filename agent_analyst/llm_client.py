import os
from typing import List, Dict, Any

import httpx
from pydantic import BaseModel

# Import conditionnel du client local
try:
    from .local_llm import LocalMistralClient
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMClient:
    """
    Client générique pour appeler Mistral 7B Instruct via :
    - Local (transformers) : provider="local" - RECOMMANDÉ ✅
    - Ollama local (/api/chat) : provider="ollama"
    - Mistral AI API (cloud) : provider="mistral"
    - API compatible OpenAI : provider="openai"
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_K_M")
        
        # Timeout par défaut : plus long pour Ollama (génération peut être lente, surtout pour textes longs)
        if timeout is None:
            timeout_env = os.getenv("LLM_TIMEOUT")
            if timeout_env:
                self.timeout = float(timeout_env)
            elif self.provider == "ollama":
                self.timeout = 1800.0  # 30 minutes pour Ollama (textes très longs peuvent prendre beaucoup de temps)
            else:
                self.timeout = 60.0  # 1 minute pour les autres
        else:
            self.timeout = timeout

        # Pour le mode local (transformers) - par défaut
        if self.provider == "local":
            if not LOCAL_LLM_AVAILABLE:
                raise ImportError(
                    "Le mode local nécessite transformers et torch. "
                    "Installe-les avec: pip install transformers torch accelerate"
                )
            self.local_client = LocalMistralClient(model_name=self.model)
        # Pour Ollama (local, sans API key)
        elif self.provider == "ollama":
            self.api_base = (api_base or os.getenv("LLM_API_BASE", "http://localhost:11434")).rstrip("/")
        # Pour Mistral AI (cloud, nécessite API key)
        elif self.provider == "mistral":
            self.api_base = "https://api.mistral.ai"
            if not self.api_key:
                raise ValueError(
                    "LLM_API_KEY est requis pour Mistral AI. "
                    "Obtenez une clé gratuite sur https://console.mistral.ai/"
                )
        # Pour API OpenAI-compatible
        else:
            self.api_base = (api_base or os.getenv("LLM_API_BASE", "")).rstrip("/")
            if not self.api_base:
                raise ValueError(
                    "LLM_API_BASE n'est pas défini. "
                    "Définissez la variable d'environnement LLM_API_BASE."
                )

    async def acomplete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Appel asynchrone à l'API du provider.
        Retourne la réponse JSON brute.
        """
        if self.provider == "local":
            # Convertir ChatMessage en dict pour le client local
            messages_dict = [{"role": m.role, "content": m.content} for m in messages]
            return await self.local_client.acomplete(
                messages=messages_dict,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif self.provider == "mistral":
            return await self._acomplete_mistral(messages=messages, temperature=temperature, max_tokens=max_tokens)
        elif self.provider == "ollama":
            return await self._acomplete_ollama(messages=messages, temperature=temperature, max_tokens=max_tokens)

        return await self._acomplete_openai(messages=messages, temperature=temperature, max_tokens=max_tokens)

    async def _acomplete_mistral(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        API Mistral AI : https://docs.mistral.ai/api/
        Endpoint: https://api.mistral.ai/v1/chat/completions
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.api_base}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _acomplete_openai(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        API compatible OpenAI (/v1/chat/completions)
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.api_base}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _acomplete_ollama(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Ollama: POST {base}/api/chat
        https://github.com/ollama/ollama/blob/main/docs/api.md
        """
        # Estimer la longueur du texte d'entrée
        input_text = " ".join([m.content for m in messages])
        input_length = len(input_text.split())
        
        # Réglages perf Ollama (optionnels via .env)
        # - OLLAMA_NUM_PREDICT: limite de tokens générés (plus petit = plus rapide)
        # - OLLAMA_NUM_CTX: taille contexte (plus petit = plus rapide et moins RAM)
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", str(max_tokens)))
        num_ctx_env = os.getenv("OLLAMA_NUM_CTX")
        num_ctx = int(num_ctx_env) if num_ctx_env else None

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,  # Limiter la longueur de la réponse
            },
        }
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx
        url = f"{self.api_base}/api/chat"
        
        # Timeout adaptatif selon la longueur du texte
        # Pour les textes longs, on augmente le timeout de manière plus généreuse
        # Estimation : ~1 seconde par 10 mots pour la génération + overhead
        estimated_timeout = min(
            self.timeout,
            max(600.0, input_length * 1.5)  # Au moins 10 min, ~1.5s par mot pour textes longs
        )
        timeout = httpx.Timeout(estimated_timeout, connect=10.0)
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                print(f"🔄 Envoi de la requête à Ollama (modèle: {self.model})...")
                if input_length > 1000:
                    estimated_minutes = int(estimated_timeout / 60)
                    print(f"   ⏱️  Texte très long détecté (~{input_length} mots)")
                    print(f"   ⏱️  Temps estimé : {estimated_minutes}-{estimated_minutes+5} minutes")
                    print(f"   💡 Pour les textes très longs, considère les diviser en plusieurs parties")
                elif input_length > 500:
                    print(f"   ⏱️  Texte long détecté (~{input_length} mots) - cela peut prendre 5-15 minutes")
                else:
                    print("   (Cela peut prendre 1-2 minutes pour la première génération)")
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Impossible de se connecter à Ollama à {self.api_base}.\n"
                f"Vérifie que Ollama est lancé : `ollama list`\n"
                f"Si Ollama n'est pas lancé, démarre-le depuis le menu Démarrer."
            ) from e
        except httpx.ReadTimeout as e:
            timeout_minutes = int(estimated_timeout / 60)
            raise TimeoutError(
                f"Timeout lors de la génération avec Ollama (modèle: {self.model}).\n"
                f"La génération prend trop de temps (> {timeout_minutes} minutes).\n"
                f"Pour les textes très longs (~{input_length} mots), cela peut être normal.\n\n"
                f"💡 Solutions :\n"
                f"1. Augmente le timeout dans .env : LLM_TIMEOUT=3600 (60 minutes)\n"
                f"2. Divise ton texte en plusieurs parties plus courtes (< 500 mots chacune)\n"
                f"3. Vérifie que tu as assez de RAM disponible (le modèle peut être lent si RAM saturée)\n"
                f"4. Vérifie que le modèle est bien téléchargé : `ollama list`\n"
                f"5. Si le modèle n'est pas là, télécharge-le : `ollama pull {self.model}`"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Modèle '{self.model}' non trouvé dans Ollama.\n"
                    f"Télécharge-le avec : `ollama pull {self.model}`\n"
                    f"Vérifie les modèles disponibles : `ollama list`"
                ) from e
            raise

        # Normaliser vers un format OpenAI-like (minimal)
        content = (data.get("message") or {}).get("content", "")
        print("✅ Réponse reçue d'Ollama")
        return {"choices": [{"message": {"content": content}}], "provider_raw": data}







