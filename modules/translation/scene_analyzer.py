import base64
import json
import logging
import re
import time

import numpy as np
import requests

from .reasoning import get_reasoning_off_params

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    """Generate concise page-level context through a vision chat endpoint."""

    DEFAULT_API_URL = "http://localhost:11434/v1"
    MAX_COMPLETION_TOKENS = 1500
    MAX_CONTEXT_WORDS = 150
    MAX_IMAGE_SIDE = 896
    CONNECT_TIMEOUT_SECONDS = 10
    READ_TIMEOUT_SECONDS = 300
    SYSTEM_PROMPT = """You help translate a comic page. The OCR already gives the words; you add visual context.

List the visual action sequence in approximate reading order (top-left to bottom-right).
Describe only: who does what, gestures, reactions, and setting details.
Refer to people by appearance (e.g. "man in red coat"); never invent names.
If a character is clearly speaking, identify them by visual cue (mouth/gesture), but never quote or translate the spoken words.
Do NOT copy, quote, or translate the OCR text or retell the story.

Format: numbered steps (1. 2. 3.), one action per line.
English only. Be terse: under 12 words per line.

Example:
1. Angry man in red coat points at a door
2. Woman flinches away
3. Rain streaks the window behind her"""

    def __init__(self, api_key: str = "", api_url: str = DEFAULT_API_URL, model: str = ""):
        self.api_key = api_key or ""
        self.model = model or ""
        base = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self.api_url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    @classmethod
    def from_settings(cls, settings) -> "SceneAnalyzer":
        credentials = settings.get_scene_analyzer_credentials()
        return cls(
            api_key=credentials.get("api_key", ""),
            api_url=credentials.get("api_url", cls.DEFAULT_API_URL),
            model=credentials.get("model", ""),
        )

    @classmethod
    def _prepare_image(cls, image: np.ndarray) -> np.ndarray:
        """Downscale large pages to reduce vision prompt processing time."""
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= cls.MAX_IMAGE_SIDE:
            return image

        import imkit as imk

        scale = cls.MAX_IMAGE_SIDE / longest_side
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return imk.resize(image, (resized_width, resized_height))

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        import imkit as imk

        payload = imk.encode_image(image, "jpg")
        return base64.b64encode(payload).decode("utf-8")

    @staticmethod
    def format_source_blocks(blocks: list) -> str:
        """Format recognized text blocks in their approximate reading order."""
        sections = []
        for block in blocks or []:
            text = re.sub(r"\s+", " ", str(getattr(block, "text", "") or "")).strip()
            if text:
                sections.append(f"BLOCK {len(sections)}:\n{text}")
        return "\n\n".join(sections)

    @classmethod
    def _trim_description(cls, text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
        text = "\n".join(line for line in lines if line)
        if not text:
            return ""

        words = text.split()
        if len(words) > cls.MAX_CONTEXT_WORDS:
            text = " ".join(words[:cls.MAX_CONTEXT_WORDS]).rstrip(".,;:") + "."
        return text

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _is_ollama(self) -> bool:
        from urllib.parse import urlparse

        parsed = urlparse(self.api_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.port == 11434:
            return True
        if "ollama" in hostname:
            return True
        return False

    def _ollama_base(self) -> str:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self.api_url)
        scheme = parsed.scheme or "http"
        return urlunparse((scheme, parsed.netloc, "", "", "", ""))

    def _query(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        """Call the vision endpoint and return a normalized result dict."""
        if self._is_ollama():
            return self._query_ollama(system_prompt, user_prompt, encoded_image)
        return self._query_openai(system_prompt, user_prompt, encoded_image)

    def _query_openai(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64," f"{encoded_image}"
                                )
                            },
                        },
                    ],
                },
            ],
            "max_completion_tokens": self.MAX_COMPLETION_TOKENS,
        }
        payload.update(
            get_reasoning_off_params(self.api_url, self.model, self._is_ollama())
        )
        request_started = time.perf_counter()
        response = requests.post(
            self.api_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
        )
        request_elapsed = time.perf_counter() - request_started
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return {
            "content": str(content),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "reasoning_chars": len(str(message.get("reasoning_content") or "")),
        }

    def _query_ollama(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        url = self._ollama_base().rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [encoded_image]},
            ],
            # Native Ollama route: the OpenAI-compatible /v1 path ignores `think`,
            # so reasoning models exhaust the budget before emitting any answer.
            "think": False,
            "num_predict": self.MAX_COMPLETION_TOKENS,
            "stream": False,
        }
        request_started = time.perf_counter()
        response = requests.post(
            url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
        )
        request_elapsed = time.perf_counter() - request_started
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return {
            "content": str(content),
            "finish_reason": data.get("done_reason"),
            "completion_tokens": data.get("eval_count"),
            "reasoning_chars": len(str(message.get("thinking") or "")),
        }

    def analyze(
        self,
        image: np.ndarray,
        source_text: str = "",
        previous_description: str | None = None,
    ) -> str | None:
        if image is None:
            logger.warning("Scene analysis skipped: no page image available.")
            return None
        if not self.model:
            logger.warning("Scene analysis skipped: no vision model configured.")
            return None

        try:
            image_shape = getattr(image, "shape", None)
            if not source_text.strip():
                return None

            if previous_description and previous_description.strip():
                task = (
                    "Analyze the page again and replace the previous visual-context "
                    "draft. Independently verify every detail against the image and "
                    "the OCR text. Keep accurate details and correct mistakes.\n\n"
                    f"PREVIOUS DRAFT:\n{previous_description.strip()}"
                )
            else:
                task = "Analyze the comic page using the image and OCR text."
            user_prompt = (
                f"{task}\n\nSOURCE TEXT FROM OCR "
                "(blocks are in approximate reading order):\n\n"
                f"{source_text.strip()}"
            )

            prepared_image = self._prepare_image(image)
            prepared_shape = getattr(prepared_image, "shape", None)
            encoded_image = self._encode_image(prepared_image)

            request_started = time.perf_counter()
            result = self._query(self.SYSTEM_PROMPT, user_prompt, encoded_image)
            content = result["content"]
            description = self._trim_description(content)
            return description or None
        except requests.exceptions.ReadTimeout as exc:
            elapsed = (
                time.perf_counter() - request_started
                if "request_started" in locals()
                else 0.0
            )
            logger.exception("Scene analysis request timed out.")
            return None
        except Exception as exc:
            logger.exception("Scene analysis request failed.")
            return None
