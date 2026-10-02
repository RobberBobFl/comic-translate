import base64
import json
import logging
import re
import time

import numpy as np
import requests

from .reasoning import get_reasoning_off_params

logger = logging.getLogger(__name__)


class TranslationReviewer:
    """Review finished translations with a dedicated OpenAI-compatible model.

    Mirrors :class:`SceneAnalyzer`: an isolated vision/text client with its own
    credentials that receives the page image (optional) plus the original and
    translated text of every block, and returns per-block suggestions. The
    model only comments on blocks worth changing; a good translation produces
    no entry.
    """

    DEFAULT_API_URL = "http://localhost:11434/v1"
    # Generous base budget: reasoning models spend completion tokens on
    # thinking before emitting text, and a failed attempt is still billed -
    # paying for one bigger request beats paying for a retry.
    MAX_COMPLETION_TOKENS = 5000
    MAX_COMPLETION_TOKENS_CEILING = 16000
    REVIEW_ATTEMPTS = 2
    MAX_PAGE_SIDE = 650
    WEBTOON_MAX_WIDTH = 300
    CONNECT_TIMEOUT_SECONDS = 10
    READ_TIMEOUT_SECONDS = 300

    SEVERITIES = ("error", "improvement")
    CATEGORIES = ("calque", "mistranslation", "terminology", "style", "scene_context", "other")
    MAX_ALTERNATIVES = 2

    SYSTEM_PROMPT = """\
You are a comic translation reviewer. You are given a list of BLOCK N entries,
each with the ORIGINAL text and its current TRANSLATION, plus (optionally) the
comic page image for context.

For every block whose translation should be corrected or improved, return one
entry. Do NOT include blocks whose translation is already good.

Fields:
- block: the block number (integer, as given in the list)
- severity: "error" or "improvement"
  - "error": the translation is wrong (mistranslation, calque from the source
    language, wrong understanding of the scene, wrong terminology).
  - "improvement": the translation is correct, but a noticeably more natural
    or better-fitting variant exists.
- category: one of calque, mistranslation, terminology, style, scene_context, other
- reason: one short sentence explaining the problem, written in the TARGET
  language of the translation
- recommended: your best replacement text, in the target language
- alternatives: up to 2 other acceptable variants (may be empty)

Rules:
1. Only comment on blocks that actually need changes. If everything is good,
   return an empty list.
2. Keep the speaker's voice: shouting stays loud, whispering stays soft,
   jokes stay jokes.
3. Never invent content that is not in the original.
4. Output ONLY this JSON, no markdown fences, no explanations:

{"blocks": [{"block": 0, "severity": "error", "category": "calque", "reason": "...", "recommended": "...", "alternatives": ["...", "..."]}]}"""

    DEFAULT_REVIEW_PROMPT = """\
Review this comic page translation from {source_lang} to {target_lang}.
For each block, compare the ORIGINAL text with its current TRANSLATION and
judge:
- meaning: does the translation say what the original says?
- naturalness: does it sound like a native {target_lang} speaker wrote it?
- tone: does it fit the scene (see the page image if provided)?
Watch especially for calques - literal borrowings from {source_lang} that
sound unnatural in {target_lang}."""

    def __init__(self, api_key: str = "", api_url: str = DEFAULT_API_URL, model: str = ""):
        self.api_key = api_key or ""
        self.model = model or ""
        base = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self.api_url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        # Current completion budget; reasoning models can exhaust it with
        # thinking before emitting any text, so retries grow it.
        self._completion_budget = self.MAX_COMPLETION_TOKENS

    @classmethod
    def from_settings(cls, settings) -> "TranslationReviewer":
        credentials = settings.get_reviewer_credentials()
        return cls(
            api_key=credentials.get("api_key", ""),
            api_url=credentials.get("api_url", cls.DEFAULT_API_URL),
            model=credentials.get("model", ""),
        )

    @staticmethod
    def apply_placeholders(template: str, source_lang: str, target_lang: str) -> str:
        """Fill {source_lang}/{target_lang} without str.format so user braces survive."""
        text = template or ""
        text = text.replace("{source_lang}", source_lang or "the source language")
        text = text.replace("{target_lang}", target_lang or "the target language")
        return text

    @classmethod
    def _prepare_image(cls, image: np.ndarray, is_webtoon: bool = False) -> np.ndarray:
        import imkit as imk

        height, width = image.shape[:2]
        if is_webtoon:
            if width <= cls.WEBTOON_MAX_WIDTH:
                return image
            scale = cls.WEBTOON_MAX_WIDTH / width
        else:
            longest_side = max(height, width)
            if longest_side <= cls.MAX_PAGE_SIDE:
                return image
            scale = cls.MAX_PAGE_SIDE / longest_side
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return imk.resize(image, (resized_width, resized_height))

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        import imkit as imk

        payload = imk.encode_image(image, "jpg")
        return base64.b64encode(payload).decode("utf-8")

    @staticmethod
    def format_review_blocks(blk_list: list) -> str:
        """Render blocks as BLOCK N entries with original + current translation.

        Blocks with no source text are skipped; blocks with source text but no
        translation are listed with "(empty)" so the reviewer can flag them.
        """
        sections = []
        for idx, blk in enumerate(blk_list or []):
            source = re.sub(r"\s+", " ", str(getattr(blk, "text", "") or "")).strip()
            if not source:
                continue
            translation = str(getattr(blk, "translation", "") or "").strip()
            if not translation:
                translation = "(empty)"
            else:
                translation = re.sub(r"\s+", " ", translation)
            sections.append(
                f"BLOCK {idx}:\nORIGINAL: {source}\nTRANSLATION: {translation}"
            )
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Transport (image-optional variants of the SceneAnalyzer queries)
    # ------------------------------------------------------------------

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
        return "ollama" in hostname

    def _ollama_base(self) -> str:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self.api_url)
        scheme = parsed.scheme or "http"
        return urlunparse((scheme, parsed.netloc, "", "", "", ""))

    def _query(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        if self._is_ollama():
            return self._query_ollama(system_prompt, user_prompt, encoded_image)
        return self._query_openai(system_prompt, user_prompt, encoded_image)

    def _query_openai(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        content: list | str
        if encoded_image:
            content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," f"{encoded_image}"},
                },
            ]
        else:
            content = user_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self._completion_budget,
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
        logger.debug("Review request took %.1fs", time.perf_counter() - request_started)
        if not response.ok:
            # Log the provider's error body - it usually explains the failure
            # (wrong model name, context limit, billing, ...).
            logger.warning(
                "Review endpoint returned HTTP %s: %s",
                response.status_code,
                response.text[:400],
            )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return {"content": str(content)}

    def _query_ollama(self, system_prompt: str, user_prompt: str, encoded_image: str) -> dict:
        message: dict = {"role": "user", "content": user_prompt}
        if encoded_image:
            message["images"] = [encoded_image]
        url = self._ollama_base().rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                message,
            ],
            # The OpenAI-compatible /v1 path ignores `think`, so reasoning
            # models exhaust the budget before emitting any answer.
            "think": False,
            "num_predict": self._completion_budget,
            "stream": False,
        }
        request_started = time.perf_counter()
        response = requests.post(
            url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=(self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS),
        )
        logger.debug("Review request took %.1fs", time.perf_counter() - request_started)
        if not response.ok:
            logger.warning(
                "Review endpoint returned HTTP %s: %s",
                response.status_code,
                response.text[:400],
            )
        response.raise_for_status()
        data = response.json()
        reply = data.get("message", {})
        content = reply.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return {"content": str(content)}

    # ------------------------------------------------------------------
    # Review entry point
    # ------------------------------------------------------------------

    def review(
        self,
        blk_list: list,
        image: np.ndarray | None = None,
        source_lang: str = "",
        target_lang: str = "",
        instructions: str = "",
        scene_description: str = "",
        is_webtoon: bool = False,
    ) -> str | None:
        """Send the block texts (and optionally the page) to the reviewer model.

        Returns the raw model response, or ``None`` when there is nothing to
        review or the request failed. Never raises for request errors - a
        failed review must not break the workflow.
        """
        blocks_text = self.format_review_blocks(blk_list)
        if not blocks_text.strip():
            logger.info("Review skipped: no source text to review.")
            return None
        if not self.model:
            logger.warning("Review skipped: no review model configured.")
            return None

        task = instructions.strip() or self.apply_placeholders(
            self.DEFAULT_REVIEW_PROMPT, source_lang, target_lang
        )
        parts = [task]
        if scene_description and scene_description.strip():
            parts.append(
                "PAGE CONTEXT (a brief scene description, may be imperfect):\n"
                f"{scene_description.strip()}"
            )
        parts.append(
            "TEXT BLOCKS (original + current translation, in reading order):\n\n"
            f"{blocks_text}"
        )
        user_prompt = "\n\n".join(parts)

        try:
            encoded_image = ""
            if image is not None:
                prepared = self._prepare_image(image, is_webtoon)
                encoded_image = self._encode_image(prepared)
        except Exception:
            logger.exception("Failed to prepare the page image for review.")
            return None

        for attempt in range(1, self.REVIEW_ATTEMPTS + 1):
            try:
                result = self._query(self.SYSTEM_PROMPT, user_prompt, encoded_image)
            except Exception:
                logger.exception("Translation review request failed.")
                return None

            content = (result.get("content") or "").strip()
            if content:
                return content

            # The request "succeeded" from the provider's point of view (it is
            # billed too), but produced no text. The usual culprit: a
            # reasoning model spent the whole completion budget thinking
            # (finish_reason="length"). Retry with a bigger budget.
            finish_reason = result.get("finish_reason")
            logger.warning(
                "Review attempt %d/%d returned empty content (finish_reason=%s, completion_tokens=%s).",
                attempt, self.REVIEW_ATTEMPTS, finish_reason,
                result.get("completion_tokens"),
            )
            if attempt < self.REVIEW_ATTEMPTS:
                if finish_reason == "length":
                    self._completion_budget = min(
                        self._completion_budget * 2, self.MAX_COMPLETION_TOKENS_CEILING
                    )
                    logger.info("Retrying the review with completion budget %d.", self._completion_budget)
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse_review_response(cls, text: str) -> dict[int, dict]:
        """Parse and normalize the reviewer's JSON into ``{block_index: entry}``.

        Accepts ``{"blocks": [...]}``, a bare list, or a ``{"block_N": {...}}``
        mapping. Entries without a usable block number or an empty
        ``recommended`` are dropped; unknown severities become "improvement"
        (the weaker claim) and unknown categories become "other".
        """
        parsed = cls._extract_json(text)
        if parsed is None:
            logger.warning("Review response is not valid JSON.")
            return {}

        entries: list[tuple[int | None, dict]] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("blocks"), list):
            entries = [(None, item) for item in parsed["blocks"] if isinstance(item, dict)]
        elif isinstance(parsed, list):
            entries = [(None, item) for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            for key, item in parsed.items():
                if isinstance(item, dict):
                    match = re.match(r"block_(\d+)$", str(key), re.IGNORECASE)
                    entries.append(
                        (int(match.group(1)) if match else None, item)
                    )

        result: dict[int, dict] = {}
        for fallback_idx, item in entries:
            block_idx = cls._coerce_block_index(item.get("block"))
            if block_idx is None:
                block_idx = fallback_idx
            if block_idx is None or block_idx < 0:
                logger.debug("Dropping review entry without a valid block number: %r", item)
                continue
            recommended = str(item.get("recommended", "") or "").strip()
            if not recommended:
                logger.debug("Dropping review entry without a replacement (block %d).", block_idx)
                continue
            result[block_idx] = {
                "severity": cls._normalize_severity(item.get("severity")),
                "category": cls._normalize_category(item.get("category")),
                "reason": str(item.get("reason", "") or "").strip(),
                "recommended": recommended,
                "alternatives": cls._normalize_alternatives(item.get("alternatives")),
            }
        return result

    @staticmethod
    def _extract_json(text: str):
        from modules.utils.translator_utils import (
            _extract_json_value,
            _parse_lenient,
            _strip_code_fences,
        )

        if not text:
            return None
        cleaned = _strip_code_fences(text)
        value = _extract_json_value(cleaned)
        if value is None:
            value = _parse_lenient(cleaned)
        if value is None:
            # Models sometimes chat before the JSON ("Sure! Here is my
            # review: {...}") - try again from the first brace/bracket.
            offsets = [cleaned.find(ch) for ch in "{["]
            offsets = [off for off in offsets if off > 0]
            if offsets:
                value = _extract_json_value(cleaned[min(offsets):])
        return value

    @staticmethod
    def _coerce_block_index(value) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.match(r"block_(\d+)$", value.strip(), re.IGNORECASE)
            if match:
                return int(match.group(1))
            try:
                return int(value.strip())
            except ValueError:
                return None
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    @classmethod
    def _normalize_severity(cls, value) -> str:
        severity = str(value or "").strip().lower()
        if severity in cls.SEVERITIES:
            return severity
        return "improvement"

    @classmethod
    def _normalize_category(cls, value) -> str:
        category = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        if category in cls.CATEGORIES:
            return category
        return "other"

    @classmethod
    def _normalize_alternatives(cls, value) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        cleaned = [str(item or "").strip() for item in value]
        return [item for item in cleaned if item][: cls.MAX_ALTERNATIVES]
