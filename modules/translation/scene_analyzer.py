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
    MAX_CONTEXT_WORDS = 500
    MAX_PAGE_SIDE = 650
    WEBTOON_MAX_WIDTH = 300
    CONNECT_TIMEOUT_SECONDS = 10
    READ_TIMEOUT_SECONDS = 300
    SYSTEM_PROMPT = """\
You are a comic-page analyst. For EACH text block listed below the image,
return exactly one entry in the EXACT format shown.

Fields:
- Speaker: one of male, female, child, robot, creature, narrator, unknown
- Emotion: one word for the speaker's emotional state
- Delivery: one of normal, shouting, whispering, thinking, narration
- Text: the text you see in the speech bubble / caption (exact words)
- Context: very short visual context (max 10 words)

Rules:
1. Process every BLOCK in the order given. Do NOT skip or merge blocks.
2. Refer to people only by appearance (e.g. "man in red coat").
   Never invent names, identities, relationships, or roles.
3. If uncertain about speaker or emotion, use "unknown".
4. Text must match what you see in the image exactly — do not translate
   or paraphrase it.
5. Keep Context short and factual. No speculation.

Output format (one entry per block, separated by a blank line):

BLOCK N: Speaker: <speaker> | Emotion: <emotion> | Delivery: <delivery>
  Text: "<exact text>"
  Context: <brief visual context>

Example (two blocks):

BLOCK 0: Speaker: male | Emotion: angry | Delivery: shouting
  Text: "I won't let you get away with this!"
  Context: man in red coat pointing at door

BLOCK 1: Speaker: female | Emotion: scared | Delivery: whispering
  Text: "Please, don't hurt me..."
  Context: woman backing away against wall"""

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
    def _prepare_image(cls, image: np.ndarray, is_webtoon: bool = False) -> np.ndarray:
        """Downscale before sending to the VLM for scene description.

        Ordinary pages are capped on their longest side. Webtoon strips are
        capped only on width so their (very large) height is preserved.
        """
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
    def format_source_blocks(blocks: list, include_coords: bool = False) -> str:
        """Format recognized text blocks in their approximate reading order.

        When *include_coords* is True each block header carries its bounding-box
        centre so the VLM can use spatial information when identifying speakers
        and speech-bubble associations.
        """
        sections = []
        for block in blocks or []:
            text = re.sub(r"\s+", " ", str(getattr(block, "text", "") or "")).strip()
            if not text:
                continue
            idx = len(sections)
            if include_coords:
                xyxy = getattr(block, "xyxy", None)
                if xyxy is not None:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        sections.append(
                            f"BLOCK {idx} (x={cx}, y={cy}):\n{text}"
                        )
                    except (TypeError, ValueError):
                        sections.append(f"BLOCK {idx}:\n{text}")
                else:
                    sections.append(f"BLOCK {idx}:\n{text}")
            else:
                sections.append(f"BLOCK {idx}:\n{text}")
        return "\n\n".join(sections)

    @classmethod
    def _trim_description(cls, text: str) -> str:
        """Clean up whitespace and enforce the word limit.

        For structured block output the method tries to keep complete block
        entries (BLOCK N: … Context: …) instead of cutting in the middle of
        one.
        """
        lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
        text = "\n".join(line for line in lines if line)
        if not text:
            return ""

        words = text.split()
        if len(words) <= cls.MAX_CONTEXT_WORDS:
            return text

        # Structured block output — trim by keeping complete blocks.
        block_starts = [
            m.start()
            for m in re.finditer(r"^BLOCK\s*\d+\s*:", text, re.MULTILINE)
        ]
        if block_starts:
            # Keep whole blocks that fit within the word budget.
            last_good = 0
            for pos in block_starts:
                prefix = text[:pos]
                if len(prefix.split()) <= cls.MAX_CONTEXT_WORDS:
                    last_good = pos
                else:
                    break
            if last_good:
                return text[:last_good].rstrip()

        # Free-form fallback — truncate at word boundary.
        text = " ".join(words[: cls.MAX_CONTEXT_WORDS]).rstrip(".,;:") + "."
        return text

    # ------------------------------------------------------------------
    # Block-level metadata parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vlm_blocks(text: str) -> list[dict]:
        """Parse VLM response into a list of per-block metadata dicts.

        Each dict has keys: speaker, emotion, delivery, text, context.
        Blocks that cannot be parsed are skipped.
        """
        entries: list[dict] = []
        if not text:
            return entries

        # Split on "BLOCK N:" pattern (N may be omitted by the model).
        block_pattern = re.compile(
            r"BLOCK\s*\d*\s*:\s*", re.IGNORECASE
        )
        raw_parts = block_pattern.split(text)
        # The first split piece is preamble (before the first BLOCK) — skip it.
        for part in raw_parts[1:]:
            part = part.strip()
            if not part:
                continue

            entry: dict = {}

            # --- Header line: Speaker: ... | Emotion: ... | Delivery: ... ---
            header_match = re.search(
                r"Speaker:\s*(\w+)"
                r".*?Emotion:\s*(\w+)"
                r".*?Delivery:\s*(\w+)",
                part,
                re.IGNORECASE,
            )
            if header_match:
                entry["speaker"] = header_match.group(1).lower()
                entry["emotion"] = header_match.group(2).lower()
                entry["delivery"] = header_match.group(3).lower()
            else:
                entry["speaker"] = "unknown"
                entry["emotion"] = ""
                entry["delivery"] = "normal"

            # --- Text field ---
            text_match = re.search(
                r'Text:\s*["\u201c](.*?)["\u201d]', part, re.DOTALL
            )
            if text_match:
                entry["text"] = text_match.group(1).strip()
            else:
                # Fallback: Text: ... (without quotes)
                text_match2 = re.search(
                    r"Text:\s*(.+?)(?:\n|Context:)", part, re.IGNORECASE
                )
                entry["text"] = text_match2.group(1).strip() if text_match2 else ""

            # --- Context field ---
            ctx_match = re.search(
                r"Context:\s*(.+?)$", part, re.IGNORECASE | re.MULTILINE
            )
            entry["context"] = ctx_match.group(1).strip() if ctx_match else ""

            entries.append(entry)

        return entries

    @staticmethod
    def _fuzzy_ratio(a: str, b: str) -> float:
        """Simple token-overlap ratio between two strings (0-100).

        Good enough for short comic lines; avoids external dependencies.
        """
        if not a or not b:
            return 0.0
        tok_a = set(a.lower().split())
        tok_b = set(b.lower().split())
        if not tok_a or not tok_b:
            return 0.0
        intersection = tok_a & tok_b
        return 100.0 * len(intersection) / max(len(tok_a), len(tok_b))

    @classmethod
    def match_scene_metadata(
        cls,
        vlm_output: str,
        ocr_blocks: list,
    ) -> tuple[dict[int, dict], list[str]]:
        """Parse VLM output and match entries to OCR blocks.

        Returns ``(metadata, log_lines)`` where *metadata* is
        ``{block_index: {speaker, emotion, delivery, text, context}}``
        and *log_lines* is a list of human-readable match descriptions
        suitable for debug logging.

        Matching strategy:
        1. If VLM returned exactly len(ocr_blocks) entries → index-based.
        2. Otherwise → fuzzy text matching against OCR text.
        """
        entries = cls._parse_vlm_blocks(vlm_output)
        log: list[str] = []
        if not entries or not ocr_blocks:
            return {}, log

        ocr_texts = [
            re.sub(r"\s+", " ", str(getattr(b, "text", "") or "")).strip()
            for b in ocr_blocks
        ]

        # Fast path: same count → index mapping
        if len(entries) == len(ocr_texts):
            result = {i: e for i, e in enumerate(entries)}
            for i, e in enumerate(entries):
                log.append(
                    f"BLOCK {i} (index match): speaker={e.get('speaker', '?')}, "
                    f"emotion={e.get('emotion', '?')}, delivery={e.get('delivery', '?')}, "
                    f"context=\"{e.get('context', '')}\""
                )
            return result, log

        # Fallback: fuzzy match each VLM entry to the best OCR block
        result: dict[int, dict] = {}
        used: set[int] = set()
        log.append(f"Fuzzy fallback: VLM returned {len(entries)} blocks, OCR has {len(ocr_texts)}")
        for entry in entries:
            vlm_text = entry.get("text", "")
            best_idx = -1
            best_score = 0.0
            for idx, ocr_text in enumerate(ocr_texts):
                if idx in used:
                    continue
                score = cls._fuzzy_ratio(vlm_text, ocr_text)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score >= 50.0:
                result[best_idx] = entry
                used.add(best_idx)
                log.append(
                    f"BLOCK {best_idx} (fuzzy match, score={best_score:.0f}%): "
                    f"speaker={entry.get('speaker', '?')}, "
                    f"emotion={entry.get('emotion', '?')}, delivery={entry.get('delivery', '?')}, "
                    f"context=\"{entry.get('context', '')}\""
                )
            else:
                log.append(
                    f"VLM entry unmatched (best_score={best_score:.0f}%): "
                    f"text={vlm_text!r}"
                )

        return result, log

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
        is_webtoon: bool = False,
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
                    "The previous draft below was inaccurate. Analyze the page "
                    "independently and produce fresh block-by-block metadata.\n\n"
                    f"PREVIOUS DRAFT:\n{previous_description.strip()}"
                )
            else:
                task = "Analyze the comic page image."
            user_prompt = (
                f"{task}\n\n"
                "TEXT BLOCKS (from OCR, in approximate reading order). "
                "For each block, return Speaker, Emotion, Delivery, Text, "
                "and Context in the exact format specified.\n\n"
                f"{source_text.strip()}"
            )

            prepared_image = self._prepare_image(image, is_webtoon)
            prepared_shape = getattr(prepared_image, "shape", None)
            encoded_image = self._encode_image(prepared_image)

            request_started = time.perf_counter()
            result = self._query(self.SYSTEM_PROMPT, user_prompt, encoded_image)
            content = result["content"]
            logger.debug("SceneAnalyzer VLM raw output:\n%s", content)
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
