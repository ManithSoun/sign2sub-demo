"""Model 2: sentence builder (list of glosses -> natural English sentence).

Pick one with SENTENCE_BUILDER = rule | groq | hf.
If the LLM fails (no internet, rate limit...), we fall back to the rule-based
builder so the demo never breaks on stage.
"""
import logging
from abc import ABC, abstractmethod

import httpx

log = logging.getLogger("sign2sub.sentence")

PROMPT = (
    "You turn sign language glosses into one natural English sentence.\n"
    "Rules:\n"
    "- Use ONLY the meaning of the given glosses. Do not add new facts, names, places or feelings.\n"
    "- You may add small grammar words (a, the, is, am, to, my...) and fix verb tense.\n"
    "- Keep the same order of ideas.\n"
    "- Reply with the sentence only, no quotes, no explanation.\n\n"
    "Glosses: {glosses}\nSentence:"
)


class SentenceBuilder(ABC):
    name = "base"

    @abstractmethod
    def build(self, words: list[str]) -> str: ...


class RuleBuilder(SentenceBuilder):
    """No AI: joins the words and tidies them up. Always works offline."""
    name = "rule"

    def build(self, words):
        if not words:
            return ""
        text = " ".join(w.replace("-", " ").replace("_", " ").lower() for w in words)
        text = " ".join("I" if t == "i" else t for t in text.split())
        return text[0].upper() + text[1:] + "."


class GroqBuilder(SentenceBuilder):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("SENTENCE_BUILDER=groq needs GROQ_API_KEY")
        self.api_key, self.model = api_key, model
        self.client = httpx.Client(timeout=8.0)

    def build(self, words):
        if not words:
            return ""
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 60,
            "messages": [{"role": "user", "content": PROMPT.format(glosses=" ".join(words))}],
        }
        if "gpt-oss" in self.model or "qwen" in self.model:
            # reasoning models think before answering: keep it short and leave room for the answer
            body.update(max_tokens=400, reasoning_effort="low")
        r = self.client.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")
        text = (r.json()["choices"][0]["message"].get("content") or "").strip().strip('"')
        return text.splitlines()[0].strip() if text else ""


class HFBuilder(SentenceBuilder):
    """Local seq2seq / causal model, optionally with a LoRA (PEFT) adapter.

    HF_BASE_MODEL   = base model name or folder (e.g. google/flan-t5-base)
    HF_ADAPTER_PATH = folder with adapter_config.json + adapter weights (optional)
    Needs: pip install transformers peft torch
    """
    name = "hf"

    def __init__(self, base_model: str, adapter_path: str):
        if not base_model:
            raise ValueError("SENTENCE_BUILDER=hf needs HF_BASE_MODEL")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.is_seq2seq = AutoConfig.from_pretrained(base_model).is_encoder_decoder
        cls = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
        model = cls.from_pretrained(base_model)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()

    def build(self, words):
        if not words:
            return ""
        # TODO: match the exact input format the adapter was fine-tuned on
        prompt = " ".join(words) if self.is_seq2seq else PROMPT.format(glosses=" ".join(words))
        inputs = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        if not self.is_seq2seq:
            out = out[:, inputs["input_ids"].shape[1]:]
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


class FallbackBuilder(SentenceBuilder):
    """Tries the main builder, uses the rule builder if it errors."""

    def __init__(self, main: SentenceBuilder):
        self.main, self.backup = main, RuleBuilder()
        self.name = main.name

    def build(self, words):
        try:
            return self.main.build(words) or self.backup.build(words)
        except Exception as e:  # never break the demo because of the LLM
            log.warning("Sentence builder %s failed (%s); using rule-based", self.main.name, e)
            return self.backup.build(words)


def build_sentence_builder(settings) -> SentenceBuilder:
    kind = settings.sentence_builder
    if kind == "rule":
        return RuleBuilder()
    if kind == "groq":
        return FallbackBuilder(GroqBuilder(settings.groq_api_key, settings.groq_model))
    if kind == "hf":
        return FallbackBuilder(HFBuilder(settings.hf_base_model, settings.hf_adapter_path))
    raise ValueError(f"Unknown SENTENCE_BUILDER={kind!r} (use 'rule', 'groq' or 'hf')")