"""All settings come from environment variables (or a .env file).

Swapping in the real model is a config change, not a code change:
  RECOGNIZER=keras
  MODEL_PATH=models/sign_model.keras
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # ---- Model 1: sign recognition ----
    # "mock"     = fake model, returns MOCK_SCRIPT words in order (UI testing only)
    # "template" = matches your signing to examples you record in the demo (no training)
    # "keras"    = real trained model (.keras / .h5)
    recognizer: str = "mock"
    model_path: Path = BASE_DIR / "models" / "sign_model.keras"
    labels_path: Path = BASE_DIR / "models" / "labels.json"
    model_config_path: Path = BASE_DIR / "models" / "model_config.json"

    # Template matching: where recorded examples are saved, and how strict matching is
    templates_path: Path = BASE_DIR / "models" / "templates.json"
    template_reject_distance: float = 0.35  # best match further than this -> low confidence
    template_sharpness: float = 12.0          # higher = bigger gap between best and 2nd best

    # Mock model: words it "recognizes" in order, so you can rehearse the demo
    mock_script: str = "I GO SCHOOL | MY MOTHER TEACHER | HELLO NAME WHAT"

    # ---- Safety: confidence filter ----
    # Words below this confidence are dropped instead of guessed
    confidence_threshold: float = 0.6

    # ---- Model 2: sentence builder ----
    # "rule" = simple join (no internet needed)
    # "groq" = Groq LLM (needs GROQ_API_KEY)
    # "hf"   = local Hugging Face model / LoRA adapter (needs transformers + peft)
    sentence_builder: str = "rule"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    hf_base_model: str = ""
    hf_adapter_path: str = ""

    # ---- API ----
    # Comma-separated keys for partner access. Empty = no key needed (demo mode)
    api_keys: str = ""
    cors_origins: str = "*"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()