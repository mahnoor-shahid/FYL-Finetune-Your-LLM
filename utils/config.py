import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # finetune-pipeline/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _flatten(data, parent_key=""):
    """Flatten nested dicts into dotted keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}."""
    rows = {}
    for key, value in data.items():
        full_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            rows.update(_flatten(value, full_key))
        else:
            rows[full_key] = value
    return rows


def print_json_table(data, title=None):
    """Print any JSON-like dict as a two-column (key | value) table."""
    rows = _flatten(data)
    if not rows:
        print("(empty)")
        return

    rows = {k: json.dumps(v) if isinstance(v, (list, bool)) or v is None else str(v)
            for k, v in rows.items()}
    key_width = max(len("Key"), *(len(k) for k in rows))
    value_width = max(len("Value"), *(len(v) for v in rows.values()))
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"

    if title:
        print(title)
    print(border)
    print(f"| {'Key'.ljust(key_width)} | {'Value'.ljust(value_width)} |")
    print(border)
    for key, value in rows.items():
        print(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    print(border)


def load_config(path=DEFAULT_CONFIG_PATH):
    """Load a JSON file and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_configuration(path=DEFAULT_CONFIG_PATH):
    """Load a JSON file, print it as a table, and return the loaded dict."""
    config = load_config(path)
    print_json_table(config, title=f"Configuration: {Path(path).name}")
    return config


# Loaded once on first import; every module doing `from utils.config import CONFIG`
# shares this same dict (Python caches imported modules).
CONFIG = load_config()


def load_api_key(name, env_path=DEFAULT_ENV_PATH, required=False):
    """
    Load an API key into os.environ[name] and return it.

    Looks in the .env file, then existing environment variables.
    Returns None if not found (or raises if required=True). Never prints the key itself.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass
    value = os.environ.get(name)

    if value:
        value = value.strip()
        os.environ[name] = value
        print(f"✅ Loaded {name} from .env / environment")
        return value

    if required:
        raise RuntimeError(f"{name} not found in {env_path} or environment")
    print(f"⚠️ {name} not found, continuing without it")
    return None


def load_api_keys(*names, env_path=DEFAULT_ENV_PATH, required=False):
    """
    Load several API keys at once. Returns a dict of {name: value} for keys that exist.

    With no names, loads every key defined in the .env file.
    Missing or empty keys are skipped (or raise if required=True).
    """
    from dotenv import dotenv_values, load_dotenv
    load_dotenv(env_path)
    if not names:
        names = list(dotenv_values(env_path))

    present = [name for name in names if (os.environ.get(name) or "").strip()]
    missing = [name for name in names if name not in present]
    if missing and required:
        raise RuntimeError(f"Missing in {env_path} or environment: {', '.join(missing)}")

    keys = {name: load_api_key(name, env_path) for name in present}
    if missing:
        print(f"Skipped (not set): {', '.join(missing)}")
    return keys
