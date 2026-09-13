import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write_text(path, content):
    """Atomic replacement only when bytes change; preserve no-op mtimes."""
    path = Path(path)
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    os.replace(temporary, path)
    return True


def write_json(path, value):
    return write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def write_jsonl(path, values):
    return write_text(path, ''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def completed_dates(today, days):
    return [(today - timedelta(days=i)).isoformat() for i in range(days, 0, -1)]


def load_token():
    """Environment wins. Read only known token keys from local .env, never execute it."""
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
    if token:
        return token
    path = Path('.env')
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() in {'HF_TOKEN', 'HUGGINGFACE_API_KEY'}:
                return value.strip().strip('\"\'')
    return None
