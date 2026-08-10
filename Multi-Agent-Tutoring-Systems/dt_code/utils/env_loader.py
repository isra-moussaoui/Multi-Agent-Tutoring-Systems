from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load .env from the project folder (next to dt_code/) or the git repo root."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / ".env",  # Multi-Agent-Tutoring-Systems/Multi-Agent-Tutoring-Systems/.env
        here.parents[3] / ".env",  # repo root .env
    ]
    for env_path in candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True)
            return
    load_dotenv(override=True)