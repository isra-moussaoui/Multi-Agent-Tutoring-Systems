from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load Multi-Agent-Tutoring-Systems/.env into os.environ."""
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print(f"Warning: .env not found at {env_path}")
