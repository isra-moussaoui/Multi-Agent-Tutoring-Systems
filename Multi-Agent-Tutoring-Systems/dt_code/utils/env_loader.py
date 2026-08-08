from pathlib import Path
from dotenv import load_dotenv

def load_env():
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    print(f"Loading .env from: {env_path}")  # temporary
    load_dotenv(dotenv_path=env_path, override=True)