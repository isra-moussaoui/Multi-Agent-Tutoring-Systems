from pathlib import Path
from dotenv import load_dotenv

def load_env():
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
<<<<<<< HEAD
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print(f"Warning: .env not found at {env_path}")
=======
    print(f"Loading .env from: {env_path}")  # temporary
    load_dotenv(dotenv_path=env_path, override=True)
>>>>>>> parent of 4e02c1c (update system to run on api calls sequentially)
