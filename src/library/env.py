import os
from dotenv import load_dotenv

load_dotenv()


def get_env(name) -> str:
    if os.getenv(name) is None:
        raise Exception(f"{name} environment variable is not set.")
    else:
        return os.environ[name]
