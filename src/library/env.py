import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def get(name, default=None) -> str:
    logger.trace(f"env.get called with name '{name}' and default '{default}'")
    if os.getenv(name) is None and default is None:
        raise Exception(f"{name} environment variable is not set.")
    elif os.getenv(name) is None:
        return default
    else:
        return os.environ[name]
