import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

ORCHESTRATOR_MODEL = "gpt-4o"
AGENT_MODEL = "gpt-4o"
MAX_TOKENS = 8192
