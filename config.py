import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ORCHESTRATOR_MODEL = "claude-opus-4-7"
AGENT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
