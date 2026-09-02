"""Run the simple LangGraph chat node.

Usage:
    python backend/scripts/simple_chat.py "hello"
    python backend/scripts/simple_chat.py -m deepseek-reasoner "hello"
"""

from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
