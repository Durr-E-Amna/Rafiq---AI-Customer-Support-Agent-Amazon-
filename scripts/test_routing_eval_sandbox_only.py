"""SANDBOX-ONLY. Verifies the routing eval scoring logic itself works
correctly, using the same FakeLLM/FakeRetriever as the rest of the dev
testing. Not part of the real project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.routing_eval_core import run_routing_eval
from scripts.sandbox_substitutes import FakeLLM, FakeRetriever

if __name__ == "__main__":
    run_routing_eval(FakeLLM(), FakeRetriever())
