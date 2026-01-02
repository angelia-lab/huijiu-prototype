# src/huijiu/apps/integrated_safe_loop_demo.py
from __future__ import annotations

from huijiu.core.orchestrator import IntegratedOrchestrator


def main() -> int:
    orchestrator = IntegratedOrchestrator()
    orchestrator.run_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
