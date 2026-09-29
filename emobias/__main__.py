"""Command line.

    python -m emobias plan      configs/<config>.yaml          calls, example prompt
    python -m emobias recognise configs/<config>.yaml [--models a,b] [--conditions c1,c2]
    python -m emobias generate  configs/generation.yaml [--models a,b] [--conditions c1,c2]
    python -m emobias analyse                                  results/measures/ and docs/data/
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="emobias")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("plan", "recognise", "generate"):
        p = sub.add_parser(cmd)
        p.add_argument("config")
        p.add_argument("--models", default=None)
        p.add_argument("--conditions", default=None)
    sub.add_parser("analyse")
    a = ap.parse_args()
    models = a.models.split(",") if getattr(a, "models", None) else None
    conditions = a.conditions.split(",") if getattr(a, "conditions", None) else None
    if a.cmd == "analyse":
        from . import analysis
        analysis.run()
        return
    import yaml
    kind = yaml.safe_load(open(a.config, encoding="utf-8")).get("task", "recognition")
    if kind == "generation":
        from . import generate as mod
    else:
        from . import recognise as mod
    cfg = mod.load_config(a.config)
    if a.cmd == "plan":
        print(json.dumps(mod.plan(cfg), indent=1, ensure_ascii=False))
    else:
        mod.run(cfg, models, conditions)


if __name__ == "__main__":
    main()
