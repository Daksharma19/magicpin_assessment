"""Produce submission.jsonl for the 30 canonical test pairs.

Usage: GROQ_API_KEY=... python generate_submission.py
"""
import json
import subprocess
import sys
from pathlib import Path

from bot import compose

ROOT = Path(__file__).parent
EXP = ROOT / "dataset" / "expanded"


def load_dir(name, key):
    return {json.loads(p.read_text())[key]: json.loads(p.read_text()) for p in (EXP / name).glob("*.json")}


def main():
    if not (EXP / "test_pairs.json").exists():
        subprocess.run([sys.executable, "generate_dataset.py", "--out", "./expanded"], cwd=ROOT / "dataset", check=True)
    cats = {p.stem: json.loads(p.read_text()) for p in (EXP / "categories").glob("*.json")}
    merchants = load_dir("merchants", "merchant_id")
    customers = load_dir("customers", "customer_id")
    triggers = load_dir("triggers", "id")
    pairs = json.loads((EXP / "test_pairs.json").read_text())["pairs"]
    with open(ROOT / "submission.jsonl", "w") as f:
        for p in pairs:
            trg = triggers[p["trigger_id"]]
            m = merchants[p["merchant_id"]]
            cust = customers.get(p.get("customer_id") or trg.get("customer_id") or "")
            out = compose(cats[m["category_slug"]], m, trg, cust)
            f.write(json.dumps({"test_id": p["test_id"], **out}, ensure_ascii=False) + "\n")
            print(f"{p['test_id']} [{trg['kind']}] {out['body']}\n")


if __name__ == "__main__":
    main()
