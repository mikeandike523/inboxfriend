#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive category-mapping wizard (label-the-invalid, prompt_toolkit edition)

Flow:
  1) Fetch DISTINCT categories from DB.
  2) One LLM call to propose ALLOWED canonicals (ensure "unknown" present).
  3) Auto-map originals already valid (case-insensitive).
  4) For remaining INVALID originals, prompt for target (name or number).
     - Autocomplete with prompt_toolkit.
     - Commands (out-of-band on the allowed set):
         !add-cat <name>
         !delete-cat <name|number>        # cannot delete 'unknown'
         !change-cat <name|number> <new>  # renames category, updates existing mappings
     - After any command, re-prompt the same original.
  5) Summary -> write stable TSV -> ask "Apply changes?" (default Yes).
  6) Apply DB updates (skip no-ops).

Install:
  pip install openai SQLAlchemy pymysql python-dotenv prompt_toolkit termcolor
"""

import os
import sys
import datetime
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# ---------- Fixed paths ----------
ENV_PATH = ".env"                       # must contain DB_URL_EXTERNAL
OPENAI_KEY_PATH = "OPENAI_API_KEY.txt"  # file with ONLY the API key

# ---------- Defaults ----------
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
TABLE_DEFAULT = "message_classification"
COLUMN_DEFAULT = "category"

# ---------- Prompt text for LLM ----------
ALLOWED_SYSTEM = """\
You are a taxonomy designer.

Task:
Given a DISTINCT list of category strings, propose a small, sensible set of canonical categories
that those items should map to.

Rules:
- Output ONLY a newline-separated list of canonical category names.
- Keep names lowercase; preserve meaningful hyphens/underscores.
- Prefer plural form for container-like categories (e.g., "notifications", "newsletters", "receipts", "reports").
- Keep semantically distinct concepts separate (e.g., "order-confirmation" vs "confirmation").
- Include "unknown" to catch anomalies that do not fit any sensible bucket.
- Do NOT include anything that clearly doesn't belong or duplicates another canonical.
- Do NOT emit any prose, bullets, or JSON—only one category per line.
"""

# ---------- Config / DB ----------
def load_config():
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        print(f"Warning: .env not found at {ENV_PATH}; relying on environment.")
    db_url = os.getenv("DB_URL_EXTERNAL")
    if not db_url:
        raise RuntimeError("DB_URL_EXTERNAL not set in environment/.env")

    if not os.path.exists(OPENAI_KEY_PATH):
        raise FileNotFoundError(f"OPENAI key file not found at {OPENAI_KEY_PATH}")
    with open(OPENAI_KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY file is empty")

    return {
        "DB_URL": db_url,
        "TABLE": os.getenv("DB_TABLE", TABLE_DEFAULT),
        "COLUMN": os.getenv("CATEGORY_COLUMN", COLUMN_DEFAULT),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
        "OPENAI_API_KEY": api_key,
    }

def connect_db(cfg):
    return create_engine(cfg["DB_URL"])

def fetch_distinct(engine, table: str, column: str) -> List[str]:
    sql = text(f"""
        SELECT DISTINCT {column} AS cat
        FROM {table}
        ORDER BY {column} ASC
    """)
    with engine.connect() as conn:
        return list(conn.execute(sql).scalars())

# ---------- LLM ----------
def call_llm_allowed(cfg, distinct_categories: List[str]) -> List[str]:
    client = OpenAI(api_key=cfg["OPENAI_API_KEY"])
    user_prompt = f"""Distinct categories:
{chr(10).join(f"- {c}" for c in distinct_categories)}

Output ONLY the canonical categories, one per line."""
    resp = client.chat.completions.create(
        model=cfg["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": ALLOWED_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    vals = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    # de-dup, preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    if "unknown" not in out:
        out.append("unknown")
    return out

# ---------- Helpers: allowed list mgmt ----------
def ensure_unknown(allowed: List[str]) -> None:
    if "unknown" not in allowed:
        allowed.append("unknown")

def allowed_index(allowed: List[str], token: str) -> int:
    """Return 0-based index for name or 'number' string. -1 if not found."""
    s = token.strip()
    if s.isdigit():
        idx = int(s) - 1
        return idx if 0 <= idx < len(allowed) else -1
    s_lc = s.lower()
    for i, a in enumerate(allowed):
        if a.lower() == s_lc:
            return i
    return -1

def print_allowed_numbered(allowed: List[str]):
    print("\n=== Allowed canonical categories ===")
    for i, c in enumerate(allowed, start=1):
        print(f"{i:>3}. {c}")
    print("(Type NAME or NUMBER. Commands: !add-cat, !delete-cat, !change-cat)")

def rebuild_completer(allowed: List[str]) -> WordCompleter:
    # Provide both names and numbers as completion words
    words = list(allowed) + [str(i) for i in range(1, len(allowed) + 1)] + [
        "!add-cat", "!delete-cat", "!change-cat"
    ]
    # WordCompleter is case-insensitive by default if ignore_case=True
    return WordCompleter(words, ignore_case=True)

# ---------- Mapping + commands ----------
def resolve_user_choice(inp: str, allowed: List[str]) -> str | None:
    s = inp.strip()
    if not s or s.startswith("!"):
        return None
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(allowed):
            return allowed[idx - 1]
        return None
    s_lc = s.lower()
    for a in allowed:
        if s_lc == a.lower():
            return a
    return None

def cmd_add_cat(args: List[str], allowed: List[str]) -> Tuple[bool, str]:
    if not args:
        return False, "Usage: !add-cat <name>"
    name = " ".join(args).strip().lower()
    if not name:
        return False, "Category name cannot be empty."
    if any(name == a.lower() for a in allowed):
        return False, f"Category '{name}' already exists."
    allowed.append(name)
    ensure_unknown(allowed)
    return True, f"Added category '{name}'."

def cmd_delete_cat(args: List[str], allowed: List[str], mapping: Dict[str, str]) -> Tuple[bool, str, List[str]]:
    """Delete by name or number. Returns (ok, msg, affected_originals) where affected are unmapped to redo."""
    if not args:
        return False, "Usage: !delete-cat <name|number>", []
    idx = allowed_index(allowed, args[0])
    if idx < 0:
        return False, f"Unknown category '{args[0]}'.", []
    name = allowed[idx]
    if name.lower() == "unknown":
        return False, "Cannot delete 'unknown'.", []
    # Remove
    removed = allowed.pop(idx)
    ensure_unknown(allowed)
    # Invalidate mappings that targeted the removed name
    affected = [o for o, t in mapping.items() if t.lower() == removed.lower()]
    for o in affected:
        mapping.pop(o, None)  # force re-ask
    return True, f"Deleted category '{removed}'. Cleared {len(affected)} mapping(s) to it.", affected

def cmd_change_cat(args: List[str], allowed: List[str], mapping: Dict[str, str]) -> Tuple[bool, str]:
    if len(args) < 2:
        return False, "Usage: !change-cat <name|number> <new-name>"
    idx = allowed_index(allowed, args[0])
    if idx < 0:
        return False, f"Unknown category '{args[0]}'."
    new_name = " ".join(args[1:]).strip().lower()
    if not new_name:
        return False, "New name cannot be empty."
    old_name = allowed[idx]
    if old_name.lower() == "unknown":
        return False, "Cannot rename 'unknown'."
    # If target already exists, just consolidate to that name
    exists_idx = allowed_index(allowed, new_name)
    allowed[idx] = new_name
    # Update existing mappings that pointed to old_name
    changed = 0
    for k, v in list(mapping.items()):
        if v.lower() == old_name.lower():
            mapping[k] = new_name
            changed += 1
    ensure_unknown(allowed)
    return True, f"Renamed '{old_name}' -> '{new_name}'. Updated {changed} mapping(s)."

def interactive_label_invalid(
    invalid_originals: List[str],
    allowed: List[str],
    mapping: Dict[str, str],
):
    if not invalid_originals:
        return

    completer = rebuild_completer(allowed)
    i = 0
    while i < len(invalid_originals):
        orig = invalid_originals[i]
        # refresh completer each loop (in case the list changed)
        completer = rebuild_completer(allowed)
        ans = prompt(f"{orig} ==> ", completer=completer).strip()

        # Commands
        if ans.startswith("!"):
            parts = ans.split()
            cmd, args = parts[0], parts[1:]
            if cmd == "!add-cat":
                ok, msg = cmd_add_cat(args, allowed)
                if not ok:
                    print(colored(msg, "red"))
                else:
                    print(msg)
                # re-prompt same original
                continue
            elif cmd == "!delete-cat":
                ok, msg, affected = cmd_delete_cat(args, allowed, mapping)
                if not ok:
                    print(colored(msg, "red"))
                else:
                    print(msg)
                    # Any affected originals that were already labeled should be re-labeled:
                    for a in affected:
                        if a not in invalid_originals:
                            invalid_originals.append(a)
                # re-prompt same original
                continue
            elif cmd == "!change-cat":
                ok, msg = cmd_change_cat(args, allowed, mapping)
                if not ok:
                    print(colored(msg, "red"))
                else:
                    print(msg)
                # re-prompt same original
                continue
            else:
                print(colored("Unknown command. Available: !add-cat, !delete-cat, !change-cat", "red"))
                continue

        # Choice resolution
        target = resolve_user_choice(ans, allowed)
        if target is None:
            print(colored("Invalid choice. Type a valid NAME or NUMBER, or use a !command.", "red"))
            continue

        mapping[orig] = target
        i += 1  # advance to next invalid

# ---------- Apply / Output ----------
def apply_updates_one_by_one(engine, table: str, column: str, mapping: Dict[str, str], dry_run: bool):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rollback_path = f"rollback_{table}_{column}_{ts}.sql"
    total = 0
    with open(rollback_path, "w", encoding="utf-8") as rb:
        rb.write(f"-- rollback helper for {table}.{column} at {ts}Z\nSTART TRANSACTION;\n")
        with engine.begin() as conn:
            for orig, new in mapping.items():
                if orig == new:
                    continue
                if dry_run:
                    print(f"[dry-run] {orig} -> {new}")
                else:
                    res = conn.execute(
                        text(f"UPDATE {table} SET {column} = :new WHERE {column} = :orig"),
                        {"new": new, "orig": orig}
                    )
                    total += res.rowcount or 0
                rb.write(f"-- Inspect {orig} -> {new}\nSELECT id, {column} FROM {table} WHERE {column}='{new}';\n")
        rb.write("COMMIT;\n")
    return total, rollback_path

def write_stable_mapping_tsv(mapping: Dict[str, str], table: str, column: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"category_mapping_{table}_{column}_{ts}.tsv"
    items = sorted(mapping.items(), key=lambda kv: (kv[0].lower(), kv[0]))
    with open(path, "w", encoding="utf-8") as f:
        for orig, tgt in items:
            f.write(f"{orig}\t{tgt}\n")
    return path

def yes_no_prompt(message: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    while True:
        ans = prompt(f"{message} [{default}] ").strip()
        if not ans:
            return default_yes
        if ans.lower() in {"y", "yes"}:
            return True
        if ans.lower() in {"n", "no"}:
            return False
        print(colored("Please type 'y' or 'n'.", "red"))

# ---------- Main ----------
def main():
    cfg = load_config()
    engine = connect_db(cfg)

    # 1) DISTINCT
    originals = fetch_distinct(engine, cfg["TABLE"], cfg["COLUMN"])
    if not originals:
        print("No categories found.")
        sys.exit(0)

    # 2) ALLOWED (LLM once)
    allowed = call_llm_allowed(cfg, originals)
    ensure_unknown(allowed)

    # 3) Auto-map valid; collect invalids
    allowed_lc_to_canon = {a.lower(): a for a in allowed}
    auto_map: Dict[str, str] = {}
    invalid: List[str] = []
    for o in originals:
        a = allowed_lc_to_canon.get(o.lower())
        if a is not None:
            auto_map[o] = a
        else:
            invalid.append(o)

    print(f"\nFound {len(originals)} unique categories.")
    print(f"- Auto-mapped valid: {len(auto_map)}")
    print(f"- Need labeling (invalid): {len(invalid)}")
    print_allowed_numbered(allowed)

    # 4) Interactive labeling for the invalids
    mapping: Dict[str, str] = dict(auto_map)  # start with automapped
    interactive_label_invalid(invalid, allowed, mapping)

    # 5) Summary
    print("\n=== Summary (first 25 shown) ===")
    for i, (o, t) in enumerate(sorted(mapping.items(), key=lambda kv: (kv[0].lower(), kv[0]))):
        if i < 25:
            print(f"{o} ==> {t}")
    if len(mapping) > 25:
        print(f"... (+{len(mapping)-25} more)")

    # 6) Write stable TSV
    tsv_path = write_stable_mapping_tsv(mapping, cfg["TABLE"], cfg["COLUMN"])
    print(f"\nWrote mapping TSV: {tsv_path}")

    # 7) Apply?
    if yes_no_prompt("Apply updates to the database now?", default_yes=True):
        updated, rb = apply_updates_one_by_one(
            engine, cfg["TABLE"], cfg["COLUMN"], mapping, dry_run=False
        )
        print("\n=== DB Result ===")
        print(f"rows updated: {updated}")
        print(f"rollback file: {rb}")
    else:
        print("\nSkipped applying DB changes (dry run complete).")

if __name__ == "__main__":
    main()
