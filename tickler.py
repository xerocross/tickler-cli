#!/usr/bin/env python3

import json, os, sys, argparse, uuid, re
from typing import Optional, Tuple, List, Dict
import datetime
from dataclasses import dataclass

TICKLER_DIR = os.path.expanduser("~/Documents/Odin/tickler")
TICKLER_FILE = os.path.join(TICKLER_DIR, "tickler.jsonl")


@dataclass
class TicklerEvent:
    type: str
    id: str
    text: Optional[str]
    when: Optional[str]
    created_at: Optional[str]
    due_at: Optional[str]


DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%b %d %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%d %B %Y",
    "%Y %b %d",
    "%Y %B %d",
    "%m/%d/%Y",
]

def ensure_storage():
    os.makedirs(TICKLER_DIR, exist_ok=True)
    if not os.path.exists(TICKLER_FILE):
        with open(TICKLER_FILE, "w", encoding="utf-8"):
            pass

def gen_id() -> str:
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    base36 = ""
    n = now
    if n == 0:
        base36 = "0"
    else:
        while n:
            n, r = divmod(n, 36)
            base36 = alphabet[r] + base36
    return f"{base36}-{uuid.uuid4().hex[:6]}"

def parse_interval(s: str) -> Tuple[str, int]:
    m = re.fullmatch(r"\s*\+?\s*(\d+)\s*(y|m|w|d|h|min)\s*", s, flags=re.I)
    if not m:
        raise ValueError("Invalid interval. Use like '1y', '2m', '3w', '10d', '6h', '30min'.")
    value = int(m.group(1))
    unit = m.group(2).lower()
    return unit, value

def add_months(start: datetime.datetime, months: int) -> datetime.datetime:
    year = start.year + (start.month - 1 + months) // 12
    month = (start.month - 1 + months) % 12 + 1
    if month in (1,3,5,7,8,10,12):
        last_day = 31
    elif month in (4,6,9,11):
        last_day = 30
    else:
        y = year
        is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        last_day = 29 if is_leap else 28
    day = min(start.day, last_day)
    return start.replace(year=year, month=month, day=day)

def add_years(start: datetime.datetime, years: int) -> datetime.datetime:
    try:
        return start.replace(year=start.year + years)
    except ValueError:
        if start.month == 2 and start.day == 29:
            return start.replace(year=start.year + years, day=28)
        raise

def apply_interval(start: datetime.datetime, unit: str, value: int) -> datetime.datetime:
    if unit == "min":
        return start + datetime.timedelta(minutes=value)
    if unit == "h":
        return start + datetime.timedelta(hours=value)
    if unit == "d":
        return start + datetime.timedelta(days=value)
    if unit == "w":
        return start + datetime.timedelta(weeks=value)
    if unit == "m":
        return add_months(start, value)
    if unit == "y":
        return add_years(start, value)
    raise ValueError("Unsupported interval unit")

def parse_date_string(s: str, now: Optional[datetime.datetime] = None) -> datetime.datetime:
    now = now or datetime.datetime.now()
    s = s.strip()

    if s.startswith("+"):
        unit, value = parse_interval(s)
        return apply_interval(now, unit, value)

    m = re.fullmatch(r"in\s+(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?)", s, flags=re.I)
    if m:
        value = int(m.group(1))
        word = m.group(2).lower()
        if word.startswith("year"):   return apply_interval(now, "y", value)
        if word.startswith("month"):  return apply_interval(now, "m", value)
        if word.startswith("week"):   return apply_interval(now, "w", value)
        if word.startswith("day"):    return apply_interval(now, "d", value)
        if word.startswith("hour"):   return apply_interval(now, "h", value)
        if word.startswith("minute"): return apply_interval(now, "min", value)

    for pat in DATE_PATTERNS:
        try:
            return datetime.datetime.strptime(s, pat)
        except ValueError:
            continue

    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        pass

    raise ValueError(f"Could not parse date/time: {s}")

def load_events():
    ensure_storage()
    events = []
    with open(TICKLER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events

def append_event(ev: dict):
    ensure_storage()
    with open(TICKLER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def snapshot_state() -> Dict[str, dict]:
    events = load_events()
    state: Dict[str, dict] = {}
    for ev in events:
        typ = ev.get("type")

        if typ == "add":
            rid = ev["id"]
            state[rid] = {
                "id": rid,
                "text": ev["text"],
                "created_at": ev["created_at"],
                "archived": False,
                "noted_history": [],
            }
            if "due_at" in ev:
                state[rid]["due_at"] = ev["due_at"]
            if "every_unit" in ev:
                state[rid]["every_unit"] = ev["every_unit"]
                state[rid]["every_value"] = ev["every_value"]
                state[rid]["next_due_at"] = ev["next_due_at"]

        elif typ == "snooze":
            rid = ev["id"]
            rec = state.get(rid)
            if rec and not rec.get("archived"):
                if "due_at" in rec:
                    rec["due_at"] = ev["new_due_at"]
                elif "next_due_at" in rec:
                    rec["next_due_at"] = ev["new_due_at"]

        elif typ == "noted":
            rid = ev["id"]
            if rid in state:
                state[rid]["noted_history"].append(ev["when"])
                if ev.get("recurring"):
                    if "next_due_at" in ev:
                        state[rid]["next_due_at"] = ev["next_due_at"]
                else:
                    state[rid]["archived"] = True

        elif typ == "archive":
            rid = ev["id"]
            if rid in state:
                state[rid]["archived"] = True

    return state

def add_one_time(due_str: str, text: str, now: Optional[datetime.datetime] = None):
    now = now or datetime.datetime.now()
    due_dt = parse_date_string(due_str, now=now)
    rid = gen_id()
    ev = {
        "type": "add",
        "id": rid,
        "text": text,
        "created_at": now.isoformat(timespec="seconds"),
        "due_at": due_dt.isoformat(timespec="seconds"),
    }
    append_event(ev)
    print(f"[added] {rid}  due {due_dt.strftime('%Y-%m-%d %H:%M')}  {text}")

def add_recurring(every_str: str, text: str, now: Optional[datetime.datetime] = None):
    now = now or datetime.datetime.now()
    unit, value = parse_interval(every_str)
    next_due = apply_interval(now, unit, value)
    rid = gen_id()
    ev = {
        "type": "add",
        "id": rid,
        "text": text,
        "created_at": now.isoformat(timespec="seconds"),
        "every_unit": unit,
        "every_value": value,
        "next_due_at": next_due.isoformat(timespec="seconds"),
    }
    append_event(ev)
    print(f"[added recurring] {rid}  every {value}{unit}  next {next_due.strftime('%Y-%m-%d %H:%M')}  {text}")

def list_due(now: Optional[datetime.datetime] = None, show_all=False):
    now = now or datetime.datetime.now()
    state = snapshot_state()
    due_items = []
    for rec in state.values():
        if rec.get("archived"):
            continue
        if "due_at" in rec:
            due_dt = datetime.datetime.fromisoformat(rec["due_at"])
            if show_all or due_dt <= now:
                due_items.append(("one", due_dt, rec))
        elif "next_due_at" in rec:
            due_dt = datetime.datetime.fromisoformat(rec["next_due_at"])
            if show_all or due_dt <= now:
                due_items.append(("recurring", due_dt, rec))

    due_items.sort(key=lambda x: x[1])
    if not due_items:
        print("Nothing due. ✨")
        return

    print(f"Due items ({len(due_items)}):")
    for kind, when, rec in due_items:
        when_str = when.strftime("%Y-%m-%d %H:%M")
        prefix = "[R]" if kind == "recurring" else "[ ]"
        print(f"{prefix} {rec['id']}  {when_str}  {rec['text']}")

def mark_noted(rid: str, now: Optional[datetime.datetime] = None):
    now = now or datetime.datetime.now()
    state = snapshot_state()
    if rid not in state:
        print(f"ID not found: {rid}", file=sys.stderr)
        sys.exit(1)
    rec = state[rid]
    if rec.get("archived"):
        print(f"Already archived: {rid}")
        return

    if "every_unit" in rec:
        current_next = datetime.datetime.fromisoformat(rec["next_due_at"])
        next_next = apply_interval(current_next, rec["every_unit"], rec["every_value"])
        ev = {
            "type": "noted",
            "id": rid,
            "when": now.isoformat(timespec="seconds"),
            "recurring": True,
            "next_due_at": next_next.isoformat(timespec="seconds"),
        }
        append_event(ev)
        print(f"[noted] {rid}  next {next_next.strftime('%Y-%m-%d %H:%M')}")
    else:
        ev = {
            "type": "noted",
            "id": rid,
            "when": now.isoformat(timespec="seconds"),
            "recurring": False,
        }
        append_event(ev)
        print(f"[archived] {rid}")

def snooze_item_to(rid: str, new_dt: datetime.datetime, now: Optional[datetime.datetime] = None):
    now = now or datetime.datetime.now()
    if new_dt <= now:
        print("Snooze target must be in the future.", file=sys.stderr)
        sys.exit(2)

    state = snapshot_state()
    if rid not in state:
        print(f"ID not found: {rid}", file=sys.stderr)
        sys.exit(1)
    rec = state[rid]
    if rec.get("archived"):
        print(f"Cannot snooze archived item: {rid}", file=sys.stderr)
        sys.exit(1)

    if "due_at" in rec:
        old_dt = datetime.datetime.fromisoformat(rec["due_at"])
        ev = {
            "type": "snooze",
            "id": rid,
            "when": now.isoformat(timespec="seconds"),
            "old_due_at": old_dt.isoformat(timespec="seconds"),
            "new_due_at": new_dt.isoformat(timespec="seconds"),
        }
        append_event(ev)
        print(f"[snoozed] {rid}  from {old_dt.strftime('%Y-%m-%d %H:%M')} → {new_dt.strftime('%Y-%m-%d %H:%M')}")
    elif "next_due_at" in rec:
        old_dt = datetime.datetime.fromisoformat(rec["next_due_at"])
        ev = {
            "type": "snooze",
            "id": rid,
            "when": now.isoformat(timespec="seconds"),
            "old_due_at": old_dt.isoformat(timespec="seconds"),
            "new_due_at": new_dt.isoformat(timespec="seconds"),
        }
        append_event(ev)
        print(f"[snoozed R] {rid}  from {old_dt.strftime('%Y-%m-%d %H:%M')} → {new_dt.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("Record has no due fields to snooze.", file=sys.stderr)
        sys.exit(1)

def print_path():
    ensure_storage()
    print(TICKLER_FILE)

# ---------- FIND helpers ----------

def collect_records(include_historical: bool) -> List[dict]:
    """
    Returns a list of records with keys: id, text, when_dt, when_str, kind ('one'|'recurring'), archived(bool)
    - When include_historical=False: only not-archived (outstanding) items
    - When include_historical=True: includes archived items too (best-effort 'when' from due/next_due)
    """
    now = datetime.datetime.now()
    state = snapshot_state()
    recs: List[dict] = []
    for rec in state.values():
        archived = bool(rec.get("archived"))
        if not include_historical and archived:
            continue

        if "due_at" in rec:
            dt = datetime.datetime.fromisoformat(rec["due_at"])
            kind = "one"
        elif "next_due_at" in rec:
            dt = datetime.datetime.fromisoformat(rec["next_due_at"])
            kind = "recurring"
        else:
            # Edge: record existed but has neither? Skip unless historical.
            if include_historical:
                dt = now
                kind = "one"
            else:
                continue

        recs.append({
            "id": rec["id"],
            "text": rec.get("text", ""),
            "when_dt": dt,
            "when_str": dt.strftime("%Y-%m-%d %H:%M"),
            "kind": kind,
            "archived": archived,
        })
    return recs

def format_line(r: dict) -> str:
    prefix = "[R]" if r["kind"] == "recurring" else "[ ]"
    if r.get("archived"):
        prefix = "[-]"  # historical marker
    return f"{prefix} {r['id']}  {r['when_str']}  {r['text']}"

def best_match(query: str, records: List[dict]) -> Optional[dict]:
    """
    Return the single best match using rapidfuzz if available, else a simple heuristic.
    """
    if not records:
        return None

    try:
        from rapidfuzz import process, fuzz
        choices = {f"{rec['id']} {rec['text']}": rec for rec in records}
        label, score, _ = process.extractOne(
            query,
            choices.keys(),
            scorer=fuzz.WRatio
        )
        return choices[label] if label is not None else None
    except Exception:
        # Fallback: naive contains/id-prefix match, else longest common substring-ish
        query_l = query.lower().strip()
        # 1) ID prefix exact
        for rec in records:
            if rec["id"].startswith(query_l):
                return rec
        # 2) Text contains
        contains = [rec for rec in records if query_l in rec["text"].lower()]
        if contains:
            # prefer nearer due date
            contains.sort(key=lambda r: r["when_dt"])
            return contains[0]
        # 3) Levenshtein-light: smallest abs length diff as a crude proxy
        return min(records, key=lambda r: abs(len(r["text"]) - len(query_l)))

def interactive_find(include_historical: bool):
    """
    Minimal fuzzy REPL using prompt_toolkit (optional dependency).
    Type to filter; ENTER selects and prints the best match. Ctrl-C/D to exit.
    """
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
    except Exception:
        print("Interactive mode requires 'prompt_toolkit'. Try: pip install prompt_toolkit", file=sys.stderr)
        sys.exit(2)

    records = collect_records(include_historical)
    if not records:
        print("No records to search.")
        return

    # Labels to display/complete on
    labels = [f"{r['id']}  {r['text']}" if r['text'] else r['id'] for r in records]
    completer = FuzzyCompleter(WordCompleter(labels, ignore_case=True))
    while True:
        try:
            q = prompt("find> ", completer=completer, complete_while_typing=True)
        except (KeyboardInterrupt, EOFError):
            print()  # newline after Ctrl-C/D
            break
        q = q.strip()
        if not q:
            continue
        match = best_match(q, records)
        if match:
            print(format_line(match))
        else:
            print("No match.")

# -----------------------------
# CLI (subcommands)
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tickler", description="Append-only JSONL tickler file")
    sub = parser.add_subparsers(dest="cmd")

    # due (default)
    sub.add_parser("due", help="List reminders due now")

    # list-all
    sub.add_parser("list-all", help="List all outstanding reminders (including future ones)")

    # add
    p_add = sub.add_parser("add", help="Add a reminder")
    g_add = p_add.add_mutually_exclusive_group(required=True)
    g_add.add_argument("-r", "--remind", metavar="WHEN",
                       help="One-time reminder at WHEN (e.g. '+2m', '2026 Jan 1', '2026-01-01')")
    g_add.add_argument("-e", "--remind-every", metavar="INTERVAL",
                       help="Recurring reminder like '1y', '2m', '3w', '10d', '6h', '30min'")
    p_add.add_argument("text", nargs="+", help="Reminder text")

    # mark
    p_mark = sub.add_parser("mark", help="Mark reminder as done/noted")
    p_mark.add_argument("id", help="Reminder ID")

    # snooze
    p_snooze = sub.add_parser("snooze", help="Snooze a reminder")
    g_snooze = p_snooze.add_mutually_exclusive_group(required=True)
    g_snooze.add_argument("-f", "--for", dest="for_", metavar="INTERVAL",
                          help="Snooze for an interval (e.g. '2h', '30min', '3d')")
    g_snooze.add_argument("-u", "--until", metavar="WHEN",
                          help="Snooze until a date/time (e.g. '2025-11-03 09:30', '+2d', 'in 3 hours')")
    p_snooze.add_argument("id", help="Reminder ID")

    # find
    p_find = sub.add_parser("find", help="Fuzzy search reminders")
    p_find.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive fuzzy finder (REPL)")
    p_find.add_argument("-H", "--historical", action="store_true",
                        help="Search all items (historical + future). Default: outstanding only.")
    p_find.add_argument("query", nargs="*",
                        help="Search query (omit in interactive mode)")

    # path
    sub.add_parser("path", help="Print path to tickler JSONL")

    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default: "tickler" -> behave like "tickler due"
    if args.cmd is None:
        list_due(now=datetime.datetime.now(), show_all=False)
        return

    now = datetime.datetime.now()

    if args.cmd == "due":
        list_due(now=now, show_all=False)
        return

    if args.cmd == "list-all":
        list_due(now=now, show_all=True)
        return

    if args.cmd == "add":
        text = " ".join(args.text).strip()
        if args.remind is not None:
            add_one_time(args.remind, text, now=now)
        else:
            add_recurring(args.remind_every, text, now=now)
        return

    if args.cmd == "mark":
        mark_noted(args.id, now=now)
        return

    if args.cmd == "snooze":
        rid = args.id
        if args.for_:
            try:
                unit, value = parse_interval(args.for_)
                new_dt = apply_interval(now, unit, value)
            except Exception as e:
                print(str(e), file=sys.stderr)
                sys.exit(2)
            snooze_item_to(rid, new_dt, now=now)
            return
        else:
            try:
                new_dt = parse_date_string(args.until, now=now)
            except Exception as e:
                print(str(e), file=sys.stderr)
                sys.exit(2)
            snooze_item_to(rid, new_dt, now=now)
            return

    if args.cmd == "find":
        include_hist = bool(args.historical)
        if args.interactive:
            interactive_find(include_hist)
            return
        # non-interactive requires a query
        if not args.query:
            print("Provide a query, or use -i for interactive mode.", file=sys.stderr)
            sys.exit(2)
        query = " ".join(args.query).strip()
        records = collect_records(include_hist)
        rec = best_match(query, records)
        if rec:
            print(format_line(rec))
            sys.exit(0)
        else:
            print("No match.", file=sys.stderr)
            sys.exit(1)

    if args.cmd == "path":
        print_path()
        return

if __name__ == "__main__":
    main()
