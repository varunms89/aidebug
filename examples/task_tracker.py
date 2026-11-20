"""Sample multi-bug Python script for testing `aidebug run examples/task_tracker.py`.

Contains 3 deliberate bugs so you can watch aidebug's multi-step fix loop:
1. Missing `json` import (NameError)
2. Typo in dict key `priorty` instead of `priority` (KeyError)
3. Adding string duration `"45"` to int total (TypeError)
"""

RAW_TASKS = """
[
  {"title": "Write unit tests", "priority": "high", "minutes": 30, "done": true},
  {"title": "Refactor parser", "priority": "medium", "minutes": "45", "done": false},
  {"title": "Update README", "priority": "low", "minutes": 15, "done": true}
]
"""


def load_tasks(raw: str) -> list:
    return json.loads(raw)


def summarize(tasks: list) -> None:
    high_priority = [t["title"] for t in tasks if t["priorty"] == "high"]
    total_minutes = sum(t["minutes"] for t in tasks)
    completed = sum(1 for t in tasks if t["done"])

    print(f"Total tasks: {len(tasks)} ({completed} done)")
    print(f"High priority: {', '.join(high_priority)}")
    print(f"Total estimated time: {total_minutes} minutes")


def main() -> None:
    tasks = load_tasks(RAW_TASKS)
    summarize(tasks)


if __name__ == "__main__":
    main()
