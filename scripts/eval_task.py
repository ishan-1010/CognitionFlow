#!/usr/bin/env python3
"""
Evaluation: after a run, verify that at least one artifact was generated
and that report files are well-formed (not empty, contain expected structure).
Run from repo root: python scripts/eval_task.py [work_dir]
"""
import os
import sys

EXPECTED_EXTENSIONS = {".md", ".png", ".json", ".py", ".csv", ".html", ".txt"}


def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "project_workspace"

    if not os.path.isdir(work_dir):
        print(f"FAIL: work_dir '{work_dir}' does not exist")
        sys.exit(1)

    artifacts = []
    for fname in os.listdir(work_dir):
        fpath = os.path.join(work_dir, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in EXPECTED_EXTENSIONS:
                artifacts.append((fname, fpath, ext))

    if not artifacts:
        print("FAIL: no artifacts found in work_dir")
        sys.exit(1)

    ok = True
    for name, path, ext in artifacts:
        size = os.path.getsize(path)
        if size == 0:
            print(f"FAIL: {name} is empty (0 bytes)")
            ok = False
        else:
            print(f"  OK: {name} ({size} bytes)")

        # For markdown reports, check they're not just raw data dumps
        if ext == ".md":
            text = open(path).read()
            if text.startswith("   ") and "|" not in text and "#" not in text:
                print(f"  WARN: {name} may contain raw data dump (no markdown structure)")

    print(f"\nTotal artifacts: {len(artifacts)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
