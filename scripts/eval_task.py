#!/usr/bin/env python3
"""
Evaluation: after a run, check that incident_report.md and server_health.png exist
and that the report contains expected keywords (spike, latency, etc.).
Run from repo root: python scripts/eval_task.py [work_dir]
"""
import os
import sys

def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "project_workspace"
    report_path = os.path.join(work_dir, "incident_report.md")
    plot_path = os.path.join(work_dir, "server_health.png")

    ok = True
    if not os.path.isfile(report_path):
        print(f"FAIL: {report_path} not found")
        ok = False
    else:
        text = open(report_path).read().lower()
        for kw in ["spike", "latency"]:
            if kw not in text:
                print(f"FAIL: keyword '{kw}' not in report")
                ok = False
        if ok:
            print("report: OK (contains expected keywords)")

    if not os.path.isfile(plot_path):
        print(f"FAIL: {plot_path} not found")
        ok = False
    else:
        print("plot: OK")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
