import json
from datetime import datetime
from pathlib import Path

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def write_report(test_name, device, metrics, regressions):
    report = {
        "test_name": test_name,
        "device": device,
        "status": "PASS" if not regressions else "FAIL",
        "metrics": metrics,
        "regressions": regressions,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(REPORT_DIR / "latest_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
