import json

def test_latest_report_pass():
    with open("reports/latest_report.json") as f:
        report = json.load(f)

    assert report["status"] == "PASS"
