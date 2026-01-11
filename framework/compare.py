def compare_metrics(current, baseline, tolerance=0.10):
    """
    Compare current metrics with baseline metrics.
    """
    regressions = []

    for key, baseline_value in baseline.items():
        if key not in current:
            continue

        current_value = current[key]
        delta = (current_value - baseline_value) / baseline_value

        if abs(delta) > tolerance:
            regressions.append({
                "metric": key,
                "baseline": baseline_value,
                "current": current_value,
                "delta_percent": round(delta * 100, 2)
            })

    return regressions
