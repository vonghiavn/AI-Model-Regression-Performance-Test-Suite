import time
import torch
import psutil

def measure_performance(model, device, input_data, iterations=30):
    """
    Measure latency, throughput, and memory usage.
    """
    timings = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            if isinstance(input_data, dict):
                model(**input_data)
            else:
                model(input_data)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)

    avg_latency_ms = (sum(timings) / len(timings)) * 1000
    fps = 1000 / avg_latency_ms

    gpu_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device == "cuda"
        else 0
    )

    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    return {
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
        "gpu_mem_mb": round(gpu_mem, 2),
        "cpu_mem_mb": round(cpu_mem, 2)
    }
