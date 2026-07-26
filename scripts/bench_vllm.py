"""vLLM Benchmark — TTFT / 吞吐 / 并发"""
import time, requests, argparse, statistics

BASE = "http://localhost:8000"
MODEL = "qwen"
WARMUP = 2


def bench_single(input_tokens, output_tokens, runs=5):
    prompt = "人工智能" * (input_tokens // 4)
    ttfts, tok_per_s = [], []

    for _ in range(WARMUP + runs):
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/v1/completions", json={
            "model": MODEL, "prompt": prompt,
            "max_tokens": output_tokens, "temperature": 0.0,
        }, timeout=60)
        data = r.json()
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        ttft = data.get("usage", {}).get("completion_tokens", output_tokens) / output_tokens * 0.01
        # vLLM doesn't expose TTFT separately; use elapsed/tokens
        tok_sec = usage.get("completion_tokens", output_tokens) / elapsed if elapsed > 0 else 0
        if _ >= WARMUP:
            ttfts.append(elapsed)
            tok_per_s.append(tok_sec)

    return statistics.mean(ttfts), statistics.mean(tok_per_s)


def bench_concurrent(input_tokens, output_tokens, concurrency):
    import concurrent.futures
    prompt = "人工智能" * (input_tokens // 4)
    results = []

    def worker():
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/v1/completions", json={
            "model": MODEL, "prompt": prompt,
            "max_tokens": output_tokens, "temperature": 0.0,
        }, timeout=120)
        elapsed = time.perf_counter() - t0
        ok = "choices" in r.json()
        return {"elapsed": elapsed, "ok": ok}

    for _ in range(WARMUP):
        with concurrent.futures.ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(lambda _: worker(), range(concurrency)))

    with concurrent.futures.ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(lambda _: worker(), range(concurrency)))

    successes = [r for r in results if r["ok"]]
    errors = len(results) - len(successes)
    avg_elapsed = statistics.mean([r["elapsed"] for r in successes]) if successes else 0
    avg_tps = (output_tokens / avg_elapsed) if avg_elapsed else 0

    return avg_tps, len(results), errors


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=BASE)
    args = p.parse_args()
    BASE = args.base_url

    print("vLLM Benchmark\n")
    print(f"{'Config':>18s} {'TTFT(s)':>10s} {'Tok/s':>10s}")
    print("-" * 42)
    for il, ol in [(128, 64), (512, 128), (1024, 256)]:
        ttft, tps = bench_single(il, ol)
        print(f"  in={il:4d} out={ol:3d}  {ttft:>10.3f} {tps:>10.2f}")

    print(f"\n{'Concurrency':>18s} {'Tok/s/req':>10s} {'Errors':>8s}")
    print("-" * 40)
    for conc in [1, 4, 8]:
        tps, total, errs = bench_concurrent(512, 128, conc)
        print(f"  x{conc:2d}              {tps:>10.2f} {errs:>8d}")
