"""稳定性验证 — 对比 offline vs vLLM 的结构化输出"""
import json, time, requests

BASE = "http://localhost:8000"
MODEL = "outputs/qwen_merged"


def eval_vllm(data_path, constrained=False, max_samples=20):
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    samples = [s for s in samples if s.get("task") == "ie_extraction"][:max_samples]

    parse_ok, strict_ok, alias_ok = 0, 0, 0
    total, latencies = 0, []

    for s in samples:
        text = s["text"]
        system_msg = user_msg = ""
        for p in text.split("<|im_start|>"):
            if p.startswith("system\n"): system_msg = p[7:].replace("<|im_end|>", "").strip()
            elif p.startswith("user\n"): user_msg = p[5:].replace("<|im_end|>", "").strip()
        if not user_msg: continue

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_msg or "输出 JSON。"},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 512, "temperature": 0.0,
        }
        if constrained:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=60)
        elapsed = time.perf_counter() - t0

        try:
            reply = r.json()["choices"][0]["message"]["content"]
            pred = json.loads(reply) if reply.strip().startswith("{") else None
        except:
            pred = None

        if pred:
            parse_ok += 1
            has_rel = "relations" in pred and isinstance(pred["relations"], list)
            has_ent = "entities" in pred and isinstance(pred["entities"], list)
            if has_rel and has_ent: alias_ok += 1
            if has_rel and has_ent and set(pred.keys()) == {"entities", "relations"}: strict_ok += 1

        total += 1
        latencies.append(elapsed)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return {
        "parse_pct": parse_ok / total * 100 if total else 0,
        "strict_pct": strict_ok / total * 100 if total else 0,
        "alias_pct": alias_ok / total * 100 if total else 0,
        "avg_latency": round(avg_lat, 3),
        "total": total,
    }


if __name__ == "__main__":
    data = "data/clean/valid.jsonl"
    print("稳定性验证 (vLLM vs 离线)\n")
    for tag, constrained in [("Normal", False), ("Constrained (json_object)", True)]:
        r = eval_vllm(data, constrained=constrained, max_samples=20)
        print(f"  {tag}: Parse={r['parse_pct']:.0f}%  Strict={r['strict_pct']:.0f}%  "
              f"Alias={r['alias_pct']:.0f}%  Latency={r['avg_latency']}s  (n={r['total']})")
    print("\n离线参考: Parse=95%  Strict=24%  Alias=95%")
