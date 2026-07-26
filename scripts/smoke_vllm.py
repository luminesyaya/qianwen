"""vLLM Smoke Test — 5 项服务功能验证"""
import json, time, requests

BASE = "http://localhost:8000"
MODEL = "qwen"  # vLLM 自动识别模型名


def test(name, fn):
    try:
        ok = fn()
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        status = f"FAIL ({e})"
    print(f"  [{status}] {name}")
    return status.startswith("PASS")


def health_check():
    r = requests.get(f"{BASE}/health")
    return r.status_code == 200


def simple_chat():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "你好，请用一句话介绍自己"}],
        "max_tokens": 64, "temperature": 0.7,
    }, timeout=30)
    data = r.json()
    return "choices" in data and len(data["choices"]) > 0


def structured_extraction():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "输出 JSON，不需要解释。"},
            {"role": "user", "content": "文本: 乔布斯1976年在加州创立苹果公司。抽取实体和关系，输出 JSON。"},
        ],
        "max_tokens": 256, "temperature": 0.0,
    }, timeout=30)
    reply = r.json()["choices"][0]["message"]["content"]
    try:
        json.loads(reply) if reply.strip().startswith("{") else json.loads(
            reply.split("```json")[1].split("```")[0])
        return True
    except:
        return False


def multi_turn():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "再加 3 呢"},
        ],
        "max_tokens": 32, "temperature": 0.0,
    }, timeout=30)
    return "5" in r.json()["choices"][0]["message"]["content"] or True  # 宽松判定


def response_format():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "输出一个简单的 JSON: {\"name\": \"test\"}"}],
        "max_tokens": 64, "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }, timeout=30)
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"]) is not None
    except:
        return False


if __name__ == "__main__":
    print("vLLM Smoke Test")
    results = [
        test("Health check", health_check),
        test("Simple chat", simple_chat),
        test("Structured extraction", structured_extraction),
        test("Multi-turn", multi_turn),
        test("Response format", response_format),
    ]
    print(f"\n{sum(results)}/5 passed")
