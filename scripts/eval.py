"""结构化抽取评测 — Parse% / Strict% / Alias-Strict%"""
import json, re, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def extract_json(text):
    """严格 JSON 提取"""
    candidates = [text.strip()]

    # ```json ... ``` 代码块
    for m in re.finditer(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL):
        candidates.append(m.group(1).strip())

    # { ... } 匹配
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{': depth += 1; start = i if depth == 1 else start
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1]); start = None

    for cand in candidates:
        try: return json.loads(cand)
        except: continue
    return None


def compute_metrics(pred, gold):
    """严格指标"""
    if pred is None:
        return {"parse": 0, "strict": 0, "alias_strict": 0}

    parse_ok = 1  # 能到这里说明 extract_json 已成功

    # Strict: keys 完全一致 + 数组长度一致
    strict_ok = 0
    if isinstance(pred, dict) and isinstance(gold, dict):
        if set(pred.keys()) == set(gold.keys()):
            same_len = all(len(pred.get(k, [])) == len(gold.get(k, []))
                           for k in pred if isinstance(pred.get(k), list)
                           and isinstance(gold.get(k), list))
            strict_ok = 1 if same_len else 0

    # Alias-Strict: 至少有 relations 和 entities 两个数组
    alias_ok = 0
    if isinstance(pred, dict):
        has_rel = "relations" in pred and isinstance(pred["relations"], list)
        has_ent = "entities" in pred and isinstance(pred["entities"], list)
        alias_ok = 1 if has_rel and has_ent else 0

    return {"parse": parse_ok, "strict": strict_ok, "alias_strict": alias_ok}


@torch.no_grad()
def evaluate(model_path, data_path, max_samples=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16
    ).to(device)
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    samples = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    if max_samples: samples = samples[:max_samples]

    results = {"parse": 0, "strict": 0, "alias_strict": 0}; total = 0

    for i, sample in enumerate(samples):
        if sample.get("task") != "ie_extraction": continue

        # 从训练时期的 ChatML 拆出原始 system + user，保持一致
        text = sample["text"]
        system_msg = ""; user_msg = ""; gold_text = ""
        for p in text.split("<|im_start|>"):
            if p.startswith("system\n"): system_msg = p[7:].replace("<|im_end|>", "").strip()
            elif p.startswith("user\n"): user_msg = p[5:].replace("<|im_end|>", "").strip()
            elif p.startswith("assistant\n"): gold_text = p[10:].replace("<|im_end|>", "").strip()

        if not user_msg: continue
        gold = extract_json(gold_text)
        if gold is None: continue

        # 用训练时一模一样的 system + user
        messages = [
            {"role": "system", "content": system_msg or "你是一个信息抽取助手"},
            {"role": "user", "content": user_msg},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(rendered)], dtype=torch.long).to(device)

        out = model.generate(input_ids, max_new_tokens=1024, temperature=0.1, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][len(input_ids[0]):], skip_special_tokens=True)
        pred = extract_json(reply)

        m = compute_metrics(pred, gold)
        if total < 3 and pred is None:
            print(f"\n  [Parse 失败 {total+1}] reply: {reply[:300]}")
        elif total < 3:
            print(f"\n  [样例 {total+1}] pred keys: {list(pred.keys()) if pred else 'None'}")
            print(f"          gold keys: {list(gold.keys()) if gold else 'None'}")
        for k in results: results[k] += m[k]
        total += 1

        if total % 20 == 0:
            print(f"  [{total}] Parse: {results['parse']/total*100:.1f}%  "
                  f"Strict: {results['strict']/total*100:.1f}%  "
                  f"Alias: {results['alias_strict']/total*100:.1f}%")

    print(f"\n{'='*50}\n评测完成 ({total} 条 ie_extraction)")
    for k in ["parse", "strict", "alias_strict"]:
        print(f"  {k}: {results[k]/total*100:.1f}%" if total else f"  {k}: N/A")
    print(f"{'='*50}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/qwen_lora/final")
    p.add_argument("--data_path", default="data/clean/valid.jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    evaluate(args.model_path, args.data_path, args.max_samples)
