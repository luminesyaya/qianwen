"""结构化抽取评测 + Alias 归一化 + 结构化质量分析"""
import json, re, torch, argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ALIAS_MAP = {
    # entity → entity 的变体
    "entity": ["entity", "entity_name", "name", "实体", "名称", "实体名", "text"],
    "entity_type": ["entity_type", "type", "类型", "类别"],
    # relations 子字段
    "head": ["head", "subject", "主体", "头实体", "head_entity"],
    "relation": ["relation", "relationship", "predicate", "关系", "关联"],
    "tail": ["tail", "object", "客体", "尾实体", "tail_entity"],
    # 顶层 key
    "entities": ["entities", "entity_list", "实体列表"],
    "relations": ["relations", "relation_list", "关系列表"],
}


def extract_json(text):
    candidates = [text.strip()]
    for m in re.finditer(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL):
        candidates.append(m.group(1).strip())
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


def normalize_keys(obj, alias_map):
    """递归归一化 JSON key"""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            canonical = k
            for ck, aliases in alias_map.items():
                if k in aliases:
                    canonical = ck
                    break
            new[canonical] = normalize_keys(v, alias_map)
        return new
    elif isinstance(obj, list):
        return [normalize_keys(item, alias_map) for item in obj]
    return obj


def compute_metrics(pred, gold):
    """完整指标：parse + strict + alias_strict"""
    if pred is None:
        return {"parse": 0, "strict": 0, "alias_strict": 0}

    parse_ok = 1

    # Strict: keys 完全一致 + 数组长度一致
    strict_ok = 0
    if isinstance(pred, dict) and isinstance(gold, dict):
        if set(pred.keys()) == set(gold.keys()):
            same_len = all(len(pred.get(k, [])) == len(gold.get(k, []))
                           for k in pred if isinstance(pred.get(k), list)
                           and isinstance(gold.get(k), list))
            strict_ok = 1 if same_len else 0

    # Alias-Strict 原始版
    alias_ok = 0
    if isinstance(pred, dict):
        has_rel = "relations" in pred and isinstance(pred["relations"], list)
        has_ent = "entities" in pred and isinstance(pred["entities"], list)
        alias_ok = 1 if has_rel and has_ent else 0

    return {"parse": parse_ok, "strict": strict_ok, "alias_strict": alias_ok}


def compute_alias_strict(pred, gold):
    """Alias 归一化后的 Strict"""
    p_norm = normalize_keys(pred, ALIAS_MAP)
    g_norm = normalize_keys(gold, ALIAS_MAP)
    if not isinstance(p_norm, dict) or not isinstance(g_norm, dict):
        return 0
    if set(p_norm.keys()) == set(g_norm.keys()):
        return 1
    return 0


def compute_structural_quality(pred):
    """结构化质量：实体做 key 率、全中文字段率、字段重叠率"""
    if not isinstance(pred, dict):
        return {"entity_key_rate": 0, "chinese_field_rate": 0, "field_overlap": 0}

    top_keys = list(pred.keys())

    # 实体做 key 率：顶层 key 是否包含中文实体名
    entity_like = sum(1 for k in top_keys if any('一' <= ch <= '鿿' for ch in k))
    entity_key_rate = entity_like / len(top_keys) if top_keys else 0

    # 全中文字段率：递归统计所有 key
    all_keys = []

    def collect_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                all_keys.append(k)
                collect_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_keys(item)
    collect_keys(pred)
    chinese_keys = sum(1 for k in all_keys if all('一' <= ch <= '鿿' or ch in '_/-' for ch in k))
    chinese_field_rate = chinese_keys / len(all_keys) if all_keys else 0

    # 字段重叠率：entities/relations 子字段与 schema 的重叠
    schema_keys = {"entity", "entity_type", "head", "relation", "tail"}
    pred_keys = set(all_keys)
    field_overlap = len(pred_keys & schema_keys) / len(schema_keys) if schema_keys else 0

    return {"entity_key_rate": round(entity_key_rate, 3),
            "chinese_field_rate": round(chinese_field_rate, 3),
            "field_overlap": round(field_overlap, 3)}


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

    results = {"parse": 0, "strict": 0, "alias_strict": 0, "alias_norm": 0}
    quality = {"entity_key_rate": 0, "chinese_field_rate": 0, "field_overlap": 0}
    total = 0

    for i, sample in enumerate(samples):
        if sample.get("task") != "ie_extraction": continue

        text = sample["text"]
        system_msg = ""; user_msg = ""; gold_text = ""
        for p in text.split("<|im_start|>"):
            if p.startswith("system\n"): system_msg = p[7:].replace("<|im_end|>", "").strip()
            elif p.startswith("user\n"): user_msg = p[5:].replace("<|im_end|>", "").strip()
            elif p.startswith("assistant\n"): gold_text = p[10:].replace("<|im_end|>", "").strip()

        if not user_msg: continue
        gold = extract_json(gold_text)
        if gold is None: continue

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
        m["alias_norm"] = compute_alias_strict(pred, gold) if pred else 0
        q = compute_structural_quality(pred) if pred else {"entity_key_rate": 0, "chinese_field_rate": 0, "field_overlap": 0}

        if total < 3 and pred:
            print(f"\n  [样例 {total+1}] Strict: {m['strict']}  Alias: {m['alias_strict']}  "
                  f"Alias-Norm: {m['alias_norm']}  质量: {q}")

        for k in results: results[k] += m[k]
        for k in quality: quality[k] += q[k]
        total += 1

        if total % 20 == 0:
            print(f"  [{total}] Parse: {results['parse']/total*100:.0f}%  "
                  f"Strict: {results['strict']/total*100:.0f}%  "
                  f"Alias: {results['alias_strict']/total*100:.0f}%  "
                  f"Alias-Norm: {results['alias_norm']/total*100:.0f}%")

    print(f"\n{'='*60}")
    print(f"评测完成 ({total} 条 ie_extraction)")
    print(f"  Parse%:            {results['parse']/total*100:.1f}%" if total else "  N/A")
    print(f"  Strict%:           {results['strict']/total*100:.1f}%")
    print(f"  Alias-Strict%:     {results['alias_strict']/total*100:.1f}%")
    print(f"  Alias-Norm-Strict%:{results['alias_norm']/total*100:.1f}%")
    print(f"  --- 结构化质量 ---")
    for k in quality:
        print(f"  {k}: {quality[k]/total:.3f}" if total else f"  {k}: N/A")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/qwen_lora/final")
    p.add_argument("--data_path", default="data/clean/valid.jsonl")
    p.add_argument("--max-samples", type=int, default=None)
    evaluate(**vars(p.parse_args()))
