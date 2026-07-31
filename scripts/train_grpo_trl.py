"""
TRL GRPO — 在自有 Qwen SFT 模型上做结构化抽取对齐

远程 4090 上跑：
  pip install trl vllm
  python train_grpo_trl.py
"""

import json, re, torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ═══════════════════
# Reward 函数 — 和 PyTRIO 版一模一样
# ═══════════════════
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


def grade_answer(response, gold_str):
    """Parse + Alias + Strict → 0~1 连续分（Strict 权重最高，唯一有区分度的维度）"""
    pred = extract_json(response)
    gold = json.loads(gold_str) if isinstance(gold_str, str) else gold_str
    if pred is None:
        return 0.0

    score = 0.2  # Parse（SFT 已到天花板，降低权重）
    has_ent = "entities" in pred and isinstance(pred["entities"], list)
    has_rel = "relations" in pred and isinstance(pred["relations"], list)
    pred_ent_n = len(pred["entities"]) if has_ent else 0
    pred_rel_n = len(pred["relations"]) if has_rel else 0
    if has_ent and has_rel:
        score += 0.1  # Alias（也都过了）
        # Strict — 唯一有区分度的维度，权重大
        gold_ent_n = len(gold.get("entities", []))
        gold_rel_n = len(gold.get("relations", []))
        # 数量越接近 gold 分越高
        ent_err = abs(pred_ent_n - gold_ent_n) / max(gold_ent_n, 1)
        rel_err = abs(pred_rel_n - gold_rel_n) / max(gold_rel_n, 1)
        strict_score = max(0, 1 - (ent_err + rel_err) / 2)
        score += 0.5 * strict_score  # Strict 占 50% 权重
    # 长度惩罚：极端短或极端长都有问题
    if has_ent and pred_ent_n == 0 and gold_ent_n > 0:
        score = max(0, score - 0.3)  # 完全没抽到实体，重罚
    return score


def extract_gold(text):
    parts = text.split("<|im_start|>assistant\n")
    return json.loads(parts[-1].replace("<|im_end|>", "").strip())


# ═══════════════════
# ★ TRL reward 函数
# ═══════════════════
def reward_fn(prompts, completions, gold_str=None, **kwargs):
    """
    TRL 自动把 dataset 的额外列(gold_str)传给这里
    gold_str: list[str] — 每条 prompt 对应的 gold JSON 字符串
    """
    if gold_str is None:
        return [0.0] * len(completions)

    rewards = []
    num_gen = kwargs.get("num_generations", 4)

    for i, completion in enumerate(completions):
        prompt_idx = i // num_gen
        gs = gold_str[prompt_idx] if prompt_idx < len(gold_str) else ""

        if isinstance(completion, list):
            text = "".join(
                c.get("content", "") if isinstance(c, dict) else str(c)
                for c in completion
            )
        else:
            text = str(completion)

        rewards.append(grade_answer(text, gs) if gs else 0.0)

    return rewards


# ═══════════════════
# 数据加载
# ═══════════════════
def load_ie_dataset(path, max_samples=50):
    """数据集单独存 prompt(不含 gold) 和 gold_str"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s.get("task") == "ie_extraction":
                prompt = s["text"].split("<|im_start|>assistant\n")[0]
                prompt += "<|im_start|>assistant\n"
                gold = extract_gold(s["text"])
                samples.append({
                    "prompt": prompt,
                    "gold_str": json.dumps(gold, ensure_ascii=False)
                })
            if len(samples) >= max_samples:
                break
    return Dataset.from_list(samples)


# ═══════════════════
# 主函数
# ═══════════════════
def main():
    MODEL_PATH = "outputs/qwen_lora/final"  # 你的 SFT LoRA
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    DATA_PATH = "data/clean/train.jsonl"
    MAX_SAMPLES = 500

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model = model.merge_and_unload()  # 合并 LoRA，推理更快

    print("Loading data...")
    dataset = load_ie_dataset(DATA_PATH, MAX_SAMPLES)
    print(f"Loaded {len(dataset)} ie_extraction samples")

    training_args = GRPOConfig(
        output_dir="outputs/grpo_ie",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=1024,
        learning_rate=2e-5,
        logging_steps=5,
        save_steps=200,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Training GRPO...")
    trainer.train()

    try:
        trainer.save_model("outputs/grpo_ie_final")
    except Exception:
        print("Disk full — saving LoRA adapter only")
        model.save_pretrained("/tmp/grpo_ie_final")
        tokenizer.save_pretrained("/tmp/grpo_ie_final")

    print("Done.")


if __name__ == "__main__":
    main()
