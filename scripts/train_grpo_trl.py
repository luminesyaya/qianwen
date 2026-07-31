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
    """Parse + Alias + Strict → 0~1 连续分"""
    pred = extract_json(response)
    gold = json.loads(gold_str) if isinstance(gold_str, str) else gold_str
    if pred is None:
        return 0.0
    score = 0.3
    has_ent = "entities" in pred and isinstance(pred["entities"], list)
    has_rel = "relations" in pred and isinstance(pred["relations"], list)
    if has_ent and has_rel:
        score += 0.2
        if len(pred["entities"]) == len(gold.get("entities", [])) and \
           len(pred["relations"]) == len(gold.get("relations", [])):
            score += 0.3
    return score


def extract_gold(text):
    parts = text.split("<|im_start|>assistant\n")
    return json.loads(parts[-1].replace("<|im_end|>", "").strip())


# ═══════════════════
# ★ TRL reward 函数
# ═══════════════════
def reward_fn(prompts, completions, **kwargs):
    """
    TRL GRPO 调用:
      prompts:     list[str] batch 个 prompt
      completions: list[str] batch×num_generations 个模型回答
      返回:        list[float] 每个回答的分数
    """
    rewards = []
    for i, completion in enumerate(completions):
        # completion 可能是 list[dict]（chat format）或 str
        if isinstance(completion, list):
            text = "".join(
                c.get("content", "") if isinstance(c, dict) else str(c)
                for c in completion
            )
        else:
            text = str(completion)

        # 对应的 prompt 索引: TRL 按顺序排列，prompt_i 对应
        # completions[i*num_generations : (i+1)*num_generations]
        prompt_idx = i // kwargs.get("num_generations", 1)
        prompt = prompts[prompt_idx] if prompt_idx < len(prompts) else ""

        # 从 prompt 末尾提取 gold JSON 字符串
        gold_str = ""
        for line in prompt.rsplit("\n", 10):
            if line.strip().startswith("{"):
                gold_str = line.strip()
                break

        rewards.append(grade_answer(text, gold_str) if gold_str else 0.0)

    return rewards


# ═══════════════════
# 数据加载
# ═══════════════════
def load_ie_dataset(path, max_samples=2000):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s.get("task") == "ie_extraction":
                # 只取 prompt 部分（assistant 标记为止）
                prompt = s["text"].split("<|im_start|>assistant\n")[0]
                prompt += "<|im_start|>assistant\n"
                gold = extract_gold(s["text"])
                samples.append({"prompt": prompt, "gold": gold})
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
    MAX_SAMPLES = 50

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
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,       # group_size
        max_prompt_length=1024,
        max_completion_length=1024,
        temperature=1.0,
        learning_rate=4e-5,
        logging_steps=1,
        save_steps=50,
        bf16=True,
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
    trainer.save_model("outputs/grpo_ie_final")
    print("Done.")


if __name__ == "__main__":
    main()
