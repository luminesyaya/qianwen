"""
Qwen2.5-1.5B-Instruct LoRA 微调 — 结构化信息抽取
"""
import json, os, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ===== 硬参数 =====
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH = "data/clean/train.jsonl"
VALID_PATH = "data/clean/valid.jsonl"
OUTPUT_DIR = "outputs/qwen_lora"
MAX_LENGTH = 1024

# ===== 超参数 =====
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
BATCH_SIZE, GRAD_ACCUM = 4, 4
MAX_STEPS, WARMUP_STEPS = 2000, 100
LR, WEIGHT_DECAY, GRAD_CLIP = 2e-4, 0.01, 1.0
LOG_EVERY, EVAL_EVERY, SAVE_EVERY = 50, 200, 500


class ChatMLDataset(Dataset):
    """
    加载 ChatML JSONL,返回 input_ids + assistant-only labels.

    Pre-fill(prefix 对比法）:
      用 template 渲染到 assistant 标记 → tokenize → 该长度的标签为 -100
      渲染完整回答 → 只在超出的部分填真实 token ID
    """
    def __init__(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]

        # tokenize 完整文本
        full = self.tokenizer(text, truncation=True, max_length=self.max_length,
                              return_tensors="pt", return_attention_mask=False)
        input_ids = full["input_ids"][0]  # (seq_len,)

        # 找到 "<|im_start|>assistant\n" 的 token 位置
        assistant_marker = "<|im_start|>assistant\n"
        prefix_ids = self.tokenizer(assistant_marker, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        # 在 input_ids 中搜索 assistant 标记
        label_mask = torch.full_like(input_ids, -100, dtype=torch.long)
        for i in range(len(input_ids) - prefix_len + 1):
            if input_ids[i:i + prefix_len].tolist() == prefix_ids:
                # assistant 内容从标记之后开始
                start = i + prefix_len
                label_mask[start:] = input_ids[start:]
                break

        return input_ids, label_mask


def collate_fn(batch):
    """左侧 padding"""
    max_len = max(len(x[0]) for x in batch)
    pad_token_id = 0  # Qwen pad token

    padded_ids, padded_labels = [], []
    for ids, labs in batch:
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([torch.full((pad_len,), pad_token_id), ids]))
        padded_labels.append(torch.cat([torch.full((pad_len,), -100), labs]))

    return torch.stack(padded_ids), torch.stack(padded_labels)


def main(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Tokenizer + 模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.gradient_checkpointing_enable()

    # 2. LoRA
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # 3. 数据
    train_ds = ChatMLDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
    valid_ds = ChatMLDataset(VALID_PATH, tokenizer, MAX_LENGTH)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, collate_fn=collate_fn)

    # 4. 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = 50 if smoke else MAX_STEPS
    warmup = min(5, steps // 2) if smoke else WARMUP_STEPS
    sched = get_cosine_schedule_with_warmup(opt, warmup, steps)

    # 5. 训练循环
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train()
    global_step = 0
    total_loss = 0.0

    train_iter = iter(train_dl)
    while global_step < steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        out = model(x, labels=y)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item()

        if (global_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad()

        if global_step % LOG_EVERY == 0:
            print(f"Step {global_step:5d}/{steps} | loss {total_loss / max(1, LOG_EVERY):.4f} | lr {sched.get_last_lr()[0]:.2e}")
            total_loss = 0.0

        if global_step % EVAL_EVERY == 0 and global_step > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vx, vy in valid_dl:
                    vx, vy = vx.to(device), vy.to(device)
                    val_loss += model(vx, labels=vy).loss.item()
            print(f"  [Eval] step {global_step} | val_loss {val_loss / len(valid_dl):.4f}")
            model.train()

        if global_step % SAVE_EVERY == 0 and global_step > 0:
            model.save_pretrained(f"{OUTPUT_DIR}/ckpt_{global_step}")

        global_step += 1

    # 6. 保存
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"\n保存到 {OUTPUT_DIR}/final")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Smoke test: 50 steps")
    args = p.parse_args()
    main(smoke=args.smoke)
