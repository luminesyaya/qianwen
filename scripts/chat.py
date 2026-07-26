"""交互式结构化抽取测试"""
import torch, json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = "你是一个严格遵循 schema 的信息抽取助手。从文本中抽取所有实体和关系三元组，只输出 JSON，不附加解释。"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/qwen_lora/final")
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16
    ).to(device)
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.eval()

    print(f"加载模型: {args.ckpt}")
    print("输入文本，输出结构化 JSON。输入 /quit 退出。\n")

    while True:
        text = input("文本: ").strip()
        if text.lower() == "/quit":
            break
        if not text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"从以下文本中抽取实体和关系三元组，输出 JSON：\n\n{text}"},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer.encode(rendered)
        inputs = torch.tensor([inputs], dtype=torch.long).to(device)

        out = model.generate(
            inputs, max_new_tokens=512, temperature=args.temperature,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
        reply = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"\n{reply}\n")


if __name__ == "__main__":
    main()
