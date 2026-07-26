"""Merge LoRA → HF 格式，vLLM 可直接加载"""
import torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def export(base_model, lora_path, output_dir):
    print(f"加载基座: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

    print(f"加载 LoRA: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merge & unload...")
    model = model.merge_and_unload()

    print(f"保存到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    print("完成。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lora", default="outputs/qwen_lora/final")
    p.add_argument("--output", default="outputs/qwen_merged")
    args = p.parse_args()
    export(args.base, args.lora, args.output)
