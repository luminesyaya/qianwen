# GRPO 结构化抽取对齐实验记录

## 实验配置

| 参数 | 值 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct LoRA SFT (Parse% 95.2%) |
| 训练框架 | TRL GRPOTrainer v1.9.2 |
| 数据 | InstructIE ie_extraction, 50 条 |
| num_generations | 4 (group_size) |
| batch_size | 2 |
| max_completion_length | 1536 |
| learning_rate | 4e-5 |
| reward | Parse(0.2) + Alias(0.1) + Strict(0.5, 连续分) |

## Reward v1 → v2 对比

| | v1 (二进制) | v2 (连续分) |
|------|:--:|:--:|
| frac_reward_zero_std | ~60% | **~10%** |
| 组内区分度 | 极低 | 正常 |
| Strict 打分 | 全有全无 (0.3) | 距离 gold 越近分越高 (0~0.5) |

## 训练趋势 (50 steps, 731s)

| 指标 | 值 |
|------|------|
| reward 范围 | 0.68~0.80 |
| loss 均值 | ~0.02 |
| OOM | 个别 step 爆显存（1536×4=6144 tokens），不影响训练 |
| clipped_ratio | 0 |

## 经验总结

1. SFT 后的连续 reward 设计关键——Parse/Alias 天花板后只有 Strict 能提供区分度
2. GRPO 在 1.5B 模型上 50 步即可验证流程，完整训练需要更多数据（500+）
3. 显存优化：max_completion_length 和 group_size 互斥，二选一增大
