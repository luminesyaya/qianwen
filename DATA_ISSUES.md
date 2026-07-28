# InstructIE 数据 Profiling 发现的 5 个问题

## 问题 1：字段不统一

**怎么发现的**：读 train 和 valid 的第一条样本，对比 key。

```python
train.keys()  → ['id','cate','text','relation','entity']
valid.keys()  → ['id','cate','input','relation','entity']
```

train 用 `text`，valid 用 `input`。且 train 的 relation 含 `head_type`/`tail_type`，valid 不含。

**处理**：Step 1 标准化——统一为 `input`，补齐缺失的 `head_type`/`tail_type`。

---

## 问题 2：跨集泄漏

**怎么发现的**：把 train 所有 `text` 存成集合，遍历 valid 检查是否重复。

```python
train_texts = set(d['text'] for d in train)
leak = sum(1 for d in valid if d['input'] in train_texts)
# → 143 条
```

**处理**：只用 `train_zh.json`，Pipeline 结束后自己切 95:5。待改进见 `IMPROVEMENTS.md` Step 0。

---

## 问题 3：cate 分布失衡

**怎么发现的**：Counter 统计所有样本的 `cate` 字段。

```
地理地区: 20000 (11.7%)    医学: 3244 (1.9%)
运输:     20000 (11.7%)    自然科学: 4308 (2.5%)
人物:     20000 (11.7%)
...
```

前 7 个 topic 占 85%，医学只有 1.9%。

**处理**：Step 5 分层采样——按 (task_type, cate) 分组，每组按比例配额，保证每个 topic 在最终 30K 中都有代表。

---

## 问题 4：relation 频率倾斜

**怎么发现的**：Counter 统计所有样本的 `relation[].relation` 字段。

```
"位于": 193,247 (18.4%)
"别名": 111,423 (10.6%)
--- 两者占总体的 29% ---
"面积":  45,801
"创立时间":  8,234
... 低频关系断崖式下降
```

**处理**：已知偏差，未做降采样。改进方向见 `IMPROVEMENTS.md` 第 2 条。

---

## 问题 5：文本长度 split 不一致

**怎么发现的**：分别计算三个 split 的文本长度分位数。

```
train: 中位数 129 字符, P95 = 468
valid: 中位数 86 字符
test:  中位数 76 字符
```

train 明显长于 valid 和 test。

**处理**：Step 2 过滤 <15 和 >800 的极端长度 + 软过滤 P99 + 自己切分保证 train/valid 同分布。

---

### 总结

| # | 问题 | 工具 | 处理 |
|------|------|------|------|
| 1 | 字段不统一 | 读第一条对比 key | Step 1 |
| 2 | 跨集泄漏 | 集合求交 | 自己切分 |
| 3 | cate 失衡 | Counter | Step 5 |
| 4 | relation 倾斜 | Counter | 已知偏差 |
| 5 | 长度不一致 | 分位数 | Step 2 + 自己切分 |

核心原则：拿到新数据先 profiling，发现的问题每个对应 Pipeline 一步处理。
