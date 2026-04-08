# V2 Heading Prefix 配置快照

> 记录本轮实验时的 RAG 管道参数

## 与 V1 的唯一差异

```diff
# chunker.py _merge_paragraphs 输出块时:
- content = "\n\n".join(x["content"] for x in cur)
+ body = "\n\n".join(x["content"] for x in cur)
  heading = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
+ content = f"[{heading}]\n{body}" if heading else body
```

chunk content 示例:
```
[第2章 Transformer > 2.1.6 多头注意力机制]
自注意力机制在一次计算中只能拟合一种相关关系...
```

## 不变参数 (同 V1)
- 分块策略: heading_aware, chunk_size=800, overlap=100
- 嵌入模型: text-embedding-v4 (1024维)
- 向量存储: Qdrant Cloud, Cosine
- 重排序: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: DeepSeek Chat API
- 测试集: docx/test.md (25题)
