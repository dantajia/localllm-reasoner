# LocalLLM Reasoner — 基于 llama.cpp 的本地大模型推理工具

一个使用 **llama.cpp C API** 在本地加载 **GGUF 量化模型（Qwen2.5-3B-Instruct, q4_k_m, 约 2 GB）** 进行批量推理的 C++ 工具：读取测试词汇文件，逐条调用模型判断"该词汇是否符合中文表达逻辑"，最终输出包含准确率统计与推理性能指标的结构化 JSON。

## 工程亮点

- **全程基于 llama.cpp C API 手写推理链路**，不依赖任何上层封装：
  `llama_model_load_from_file → llama_chat_apply_template → llama_tokenize → llama_batch_decode → llama_sampler_sample`，完整经历 prompt 模板化、tokenize、批处理 decode、采样器（greedy, temp=0）全流程
- **RAII 会话管理**：`LLMSession` 封装 model/context/sampler 的生命周期，析构时自动释放 `llama_sampler_free / llama_free / llama_model_free`，无泄漏
- **完整推理性能指标体系**（业界标准三指标，附目标值约束）：
  | 指标 | 含义 | 目标 |
  |---|---|---|
  | **TTFT** (Time To First Token) | 首 token 延迟 | ≤ 500 ms |
  | **TPOT** (Tokens Per sec Over Time) | 解码吞吐 | ≥ 30 tok/s |
  | **E2E** | 端到端推理时长 | ≤ 1000 ms |
- **结果归一化**：中英文混合回答做 keyword 归一（是/正确/符合/合理/true/yes → 判断为"符合"），保证批处理统计口径一致
- **结构化输出**：`accuracy{success,fail,total} + costtime + performance{avg_ttft_ms, avg_tpot_tokens_per_sec, avg_e2e_ms}` 的 JSON 结果文件

## 推理核心节选

```cpp
// 贪心采样链（确定性输出，保证批处理可复现）
smpl = llama_sampler_chain_init(...);
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

// 逐条推理并计时三段
perf.ttft_ms = t_first_token - t_start;                       // 首 token 延迟
perf.tpot_tokens_per_sec = n_decoded / ((e2e - ttft) / 1000);  // 稳态吞吐
perf.e2e_ms = t_end - t_start;                                 // 端到端
```

## 输出示例

```json
{
    "accuracy": { "success": 87, "fail": 13, "total": 100 },
    "costtime": 182430,
    "performance": {
        "avg_ttft_ms": 412,
        "avg_tpot_tokens_per_sec": 38.6,
        "avg_e2e_ms": 893
    }
}
```

## 构建与运行

```bash
# 依赖：llama.cpp 源码编译产物（include/ + lib/） + Qwen2.5-3B-Instruct-GGUF 模型
# 模型下载：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF （q4_k_m, ~2GB）

# Windows + VS2022
script\build.bat

# 运行
localllm.exe --input words.txt --output result.json
```

## 可拓展方向

- 首 token 预取缓存（系统 prompt 复用）以降低 TTFT
- llama.cpp 并行批处理（`n_parallel > 1`）提升吞吐
- 支持其他 GGUF 量化档位（q5_k_m / q8_0）下的准确率-性能折衷对比
