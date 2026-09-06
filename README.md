# LocalLLM Reasoner

基于 llama.cpp C API 在本地运行 Qwen2.5-3B-Instruct GGUF 量化模型，对输入的测试词汇列表做批量推理（判断每个词汇是否符合中文表达逻辑），输出含准确率与推理性能指标的结构化 JSON 结果。

**背景**：WPS C++ 培训课程第 1 周作业，目标是练习使用开源 C 库做工程集成与性能测量。

## 功能

1. 加载 GGUF 模型（Qwen2.5-3B-Instruct-q4_k_m 量化，约 2GB）
2. 按行读取测试词汇文件，逐条调用模型推理
3. 将每次返回值（含是/否判断）及汇总指标写入 JSON 结果文件

## 性能指标

| 指标 | 说明 |
|---|---|
| TTFT | 首 Token 响应时间（Time To First Token），毫秒 |
| TPOT | 解码吞吐（Tokens per sec over Time） |
| E2E | 端到端推理时长，毫秒 |

## 项目结构

| 路径 | 说明 |
|---|---|
| main.cpp | 程序入口：读取参数、加载模型、推理并输出结果 |
| CMakeLists.txt | CMake 构建配置 |
| script/build.bat | Windows 一键构建脚本 |

## 依赖

- llama.cpp 源码及构建产物（include/ 与 lib/）
- [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)（本工程使用 q4_k_m 版本）

## 构建与运行

    # 1. 安装依赖：将 llama.cpp 编译产物放置到 CMake 可检索的位置（详见 script/build.bat）
    # 2. 双击运行
    script/build.bat

    # 3. 使用
    localllm.exe --input words.txt --output result.json

## 输出示例（文件结构仅供参考）

    {
        "accuracy": { "success": 0, "fail": 0, "total": 0 },
        "costtime": 0,
        "performance": {
            "avg_ttft_ms": 0,
            "avg_tpot_tokens_per_sec": 0,
            "avg_e2e_ms": 0
        }
    }

## 后续计划

- 支持 `n_ctx`、`n_predict` 等推理参数通过命令行传入
- 支持一次批量测测多份词汇文件
- 对不同量化档位的模型做推理性能对比
