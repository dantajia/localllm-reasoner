#include "llama.h"
#include "ggml.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

static const std::string MODEL_PATH = "E:\\models\\qwen2.5-3b-instruct-q4_k_m.gguf";
static const int N_THREADS = 6;
static const int N_CTX = 128;
static const int N_PREDICT = 2;

static FILE* g_log = nullptr;
static void logMsg(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (g_log) { vfprintf(g_log, fmt, args); fflush(g_log); }
    va_end(args);
}

struct TestResult {
    int success;
    int fail;
    int total;
};

struct PerfMetrics {
    double ttft_ms;
    double tpot_tokens_per_sec;
    double e2e_ms;
};

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> readLines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", path.c_str());
        return lines;
    }
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

static bool writeJsonFile(const std::string& path, const std::string& json) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot write file %s\n", path.c_str());
        return false;
    }
    f << json;
    return true;
}

struct LLMSession {
    llama_model* model;
    llama_context* ctx;
    llama_sampler* smpl;
    const llama_vocab* vocab;
    bool ready;

    LLMSession() : model(nullptr), ctx(nullptr), smpl(nullptr), vocab(nullptr), ready(false) {}

    ~LLMSession() {
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool init(const std::string& modelPath) {
        llama_log_set([](enum ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);

        ggml_backend_load_all();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        model = llama_model_load_from_file(modelPath.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "Error: failed to load model: %s\n", modelPath.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = N_CTX;
        cparams.n_batch = N_CTX;
        cparams.n_threads = N_THREADS;
        cparams.n_threads_batch = N_THREADS;

        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "Error: failed to create context\n");
            return false;
        }

        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        ready = true;
        return true;
    }

    PerfMetrics chat(const std::string& userMsg, std::string& response) {
        PerfMetrics perf = {0, 0, 0};
        response.clear();

        if (!ready) return perf;

        auto t_start = std::chrono::high_resolution_clock::now();

        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        std::vector<llama_chat_message> messages;
        messages.push_back({"user", userMsg.c_str()});

        const char* tmpl = llama_model_chat_template(model, nullptr);

        std::vector<char> formatted(N_CTX);
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "Error: failed to apply chat template\n");
            return perf;
        }

        std::string prompt(formatted.begin(), formatted.begin() + new_len);

        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt);
        llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true);

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        auto t_first_token = t_start;
        bool first_token = true;
        int n_decoded = 0;

        for (int n_pos = 0; n_pos < n_prompt + N_PREDICT; ) {
            if (n_pos + batch.n_tokens > N_CTX) break;

            if (llama_decode(ctx, batch)) break;
            n_pos += batch.n_tokens;

            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token)) break;

            if (first_token) {
                t_first_token = std::chrono::high_resolution_clock::now();
                perf.ttft_ms = std::chrono::duration<double, std::milli>(t_first_token - t_start).count();
                first_token = false;
            }

            char buf[64];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n < 0) break;

            response += std::string(buf, n);
            n_decoded++;
            batch = llama_batch_get_one(&new_token, 1);
        }

        llama_sampler_reset(smpl);

        auto t_end = std::chrono::high_resolution_clock::now();
        perf.e2e_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (n_decoded > 0 && perf.e2e_ms > perf.ttft_ms) {
            perf.tpot_tokens_per_sec = n_decoded / ((perf.e2e_ms - perf.ttft_ms) / 1000.0);
        }

        return perf;
    }
};

static bool checkYesNo(const std::string& response) {
    std::string lower = response;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("yes") != std::string::npos ||
        lower.find("是") != std::string::npos ||
        lower.find("正确") != std::string::npos ||
        lower.find("符合") != std::string::npos ||
        lower.find("合理") != std::string::npos ||
        lower.find("true") != std::string::npos ||
        lower.find("对") != std::string::npos) {
        return true;
    }
    return false;
}

static std::string buildPrompt(const std::string& word) {
    return "判断以下词汇是否符合中文表达逻辑，只回答是或否：\n" + word;
}

int main(int argc, char* argv[]) {
    std::setlocale(LC_ALL, "zh_CN.UTF-8");
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    g_log = fopen("llm_debug.log", "w");
    logMsg("=== localllm started ===\n");

    std::string inputFile;
    std::string outputFile;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            inputFile = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputFile = argv[++i];
        }
    }
    logMsg("input: %s\noutput: %s\n", inputFile.c_str(), outputFile.c_str());

    if (inputFile.empty() || outputFile.empty()) {
        printf("Usage: localllm.exe --input <test_file> --output <result_file>\n");
        if (g_log) fclose(g_log);
        return 1;
    }

    auto lines = readLines(inputFile);
    if (lines.empty()) {
        fprintf(stderr, "Error: no input data\n");
        if (g_log) fclose(g_log);
        return 1;
    }
    logMsg("Read %zu lines\n", lines.size());

    fprintf(stderr, "Loading model: %s\n", MODEL_PATH.c_str());
    logMsg("Loading model: %s\n", MODEL_PATH.c_str());
    LLMSession session;
    if (!session.init(MODEL_PATH)) {
        fprintf(stderr, "Error: model initialization failed\n");
        logMsg("Model init failed!\n");
        if (g_log) fclose(g_log);
        return 1;
    }
    fprintf(stderr, "Model loaded. Threads=%d Context=%d Predict=%d\n", N_THREADS, N_CTX, N_PREDICT);
    logMsg("Model loaded. Threads=%d Context=%d\n", N_THREADS, N_CTX);

    TestResult result = {0, 0, (int)lines.size()};
    double total_ttft = 0, total_tpot = 0, total_e2e = 0;

    auto t_global_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < lines.size(); ++i) {
        std::string prompt = buildPrompt(lines[i]);
        std::string response;
        PerfMetrics perf = session.chat(prompt, response);

        bool isYes = checkYesNo(response);
        if (isYes) result.success++;
        else result.fail++;

        total_ttft += perf.ttft_ms;
        total_tpot += perf.tpot_tokens_per_sec;
        total_e2e += perf.e2e_ms;

        fprintf(stderr, "[%zu/%zu] %s -> %s -> %s (TTFT:%.0fms TPOT:%.1ft/s E2E:%.0fms)\n",
               i + 1, lines.size(),
               lines[i].c_str(), response.c_str(), isYes ? "是" : "否",
               perf.ttft_ms, perf.tpot_tokens_per_sec, perf.e2e_ms);

        logMsg("[%zu/%zu] %s -> %s (TTFT:%.0f TPOT:%.1f E2E:%.0f)\n",
               i+1, lines.size(), lines[i].c_str(), response.c_str(),
               perf.ttft_ms, perf.tpot_tokens_per_sec, perf.e2e_ms);
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    int64_t totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global_start).count();

    double avg_ttft = result.total > 0 ? total_ttft / result.total : 0;
    double avg_tpot = result.total > 0 ? total_tpot / result.total : 0;
    double avg_e2e = result.total > 0 ? total_e2e / result.total : 0;

    fprintf(stderr, "\n========== Results ==========\n");
    fprintf(stderr, "Total: %d  Success: %d  Fail: %d\n", result.total, result.success, result.fail);
    fprintf(stderr, "Time: %lld ms\n", totalMs);
    fprintf(stderr, "Avg TTFT: %.0f ms (target: <=500ms)\n", avg_ttft);
    fprintf(stderr, "Avg TPOT: %.1f tokens/s (target: >=30)\n", avg_tpot);
    fprintf(stderr, "Avg E2E:  %.0f ms (target: <=1000ms)\n", avg_e2e);

    logMsg("Results: total=%d success=%d fail=%d time=%lld TTFT=%.0f TPOT=%.1f E2E=%.0f\n",
           result.total, result.success, result.fail, totalMs, avg_ttft, avg_tpot, avg_e2e);

    char json[8192];
    snprintf(json, sizeof(json),
        "{\n"
        "    \"accuracy\":\n"
        "    {\n"
        "        \"success\": %d,\n"
        "        \"fail\": %d,\n"
        "        \"total\": %d\n"
        "    },\n"
        "    \"costtime\": %lld,\n"
        "    \"performance\":\n"
        "    {\n"
        "        \"avg_ttft_ms\": %.0f,\n"
        "        \"avg_tpot_tokens_per_sec\": %.1f,\n"
        "        \"avg_e2e_ms\": %.0f\n"
        "    }\n"
        "}\n",
        result.success, result.fail, result.total, totalMs,
        avg_ttft, avg_tpot, avg_e2e);

    if (!writeJsonFile(outputFile, json)) {
        if (g_log) fclose(g_log);
        return 1;
    }

    fprintf(stderr, "Result written to: %s\n", outputFile.c_str());
    logMsg("Done!\n");
    if (g_log) fclose(g_log);
    return 0;
}
