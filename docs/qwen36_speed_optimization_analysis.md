# Qwen3.6-35B-A3B-FP8 服务速度优化分析

更新时间：2026-09-17

## 当前运行状态

本机当前有两路独立 vLLM 服务：

| 项目 | GPU 0 / 8000 | GPU 1 / 8001 |
| --- | --- | --- |
| 模型 | `/home/ldl/qwen36-fp8/model/Qwen3.6-35B-A3B-FP8` | 相同 |
| vLLM | 0.19.0 | 0.19.0 |
| Tensor Parallel | 1 | 1 |
| `max_model_len` | 10240 | 10240 |
| `max_num_seqs` | 24 | 24 |
| `gpu_memory_utilization` | 0.5 | 0.5 |
| Prefix caching | 关闭 | 关闭 |
| 显存占用 | 约 40.6 GiB | 约 40.6 GiB |
| MTP | 未启用 | 未启用 |

当前进程的实际命令行和 vLLM metrics 都显示 `speculative_config=None`，所以模型虽然在 `config.json` 的 `text_config` 中声明了 `mtp_num_hidden_layers=1`，并不代表当前服务已经启用 MTP。

另外，`8010` 是当前的负载均衡入口，后端是 `8000,8001`。如果客户端直接访问 8000 或 8001，会绕过这个均衡器。

## MTP 可行性

可以启用。当前权重的模型类型是 `qwen3_5_moe`，配置包含 `mtp_num_hidden_layers=1`；本机 vLLM 0.19.0 的 speculative 配置代码包含 Qwen3.5 MTP 适配，并且已有本地实验成功加载 drafter：

```text
--speculative-config '{"model":"/home/ldl/qwen36-fp8/model/Qwen3.6-35B-A3B-FP8","method":"mtp","num_speculative_tokens":1}'
```

成功运行时日志出现了 `Detected MTP model`、`Sharing target model embedding weights`，并输出了 `SpecDecoding metrics`。历史测试中平均 draft acceptance rate 大约为 82%–98%，说明这份权重的 MTP 头不是空配置。

推荐的第一组参数是：

```bash
--speculative-config '{"model":"/home/ldl/qwen36-fp8/model/Qwen3.6-35B-A3B-FP8","method":"mtp","num_speculative_tokens":1}'
--no-enable-prefix-caching
```

不要第一步就把 `num_speculative_tokens` 调到 2 或 3。当前模型只声明了 1 个 MTP hidden layer；vLLM 的公开 issue 也显示，多 token MTP 在 Qwen3.5 系列上仍有实现/接受率讨论，应该先用 1 做基准。

## 已有本机证据

历史单卡实验在相近配置下已经验证过 MTP 能启动，日志位置为 `/home/ldl/qwen36-fp8/bench/mtp_retest_eager/mtp.log` 和 `/home/ldl/qwen36-fp8/bench/hist_short_210225/mtp_piecewise.log`。

其中 `mtp_piecewise.log` 记录了：

- MTP 成功加载；
- 单并发阶段接受率约 82%–98%；
- `Mean acceptance length` 约 1.8–2.0，意味着一次 target forward 平均能确认接近两个 token；
- 并发请求增多后，MTP 仍能工作，但收益取决于 batch、输出长度和 acceptance rate。

`mtp_retest_eager/mtp_bench.json` 与 `baseline_eager_bench.json` 不是严格的同条件 A/B：测试时端口、显存预算、编译/图模式和测试 suite 不完全一致。因此它们只能证明 MTP 有可用性和潜在收益，不能直接作为最终性能结论。

## 优化建议

### 1. 先做 MTP 单卡 A/B

分别启动一个普通服务和一个 MTP 服务，保持模型、GPU、`max_model_len`、显存比例、编译模式、请求集合完全一致。至少记录：

- TTFT、TPOT、端到端延迟的 p50/p95；
- 单请求短输出和长输出；
- 并发 1、4、8、16 的 completion tokens/s；
- `SpecDecoding metrics` 中的 acceptance rate 和 mean acceptance length；
- GPU 利用率、显存、KV cache 使用率；
- 错误率和输出一致性。

MTP 的目标应主要看低并发 decode latency/TPOT；如果目标是高并发总吞吐，普通服务可能更好。官方 Qwen3.5/3.6 配方明确把 MTP 放在 latency-focused serving，并提示高负载下 throughput 可能下降。

### 2. 当前双卡部署不要把两张卡合并成 TP=2

当前两路 TP=1 服务能让两个请求流并行使用两张 A100。MTP 试验也应保持每路 TP=1。除了更容易隔离 A/B，vLLM 社区目前仍有 Qwen3.5 MTP 在 TP>=2 下 drafter shape mismatch 的公开 issue；本机没有必要为 MTP 引入这个风险。

### 3. Prefix caching 根据请求模式选择

当前服务关闭 prefix caching，适合动态 prompt、低重复前缀和 MTP latency 测试。若实际请求有大量相同 system prompt、相同图像/历史对话前缀，应单独测试开启 prefix caching；它主要降低重复 prefill，不会直接提高纯 decode 的 token/s。官方配置把高并发吞吐场景和 prefix caching 放在一起，而 latency-focused MTP 示例则关闭/不启用 prefix caching。

### 4. 先恢复合理的 KV cache 预算

当前实际服务是 `gpu_memory_utilization=0.5`，每卡约 40.6 GiB，KV blocks 为 126；这不是此前约定的 0.6。MTP 会额外消耗 drafter/图捕获相关显存，不能机械地把它改成 0.6。建议：

- 普通服务先测 `0.6`；
- MTP 服务从 `0.55` 起测；
- 观察启动时 available KV cache、CUDA graph memory 和是否 OOM，再决定是否上调；
- `max_num_seqs=24` 只在确有 24 路并发需求时保留，低并发 latency 服务可降到 4–8 以减少调度和图捕获成本。

### 5. A100 上不要期待 FP8 原生收益

本机日志明确提示 A100 没有 native FP8 computation，当前使用 Marlin 做 weight-only FP8，compute-heavy workload 可能受影响。换成更适合 A100 的权重格式/后端需要重新评估显存和质量，不能仅凭文件名 FP8 判断一定更快。MTP 是当前风险较低、最值得先验证的速度优化点。

## 推荐试验顺序

1. 保留当前两个普通服务作为 baseline，记录同一批请求的指标。
2. 只替换 GPU 1 的 8001 服务为 MTP-1，使用 `--no-enable-prefix-caching`、TP=1、`gpu_memory_utilization=0.55`。
3. 等待服务 ready 后做并发 1/4/8/16 的固定请求 A/B。
4. 如果低并发 TPOT 明显下降且错误率为 0，再考虑将 MTP 服务切到 8010 负载均衡入口。
5. 只有在 MTP-1 的 acceptance rate 和稳定性都满足要求后，才试 `num_speculative_tokens=2`。

## 参考资料

- [vLLM Qwen3.5/Qwen3.6 serving recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)：Qwen3.6 MTP 命令，以及 latency-focused MTP-1 与 throughput-focused 配置区分。
- [vLLM MTP documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md)：MTP 配置、`num_speculative_tokens` 和支持范围说明。
- [vLLM speculative configuration source](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)：Qwen3.5 模型类型到 `qwen3_5_mtp` 的适配逻辑。
- [Qwen3.5/3.6 MTP TP issue](https://github.com/vllm-project/vllm/issues/52480)：Qwen3.5 MTP 在 TP>=2 下的 drafter shape mismatch 报告。
- [Qwen3.5 MTP multi-token discussion](https://github.com/vllm-project/vllm/issues/52688)：`num_speculative_tokens>1` 与多 MTP 层的实现讨论。
