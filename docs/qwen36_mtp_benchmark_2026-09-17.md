# Qwen3.6 双卡 MTP Benchmark

日期：2026-09-17

## 测试环境

- GPU：2 × NVIDIA A100-SXM4-80GB
- 模型：`Qwen3.6-35B-A3B-FP8`
- vLLM：0.19.0
- 部署：每张 GPU 一个 TP=1 实例，前置 least-inflight TCP load balancer
- `max_model_len=10240`
- `max_num_seqs=24`
- Prefix caching：关闭
- Baseline：`gpu_memory_utilization=0.5`，默认 `FULL_AND_PIECEWISE` CUDA graph
- MTP：`gpu_memory_utilization=0.55`，`PIECEWISE` CUDA graph
- Benchmark：预热后运行；强制生成固定 `max_tokens`，避免 EOS 和输出长度差异影响吞吐比较

## 强化测试结果

| 场景 | Baseline tok/s | MTP-2 tok/s | MTP-2 | MTP-3 tok/s | MTP-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 并发 1，短 prompt | 144.8 | 125.0 | -13.6% | 140.2 | -3.1% |
| 并发 1，长 prompt | 153.6 | 133.5 | -13.1% | 150.1 | -2.3% |
| 并发 4，mixed | 515.6 | 404.7 | -21.5% | 492.6 | -4.5% |
| 并发 8，mixed | 856.5 | 787.1 | -8.1% | 948.3 | +10.7% |
| 并发 16，mixed | 1314.8 | 1409.7 | +7.2% | 1600.6 | +21.7% |
| 并发 24，mixed | 1647.3 | 1837.1 | +11.5% | 1991.7 | +20.9% |
| 并发 8，长上下文 | 675.1 | 662.5 | -1.9% | 737.4 | +9.2% |
| 并发 8，短共享前缀 | 799.3 | 740.9 | -7.3% | 870.6 | +8.9% |

所有强化测试请求均成功，无推理错误。

## 延迟结果

| 场景 | Baseline TTFT | MTP-3 TTFT | Baseline 平均延迟 | MTP-3 平均延迟 |
| --- | ---: | ---: | ---: | ---: |
| 并发 1，短 prompt | 0.109s | 0.122s | 0.884s | 0.913s |
| 并发 1，长 prompt | 0.110s | 0.126s | 1.667s | 1.705s |
| 并发 4，mixed | 0.191s | 0.244s | 1.488s | 1.535s |
| 并发 8，mixed | 0.288s | 0.225s | 1.779s | 1.563s |
| 并发 16，mixed | 0.379s | 0.295s | 2.260s | 1.875s |
| 并发 24，mixed | 0.413s | 0.363s | 2.763s | 2.249s |
| 并发 8，长上下文 | 0.406s | 0.358s | 1.507s | 1.345s |
| 并发 8，短共享前缀 | 0.293s | 0.216s | 1.278s | 1.144s |

## MTP 接受率

- MTP-1 首轮测试：平均 draft acceptance 约 92%，mean acceptance length 约 1.92–1.93。
- MTP-2 强化测试末段：平均 draft acceptance 约 75.5%–77.5%，分位置约 82%–84%、69%–71%。
- MTP-3 强化测试末段：平均 draft acceptance 约 66.4%–67.4%，分位置约 83%、67%–69%、49%–50%；mean acceptance length 约 3.0。

MTP-3 的第三个 draft token 接受率虽然只有约 50%，但在并发 8 以上仍显著提高整体吞吐，说明这台 A100 上减少 target-model decode 次数带来的收益超过额外 draft 开销。

## 显存与 KV cache

| 配置 | 每卡显存 | GPU memory utilization | GPU KV blocks |
| --- | ---: | ---: | ---: |
| Baseline | 40.6 GiB | 0.50 | 126 |
| MTP-2 | 45.4 GiB | 0.55 | 259 |
| MTP-3 | 45.9 GiB | 0.55 | 259 |

MTP-3 比 baseline 每卡多占约 5.3 GiB，但仍保留约 35 GiB 物理显存余量。

## 结论

- 如果实际并发通常为 1–4，保留 baseline 更合适；MTP-3 会带来约 2%–5% 的吞吐和延迟损失。
- 如果实际并发通常达到 8–24，MTP-3 是当前最优配置，吞吐提高约 9%–22%，并同步降低 TTFT 和端到端延迟。
- MTP-2 不推荐：低并发损失明显，高并发收益也低于 MTP-3。
- 对“只共享很短 system prompt、后续内容不同”的请求，MTP-3 在并发 8 下提升约 8.9%；这个收益来自 speculative decoding，不依赖 prefix cache。
- Prefix caching 的独立测试本轮没有完成。首次测试暴露出启动环境缺少 Python include path，导致 Mamba block-copy Triton kernel 编译失败；该路径已经补齐，但按用户优先级先完成了 MTP 测试。

## 推荐部署参数

```text
--tensor-parallel-size 1
--max-model-len 10240
--max-num-seqs 24
--gpu-memory-utilization 0.55
--no-enable-prefix-caching
--compilation-config '{"cudagraph_mode":"PIECEWISE"}'
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

模型路径也可以显式放入 `speculative-config.model`，本次测试使用的是当前 target model 路径。

## 产物

- 全部 JSON、metrics 和服务日志：`/home/ldl/outputs/qwen36_dual_mtp_20260917_011533`
- Benchmark 客户端：`/home/ldl/qwen36-fp8/bench/bench_mtp_core.py`
- 双卡测试脚本：`/home/ldl/qwen36-fp8/bench/run_dual_mtp_robust.sh`
