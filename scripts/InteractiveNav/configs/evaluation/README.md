# V3 benchmark configuration

`benchmark_eval.conf` is the single-episode launch profile. It includes dynamic
step budgets, recorder persistence, video, ROS synchronization, and algorithm
YAML paths. It uses Bash syntax; defaults preserve existing environment values.
An optional third launcher argument supplies a trusted override file sourced
after defaults. Use direct assignments in that override file.

```bash
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  /home/ldl/outputs/interactive-nav/my_eval 0
```

For example, a small override file can contain:

```bash
FAST_EVAL=true
ROS_MASTER_URI=http://127.0.0.1:15101
```

Pass its path as the third argument. Batch jobs can select it through
`EVAL_CONFIG`; batch controls episode indices, worker ports and the step cap.
Avoid overriding worker ports in a shared batch profile. Configuration hashes
participate in the batch resume identity.

Every run saves `config/effective_config.env`, default settings, the four
algorithm YAMLs, and generated ROS logging configuration. Credential file
contents are not copied. The evaluator also preserves its run manifest and the
per-episode effective budget. `effective_config.env` is an audit snapshot;
worker-specific values must be changed before reusing it elsewhere.

Default persistence:

| Setting | Default | Effect |
|---|---|---|
| `STEP_BUDGET_MODE` | `dynamic` | GT path plus interaction budget |
| `MAX_STEPS` | `2000` | Upper cap, also batch CLI default |
| `RECORDER_SAVE_EVENTS` | `false` | Do not create `debug/events.jsonl` |
| `RECORDER_COMPACT_STEPS` | `true` | Remove candidate `graph_context`; retain only terminal/completion summaries from decision trace |
| `RECORDER_COMPRESS_STEPS` | `true` | Flush gzip level 3 JSONL per step |
| `OFFLINE_SAVE_COMPOSITE_FRAMES` | `false` | Encode six-panel video without persisting derived PNGs |
| `ROS_LOG_LEVEL` | `WARN` | Suppress routine ROS INFO/DEBUG logs |
| `ROS_LOG_MAX_BYTES` | `5242880` | Python ROS log rotation threshold per file |
| `ROS_LOG_BACKUP_COUNT` | `2` | Keep two rotated Python ROS log files |

`step_boundaries.jsonl.gz` retains full unified graph, observations, poses,
plans, map receipts, candidate actions, selection/execution/feedback and
termination state for offline rendering. It is not a complete M2-input replay:
duplicated graph context and verbose decision diagnostics are intentionally
omitted. Drain, six-panel rendering and room replay accept old JSONL and new
gzip recordings. Original interactive-navigation recorder defaults are unchanged.

Python ROS logs retain roughly 15 MiB per process with default rotation,
subject to a final message overshooting the threshold. WARN filtering also
reduces C++ ROS and console logs; this is not a hard aggregate directory quota.
Shell stdout/stderr logs can still grow if a process prints repeatedly.

`FAST_EVAL=true` skips recorder, camera PNG persistence and offline video, while
retaining scoring traces, model metrics and logs. It does not change smooth
interaction or public perception. Recording remains 15 FPS with exact steps.

M2 settings live in `../semantic_decision/object_goal_v3_full_mllm.yaml`:
`model.selection_reasoning_effort=low` overrides the shared `.env` reasoning
default for M2 only; `model.max_tokens=1536` includes reasoning and final JSON,
and the request timeout stays 12 seconds. The prompt follows evidence,
dependencies, compatibility, progress and final validation. Only the final
ranking JSON is consumed. Backend reasoning support must be checked separately;
requesting it does not prove that the server produced reasoning tokens.
