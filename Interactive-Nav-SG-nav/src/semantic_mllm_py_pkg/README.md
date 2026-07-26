# semantic_mllm_py_pkg

Shared model infrastructure for modular interactive navigation.

The package intentionally separates model transport from the three navigation roles:

- `attribute_inference`: visual interaction attributes for graph enrichment
- `subgoal_selection`: selection among deterministic valid candidate IDs
- `skill_planning` and `visual_verification`: semantic interaction skills and feedback

Supported transports are `disabled`, `mock`, `command`, generic HTTP, OpenAI Chat
Completions, and OpenAI Responses. Every response records latency, token usage, and
completion TPS when the provider returns usage metadata.

Independent ablation modes:

```text
module1: static_semantic | dynamic_rule | dynamic_mllm
module2: rule_cost | mllm_score
module3: direct_atomic | rule_verified | mllm_skill_verified
```

Models never directly create navigation coordinates, joint axes, force trajectories,
or graph mutations. Their JSON output is validated before the mapping, decision, or
execution module applies it.
