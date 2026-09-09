# InteractiveNav V3 ProcTHOR validation benchmark

This directory carries the three scoring domains from frozen release
`interactive-nav-v3-procthor10k-val-release-v1.1`:

- `channel.json.gz`: 1,000 episodes
- `container.json.gz`: 976 episodes
- `mixed.json.gz`: 992 episodes

The archives were created with `gzip -n -9`. Decompressing each file produces
the exact frozen JSON bytes identified by the uncompressed SHA-256 values in
`manifest.json`. The redundant 135 MB aggregate `benchmark.json` is omitted;
its 2,968 episodes are exactly the three domain lists concatenated.

`pinned_assets.json` records the robot, ProcTHOR validation scene, and THOR
object versions used by the release. The wrapper applies it automatically and
records its content hash in every run manifest/signature. Override it explicitly
with `--pinned-assets-file` only when auditing another compatible asset build.

Both the V3 evaluator and the mixed-domain wrapper accept `.json` and
`.json.gz` transparently. The wrapper uses this directory by default:

```bash
python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --output-dir /home/ldl/outputs/interactive-nav/external_policy_eval \
  --policy factory \
  --policy-factory your_package.your_policy:build_policy \
  --workers 3
```

Use `--episodes-per-domain 1 --no-render-topdown` for a small integration run.
Outputs include `run_manifest.json`, `progress.json`, `progress.log`,
`results.json`, `summary.json`, `summary.csv`, and per-episode traces.
