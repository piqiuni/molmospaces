# InteractiveNav V3 ProcTHOR validation benchmark (legacy v1.1)

This is the historical runtime-qualified bundle. The complete v1.2 candidate
release is now the repository default at
`../interactive_nav_v3_procthor10k_val_release_v1_2`; use this directory only
when reproducing an older v1.1 run.

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
`.json.gz` transparently. Pass this directory explicitly with
`--benchmark-root` when reproducing v1.1; the wrapper no longer selects it by
default:

```bash
python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --benchmark-root scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_1 \
  --pinned-assets-file scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_1/pinned_assets.json \
  --output-dir /home/ldl/outputs/interactive-nav/external_policy_eval \
  --policy factory \
  --policy-factory your_package.your_policy:build_policy \
  --workers 3
```

Use `--episodes-per-domain 1 --no-render-topdown` for a small integration run.
Outputs include `run_manifest.json`, `progress.json`, `progress.log`,
`results.json`, `summary.json`, `summary.csv`, and per-episode traces.
