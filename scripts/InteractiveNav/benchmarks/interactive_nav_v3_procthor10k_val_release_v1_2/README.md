# InteractiveNav V3 ProcTHOR validation benchmark v1.2

This directory contains the complete repaired candidate benchmark: 3,000
episodes, with 1,000 episodes in each scoring domain:

- `channel.json.gz`: 1,000 episodes
- `container.json.gz`: 1,000 episodes
- `mixed.json.gz`: 1,000 episodes

The archives were created with `gzip -n -9`. Their uncompressed bytes are the
exact source-release JSON bytes identified by `manifest.json`. The source
aggregate contains the same 3,000 episodes concatenated in domain order; it is
not duplicated in the repository. `benchmark.json` in the source release has
SHA-256 `4021ca2bebd9c875ccc4df70c746d9ed7f2376d13247fa1b7e98f2e9690b219f`.

`pinned_assets.json` records the robot, ProcTHOR validation scene, and THOR
object versions used by the release. The evaluator uses this directory by
default:

```bash
python scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py \
  --output-dir /home/ldl/outputs/interactive-nav/external_policy_eval \
  --policy factory \
  --policy-factory your_package.your_policy:build_policy \
  --workers 3
```

Use `--episodes-per-domain 1 --no-render-topdown` for a small integration run.
Verify archive and source hashes from this directory with:

```bash
sha256sum -c checksums.sha256
```
