"""Opt-in wall-clock and call-level profiling for one synchronous interaction."""
import cProfile
import json
import pstats
import time


def profile_interaction(run, directory, **kwargs):
    directory.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    started = time.perf_counter()

    def snapshot():
        profiler.dump_stats(str(directory / "calls.pstats"))
        stats = pstats.Stats(profiler)
        rows = []
        for (file, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
            rows.append(dict(file=file, line=line, function=function, calls=calls,
                             own_seconds=own, cumulative_seconds=cumulative))
        rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
        (directory / "calls.json").write_text(json.dumps(rows, indent=2))

    with (directory / "steps.jsonl").open("a", buffering=1) as stream:
        def record(index, phase, operation, seconds):
            stream.write(json.dumps(dict(index=index, phase=phase, operation=operation,
                seconds=seconds, elapsed_seconds=time.perf_counter() - started,
                timestamp=time.time())) + "\n")
            if operation == "after_step" and index % 10 == 0:
                profiler.disable()
                snapshot()
                profiler.enable()
        profiler.enable()
        try:
            return run(**kwargs, timing_sink=record)
        finally:
            profiler.disable()
            snapshot()
