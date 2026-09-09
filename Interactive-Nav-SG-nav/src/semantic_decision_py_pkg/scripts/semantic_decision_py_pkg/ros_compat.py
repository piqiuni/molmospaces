from __future__ import annotations


def patch_roslogging_findcaller_for_py311() -> None:
    import sys

    if sys.version_info < (3, 11):
        return
    try:
        import logging
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_semantic_decision_safe", False):
        return

    def safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    safe_find_caller._semantic_decision_safe = True
    roslogging.RospyLogger.findCaller = safe_find_caller
