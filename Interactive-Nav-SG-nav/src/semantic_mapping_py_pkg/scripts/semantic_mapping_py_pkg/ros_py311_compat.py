import logging
import sys


def patch_roslogging_findcaller_for_py311():
    """Patch rospy logging for Python 3.11+, where rosgraph expects a 4-tuple from findCaller()."""
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_semantic_mapping_py_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    _safe_find_caller._semantic_mapping_py_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller
