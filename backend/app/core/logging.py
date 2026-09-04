"""
Logging setup.

Each graph node logs a single structured line on entry/exit, in the
format the user asked to see in the console during a run:

  [Classifier] Dataset classified — type=Aggregated A/B Test Data, users=12400
  [Planner] Intent detected — Full Experiment Review
  [Validation] SRM passed (p=0.83)
  [Experiment] Welch's t-test completed — p=0.041, significant=True
  [Decision] Report generated — confidence=HIGH

This is deliberately plain `logging` for now (per project decision) —
swapping the handler/formatter for LangSmith tracing later requires no
changes to the call sites in graph/nodes/*, only to this setup.
"""

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_node_logger(node_name: str) -> logging.Logger:
    """
    Returns a plain logger for a graph node. Node code prefixes its own
    messages with the bracketed node name, e.g.:

        log = get_node_logger("Classifier")
        log.info("[Classifier] Dataset classified — type=%s, users=%d", ...)

    Kept as a thin wrapper (not a LoggerAdapter) so call sites stay
    simple standard-library logging calls — easy to swap for
    LangSmith's tracing decorators later without touching this file's
    public interface.
    """
    configure_logging()
    return logging.getLogger(f"graph.{node_name.lower()}")
