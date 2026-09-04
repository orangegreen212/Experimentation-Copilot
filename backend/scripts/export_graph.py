"""
Regenerates docs/graph.mmd directly from the compiled LangGraph object.

Run this after any change to graph_builder.py's node/edge structure:

    python3 scripts/export_graph.py

The diagram is never hand-edited — it's always a mechanical dump of
`experiment_review_graph.get_graph().draw_mermaid()`, so it can't drift
from what the graph actually does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.graph_builder import export_mermaid  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "graph.mmd"


def main() -> None:
    mermaid = export_mermaid()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(mermaid)
    print(f"Wrote {OUTPUT_PATH}")
    print(mermaid)


if __name__ == "__main__":
    main()
