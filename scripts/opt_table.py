#!/usr/bin/env python3
"""Print the measured optimization table, or rewrite it inside docs/ARCHITECTURE.md.

python scripts/opt_table.py            # print
python scripts/opt_table.py --write    # update docs/ARCHITECTURE.md between its markers
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyc.gpu.report import opt_table_markdown

START, END = "<!-- opt-table:start -->", "<!-- opt-table:end -->"


def main() -> int:
    table = opt_table_markdown()
    if "--write" not in sys.argv:
        print(table)
        return 0
    doc = ROOT / "docs" / "ARCHITECTURE.md"
    text = doc.read_text()
    new = re.sub(
        f"{re.escape(START)}.*?{re.escape(END)}", f"{START}\n{table}\n{END}", text, flags=re.DOTALL
    )
    doc.write_text(new)
    print(f"updated {doc.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
