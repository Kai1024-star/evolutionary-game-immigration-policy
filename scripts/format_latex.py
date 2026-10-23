"""Keep the parameter table together in the standalone generated notes."""
from pathlib import Path
import re

path = Path(__file__).resolve().parents[1] / "docs/English_Model_Notes.tex"
text = path.read_text()
pattern = r"\\section\{Parameter\s+dictionary\s+and\s+baselines\}"
if not re.search(pattern, text):
    raise RuntimeError("Expected parameter-section marker was not generated")
text = re.sub(pattern, lambda m: "\\clearpage\n" + m.group(), text, count=1)
path.write_text(text)
