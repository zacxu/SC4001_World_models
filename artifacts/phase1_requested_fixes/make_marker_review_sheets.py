"""Make labelled QA sheets; retain each original 128x96 actor image unchanged."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parent
index = []
for path in sorted((root / "markers").glob("*/marker_audit.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    captures = report["captures"]
    columns = 3
    sheet = Image.new("RGB", (columns * 192, ((len(captures) + columns - 1) // columns) * 140), "#202020")
    draw = ImageDraw.Draw(sheet)
    records = []
    for position, capture in enumerate(captures):
        camera = capture["camera_player"]
        source = Path(capture["image_paths"][camera])
        with Image.open(source) as frame:
            if frame.size != (128, 96):
                raise ValueError(f"unexpected actor size: {source}")
            x, y = position % columns * 192, position // columns * 140
            draw.text((x + 3, y + 2), f"{path.parent.name} P{camera} {capture['state']}", fill="white")
            draw.text((x + 3, y + 16), capture["view"], fill="white")
            sheet.paste(frame.convert("RGB"), (x + 32, y + 35))
        records.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "camera_player": camera, "state": capture["state"], "view": capture["view"]})
    target = path.parent / "review_sheet.png"
    sheet.save(target)
    index.append({"case": path.parent.name, "sheet_path": str(target),
                  "sheet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": records})
(root / "marker_review_sheet_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"sheets": len(index), "frames": sum(len(item["frames"]) for item in index)}))
