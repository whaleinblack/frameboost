# Converts the RIFE ONNX file to the raw weights the WGSL runtime loads: models/rife49.bin + .json.
# Usage: python export_model.py path/to/rife49.onnx
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

models = Path(__file__).resolve().parent.parent / "models"
models.mkdir(exist_ok=True)
graph = onnx.load(sys.argv[1]).graph

index = {}
chunks = []
offset = 0
for init in graph.initializer:
    if not init.name.startswith(("encode.", "block")):
        continue
    data = numpy_helper.to_array(init).astype(np.float32)
    index[init.name] = {"offset": offset, "shape": list(data.shape)}
    chunks.append(data.tobytes())
    offset += data.size

(models / "rife49.bin").write_bytes(b"".join(chunks))
(models / "rife49.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf8")
print(len(index), "tensors,", offset, "weights,", offset * 4, "bytes")
