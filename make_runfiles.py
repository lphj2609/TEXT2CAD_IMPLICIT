"""Pack Text2CAD-Implicit runfiles into a zip under D:\\doosan\\model."""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent  # D:\doosan
model_dir = Path(__file__).resolve().parent    # D:\doosan\model

files = [
    "train_pipeline.py",
    "run_pdf_to_implicit_mesh.py",
    "run_text2cad_implicit_mesh.py",
    "run_text2cad_implicit_mesh_2.py",
    "scripts/run_train_with_occ.bat",
    "config/model_config.yaml",
]

zip_path = model_dir / "text2cad_implicit_runfiles.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        p = root / f
        if p.exists():
            zf.write(p, f)
            print("Added:", f)
        else:
            print("Skip (not found):", f)
print("Created:", zip_path)
