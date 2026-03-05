# Text2CAD-Implicit Runfiles Description

This document briefly describes the contents of `text2cad_implicit_runfiles.zip` and how to use them.  
After extracting the archive, restore the files under the project root (`D:\doosan`) with the same directory layout.

---

## Included Files

| Path | Description |
|------|-------------|
| `train_pipeline.py` | **Main training entry point**. MoE + Multi-task SFT + GRPO + KD + (optional) Text2CAD integrated training. |
| `run_pdf_to_implicit_mesh.py` | **Full pipeline one-shot run**. Runs training → PDF prompt extraction → implicit mesh (PLY) generation in one go. |
| `run_text2cad_implicit_mesh.py` | **Inference / mesh generation**. Text2CAD + ImplicitMeshDecoder + SDF + marching_cubes to produce PLY. Supports CLI options. |
| `run_text2cad_implicit_mesh_2.py` | **Experiment clone** of the same pipeline with a separate config. |
| `scripts/run_train_with_occ.bat` | **Run training with OCC environment** (Windows). Runs `train_pipeline.py` with `.occ_env` Python and sets OpenCASCADE DLL paths. |
| `config/model_config.yaml` | **Global config**: MoE, GRPO, Text2CAD flags, data paths, etc. |

---

## How to Run

### 1. Training only

```bash
# Default Python/conda environment
python train_pipeline.py

# With OpenCASCADE (Windows, when .occ_env exists)
scripts\run_train_with_occ.bat
```

### 2. Full pipeline (training → mesh generation)

```bash
cd D:\doosan
python run_pdf_to_implicit_mesh.py
```

- Outputs: `checkpoints/text2cad/` (checkpoints and CAD samples), `checkpoints/text2cad_implicit_mesh/` (PLY meshes).

### 3. Mesh generation with a trained model only

```bash
python run_text2cad_implicit_mesh.py --prompt "..." --output-dir ... [--cad-latent-path ...]
```

- Use `run_text2cad_implicit_mesh_2.py` the same way for separate experiment configs.

---

## Dependencies and Paths

- The project root must be `D:\doosan` at run time; the full repo layout (e.g. `src/`, `config/`) is required.
- The zip contains **only the entry scripts and config** needed to run. Keep the rest of the repo (e.g. model and data code under `src/`) in place or use the full repository.
- For pipeline details, see `PIPELINE_OVERVIEW_TEXT2CAD_IMPLICIT.md`.

---

## Regenerating the zip

Run `model/make_runfiles_zip.py` to recreate `text2cad_implicit_runfiles.zip` with the same file list:

```bash
python D:\doosan\model\make_runfiles_zip.py
```
