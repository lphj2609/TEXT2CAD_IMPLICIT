# Text2CAD-Implicit 실행 파일 설명

`text2cad_implicit_runfiles.zip`에 포함된 파일들과 사용 방법을 간략히 정리한 문서입니다.  
압축 해제 후 프로젝트 루트(`D:\doosan`)에 동일한 디렉터리 구조로 복원하여 사용하세요.

---

## 포함 파일 목록

| 경로 | 설명 |
|------|------|
| `train_pipeline.py` | **학습 메인 진입점**. MoE + Multi-task SFT + GRPO + KD + (옵션) Text2CAD 통합 학습. |
| `run_pdf_to_implicit_mesh.py` | **전체 파이프라인 원샷 실행**. 학습 → PDF 프롬프트 추출 → implicit 메쉬(PLY) 생성까지 한 번에 수행. |
| `run_text2cad_implicit_mesh.py` | **추론/메쉬 생성**. Text2CAD + ImplicitMeshDecoder + SDF + marching_cubes로 PLY 생성. CLI 옵션 지원. |
| `run_text2cad_implicit_mesh_2.py` | 위와 동일 파이프라인의 **설정 분리 실험용** 복제본. |
| `scripts/run_train_with_occ.bat` | **OCC 환경으로 학습 실행** (Windows). `.occ_env` Python으로 `train_pipeline.py` 실행, OpenCASCADE DLL 경로 설정. |
| `config/model_config.yaml` | MoE, GRPO, Text2CAD 사용 여부, 데이터 경로 등 **전체 설정** 파일. |

---

## 실행 방법 요약

### 1. 학습만 수행

```bash
# 일반 Python/conda 환경
python train_pipeline.py

# OpenCASCADE 사용 시 (Windows, .occ_env 있을 때)
scripts\run_train_with_occ.bat
```

### 2. 학습부터 메쉬 생성까지 한 번에

```bash
cd D:\doosan
python run_pdf_to_implicit_mesh.py
```

- 출력: `checkpoints/text2cad/` (체크포인트·CAD 샘플), `checkpoints/text2cad_implicit_mesh/` (PLY 메쉬)

### 3. 이미 학습된 모델로 메쉬만 생성

```bash
python run_text2cad_implicit_mesh.py --prompt "..." --output-dir ... [--cad-latent-path ...]
```

- `run_text2cad_implicit_mesh_2.py`는 별도 실험 설정용으로 동일한 방식으로 실행.

---

## 의존성 및 경로

- 실행 시 프로젝트 루트가 `D:\doosan`이어야 하며, `src/`, `config/` 등 전체 레포 구조가 필요합니다.
- zip에는 **실행/설정에 필요한 진입점만** 포함되어 있습니다. `src/` 내 모델·데이터 코드는 별도로 두거나 전체 레포를 사용하세요.
- 상세 파이프라인 구성은 `PIPELINE_OVERVIEW_TEXT2CAD_IMPLICIT.md`를 참고하세요.

---

## zip 재생성

`model/make_runfiles_zip.py`를 실행하면 동일한 목록으로 `text2cad_implicit_runfiles.zip`을 다시 만들 수 있습니다.

```bash
python D:\doosan\model\make_runfiles_zip.py
```
