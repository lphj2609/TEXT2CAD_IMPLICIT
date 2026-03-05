#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pdf_to_implicit_mesh.py

전체 파이프라인 (원샷 실행):

1) PDF 로딩 + MoE + Multi-task SFT + GRPO + KD + Text2CAD 파이프라인
   - `train_pipeline.run_text2cad_pipeline(config, device)` 호출
   - 설정에 따라 Text2CAD 학습(SFT/GRPO/KD) 및 CAD 샘플 생성까지 수행
2) 학습/혹은 기존 체크포인트 기반 DoosanText2CAD 로드
3) PDF에서 간단한 프롬프트들을 추출
4) `ImplicitMeshDecoder` + SDF + marching_cubes 로 뉴럴 implicit 메쉬(PLY) 생성

실행:
    cd D:\\doosan
    python run_pdf_to_implicit_mesh.py

출력:
    - Text2CAD 체크포인트 및 CAD 샘플:
        checkpoints/text2cad/...
    - implicit mesh 결과:
        checkpoints/text2cad_implicit_mesh/...
"""

from __future__ import annotations

from pathlib import Path

import torch


def main() -> None:
    # 1. 공통 설정/디바이스
    from train_pipeline import load_config, run_text2cad_pipeline

    project_root = Path(__file__).parent
    config_path = project_root / "config" / "model_config.yaml"
    config = load_config(str(config_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FullPipeline] Using device: {device}")

    # 2. Text2CAD + MoE + GRPO + KD 파이프라인 실행
    t2c_cfg = config.get("text2cad", {})
    if t2c_cfg.get("use_text2cad", False):
        print("\n[FullPipeline] Step 1: Text2CAD + MoE 파이프라인 실행 (SFT/GRPO/KD + CAD 샘플 생성)...")
        _ = run_text2cad_pipeline(config, device)
        print("[FullPipeline] Text2CAD 파이프라인 단계 완료.")
    else:
        print(
            "[FullPipeline] config.text2cad.use_text2cad 가 false로 되어 있어 "
            "Text2CAD 학습을 건너뜁니다. (기존 체크포인트만 사용하거나 랜덤 초기화 모델 사용)"
        )

    # 3. implicit mesh 모듈에서 Text2CAD 모델 및 프롬프트 로딩
    from run_text2cad_implicit_mesh import (  # type: ignore[import-not-found]
        _load_doosan_text2cad,
        _extract_prompts_from_pdfs,
        generate_implicit_meshes_from_text,
    )

    print("\n[FullPipeline] Step 2: DoosanText2CAD 모델 로드 (체크포인트가 있으면 로드)...")
    model = _load_doosan_text2cad(config, device)

    print("\n[FullPipeline] Step 3: PDF에서 프롬프트 추출...")
    prompts = _extract_prompts_from_pdfs(config)
    print(f"[FullPipeline]   - 추출된 프롬프트 개수: {len(prompts)}")

    # 4. implicit SDF + marching_cubes 로 메쉬 생성
    out_dir = project_root / "checkpoints" / "text2cad_implicit_mesh"
    print(f"\n[FullPipeline] Step 4: Implicit SDF + marching_cubes 메쉬 생성 → {out_dir}")
    generate_implicit_meshes_from_text(
        model=model,
        prompts=prompts,
        output_dir=out_dir,
        device=device,
        grid_res=64,
    )

    print("\n[FullPipeline] 전체 파이프라인 완료.")


if __name__ == "__main__":
    main()

