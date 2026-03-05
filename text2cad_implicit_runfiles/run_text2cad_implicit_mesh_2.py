#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_text2cad_implicit_mesh_2.py

run_text2cad_implicit_mesh.py 를 그대로 복제한 버전입니다.
실험 설정을 분리해서 사용하기 위한 클론 스크립트입니다.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional
import math
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    # marching_cubes를 위해 scikit-image가 있으면 사용
    from skimage.measure import marching_cubes

    _HAS_SKIMAGE = True
except Exception:
    marching_cubes = None
    _HAS_SKIMAGE = False

try:
    import trimesh

    _HAS_TRIMESH = True
except Exception:
    trimesh = None
    _HAS_TRIMESH = False

try:
    import pyglet  # trimesh viewer backend

    _HAS_PYGLET = True
except Exception:
    pyglet = None  # type: ignore
    _HAS_PYGLET = False

try:
    import pyrender

    _HAS_PYRENDER = True
except Exception:
    pyrender = None
    _HAS_PYRENDER = False

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

try:
    import imageio.v2 as imageio  # type: ignore

    _HAS_IMAGEIO = True
except Exception:
    try:
        import imageio  # type: ignore

        _HAS_IMAGEIO = True
    except Exception:
        imageio = None
        _HAS_IMAGEIO = False

try:
    # 선택 사항: depth-to-image 모델이 설치된 경우에만 사용
    from diffusers import StableDiffusionDepth2ImgPipeline  # type: ignore

    _HAS_DIFFUSERS = True
except Exception:
    StableDiffusionDepth2ImgPipeline = None  # type: ignore
    _HAS_DIFFUSERS = False


# -----------------------------------------------------------------------------
# 간단한 Implicit Mesh Decoder (cad_vec + xyz → SDF)
# -----------------------------------------------------------------------------


class ImplicitMeshDecoder(nn.Module):
    """
    입력:
        - cad_latent: (B, D) Text2CAD에서 나온 cad_vec (또는 그 일부)
    출력:
        - sdf_grid: (B, R, R, R) SDF 값 (voxel grid)

    기본 설계는 DeepSDF 스타일의 좌표 기반 MLP이지만,
    단일 스텝 noise injection + denoising MLP 를 추가해
    diffusion-풍 regularization 을 위한 경량 버전으로 구성한다.
    (현재 스크립트에서는 학습되지 않은 상태로도 실행만 되도록 설계)
    """

    def __init__(
        self,
        cad_dim: int,
        hidden_dim: int = 256,
        grid_res: int = 64,
        bound: float = 1.2,
        plane_feat_dim: int = 32,
        use_denoising: bool = True,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_res = grid_res
        self.bound = bound
        self.use_denoising = use_denoising
        self.noise_std = noise_std
        self.plane_feat_dim = plane_feat_dim

        # Tri-plane feature maps: XY, YZ, ZX (C, R, R)
        self.feat_xy = nn.Parameter(
            torch.randn(plane_feat_dim, grid_res, grid_res) * 0.01
        )
        self.feat_yz = nn.Parameter(
            torch.randn(plane_feat_dim, grid_res, grid_res) * 0.01
        )
        self.feat_zx = nn.Parameter(
            torch.randn(plane_feat_dim, grid_res, grid_res) * 0.01
        )

        # cad_vec 를 plane feature와 비슷한 차원으로 투영하는 작은 MLP
        self.cad_proj = nn.Sequential(
            nn.Linear(cad_dim, plane_feat_dim),
            nn.SiLU(),
        )

        # Tri-plane에서 샘플링한 feature F(x)와 cad feature g(c)를 합친 뒤
        # 작은 MLP 로 SDF를 예측하는 TinyMLP
        in_dim = plane_feat_dim * 3 + plane_feat_dim  # [F_xy, F_yz, F_zx, g(c)]
        self.base_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 단일-스텝 denoising MLP: noisy SDF → refined SDF (diffusion-풍 regularization)
        self.denoise_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cad_latent: torch.Tensor) -> torch.Tensor:
        """
        cad_latent: (B, D)
        return: (B, R, R, R) SDF grid
        """
        B, D = cad_latent.shape
        R = self.grid_res

        device = cad_latent.device

        # 3D voxel 인덱스 (i,j,k) 생성: 0..R-1
        idx = torch.arange(R, device=device)
        ii, jj, kk = torch.meshgrid(idx, idx, idx, indexing="ij")  # (R,R,R)
        ii_f = ii.flatten()  # (N,)
        jj_f = jj.flatten()
        kk_f = kk.flatten()
        N = ii_f.numel()

        # Tri-plane features에서 좌표별 feature 샘플링
        # F_xy(x,y), F_yz(y,z), F_zx(z,x)
        F_xy = self.feat_xy[:, ii_f, jj_f].permute(1, 0)  # (N, C)
        F_yz = self.feat_yz[:, jj_f, kk_f].permute(1, 0)  # (N, C)
        F_zx = self.feat_zx[:, kk_f, ii_f].permute(1, 0)  # (N, C)
        F_planes = torch.cat([F_xy, F_yz, F_zx], dim=-1)  # (N, 3C)

        # cad_vec 를 plane feature 차원으로 투영 후, 좌표마다 broadcast
        cad_feat = self.cad_proj(cad_latent)  # (B, C)
        cad_feat_exp = cad_feat.unsqueeze(1).expand(B, N, -1)  # (B, N, C)

        F_planes_exp = F_planes.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3C)

        # 최종 TinyMLP 입력: [F_planes(x), g(cad_vec)]
        x = torch.cat([F_planes_exp, cad_feat_exp], dim=-1)  # (B, N, 4C)
        x = x.view(B * N, -1)

        # 기본 SDF 예측
        sdf = self.base_mlp(x)  # (B*N, 1)

        if self.use_denoising and self.noise_std > 0:
            # 단일 스텝 noise 주입
            noise = torch.randn_like(sdf) * self.noise_std
            sdf_noisy = sdf + noise
            # denoising MLP 를 통한 간단한 regularization
            sdf = self.denoise_mlp(sdf_noisy)

        sdf = sdf.view(B, self.grid_res, self.grid_res, self.grid_res)
        return sdf


def _render_depth_map_from_mesh(mesh, img_size: int = 256) -> Optional[np.ndarray]:
    """
    메쉬를 기준 시점에서 렌더링한 depth map을 생성한다.
    - pyrender가 있으면 실제 오프스크린 렌더링 사용
    - pyrender가 없으면 메쉬의 bounding box를 기반으로 한 간단한
      절차적(radial) depth map을 생성해 사용한다.
    depth 값은 [0, 1] 범위로 정규화된다.
    """
    if not _HAS_PYRENDER:
        # pyrender가 없을 때는 메쉬 중심을 기준으로 한 간단한 radial depth map 생성
        h = w = img_size
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r_min, r_max = float(r.min()), float(r.max())
        depth = (r - r_min) / (r_max - r_min + 1e-8)
        depth = 1.0 - depth  # 중심이 가깝고, 바깥이 먼 형태
        return depth.astype(np.float32)

    scene = pyrender.Scene()
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_pr)

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    cam_dist = 2.5
    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, 3] = np.array([0.0, 0.0, cam_dist], dtype=np.float32)
    cam_node = scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    scene.remove_node(cam_node)
    scene.remove_node(light_node)

    depth = depth.astype(np.float32)
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return None
    dmin, dmax = float(valid.min()), float(valid.max())
    if dmax - dmin < 1e-6:
        return None
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)
    depth_norm[~np.isfinite(depth_norm)] = 1.0
    return depth_norm


def _generate_texture_from_prompt_and_depth(
    prompt: str,
    depth_map: np.ndarray,
    out_path: Path,
) -> bool:
    """
    prompt + depth map을 이용해 텍스처 이미지를 생성한다.
    - diffusers(StableDiffusionDepth2Img)가 설치되어 있으면 실제 depth-to-image 사용
    - 없으면 prompt를 해시한 색상으로 간단한 체커보드 텍스처 생성
    반환값: True면 모델 기반, False면 procedural 기반.
    """
    h, w = depth_map.shape

    if _HAS_DIFFUSERS and _HAS_PIL:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-depth"
            )
            pipe = pipe.to(device)

            depth_img = Image.fromarray((depth_map * 255.0).astype("uint8"))
            init_image = Image.new("RGB", (w, h), color=(128, 128, 128))

            result = pipe(
                prompt=prompt,
                image=init_image,
                depth_map=depth_img,
                strength=0.8,
                guidance_scale=7.5,
            ).images[0]
            result.save(out_path)
            return True
        except Exception as e:
            print(f"[ImplicitMesh] depth-to-image 텍스처 생성 실패({e}) → procedural 텍스처 사용")

    # Procedural fallback (Pillow 필요)
    if not _HAS_PIL:
        print("[ImplicitMesh] Pillow 미설치로 텍스처 생성을 건너뜁니다. (pip install pillow)")
        return False

    import hashlib

    hsh = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    base_color = np.array(
        [
            int(hsh[0:2], 16),
            int(hsh[2:4], 16),
            int(hsh[4:6], 16),
        ],
        dtype=np.uint8,
    )

    img = np.zeros((h, w, 3), dtype=np.uint8)
    tile = max(4, min(h, w) // 16)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx // tile + yy // tile) % 2) == 0
    img[:] = (base_color // 2)[None, None, :]
    img[mask] = base_color

    tex_img = Image.fromarray(img)
    tex_img.save(out_path)
    return False


def _save_textured_obj(
    mesh,
    texture_filename: str,
    obj_path: Path,
    mtl_path: Path,
) -> None:
    """
    메쉬를 OBJ + MTL로 저장하고, 단순 XZ 평면 기준 UV를 생성한다.
    texture_filename은 OBJ/MTL과 같은 디렉터리에 있는 PNG를 가리킨다고 가정.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # XZ 평면 투영으로 UV 생성
    xz = verts[:, [0, 2]]
    mins = xz.min(axis=0)
    maxs = xz.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-6)
    uv = (xz - mins) / denom  # [0,1]

    # MTL 작성
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("newmtl material_0\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ka 0.000000 0.000000 0.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.0\n")
        f.write(f"map_Kd {texture_filename}\n")

    # OBJ 작성
    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl material_0\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uv:
            # v 좌표는 이미지 좌표계와 맞추기 위해 뒤집는다.
            f.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")

        # f: 인덱스는 1부터 시작, v/vt 인덱스를 동일하게 사용
        for face in faces:
            i1, i2, i3 = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")


def _apply_prompt_based_texture(
    mesh,
    prompt: str,
    output_dir: Path,
    sample_idx: int,
) -> None:
    """
    Figure 3 스타일 아이디어를 단순화한 텍스처링 단계:
    - mesh → depth map (procedural layout의 결과를 depth로 본다고 가정)
    - prompt + depth map → 텍스처 이미지 (모델 또는 procedural)
    - OBJ + MTL + PNG 조합으로 텍스처드 메쉬 저장
    """
    depth = _render_depth_map_from_mesh(mesh, img_size=256)
    if depth is None:
        print("[ImplicitMesh] depth map 생성 실패로 텍스처링을 건너뜁니다.")
        return

    tex_path = output_dir / f"sample_{sample_idx:03d}_tex.png"
    used_model = _generate_texture_from_prompt_and_depth(prompt, depth, tex_path)

    obj_path = output_dir / f"sample_{sample_idx:03d}.obj"
    mtl_path = output_dir / f"sample_{sample_idx:03d}.mtl"
    _save_textured_obj(mesh, tex_path.name, obj_path, mtl_path)

    print(
        f"  → textured OBJ saved: {obj_path.name} "
        f"(texture: {tex_path.name}, model_based={used_model})"
    )


def _interactive_preview_mesh(mesh) -> None:
    """
    간단한 마우스 인터랙티브 뷰어.
    - 왼쪽 버튼 드래그: 화면 평면 상에서 메쉬 전체 평행 이동
    - 오른쪽 버튼 드래그: 화면 평면 기준 회전 (yaw/pitch)
    - 가운데 버튼 드래그 또는 휠 스크롤: 줌 인/아웃
    - 별도의 상태 저장은 하지 않고, 시각적 확인용으로만 사용.
    """
    # pyglet 또는 trimesh viewer가 없으면 조용히 건너뜀 (에러/경고 출력 X)
    if not (_HAS_TRIMESH and _HAS_PYGLET):
        return
    try:
        import pyglet  # type: ignore
        from trimesh.viewer import SceneViewer  # type: ignore

        scene = trimesh.Scene(mesh)
        viewer = SceneViewer(scene, start_loop=False)

        drag_state = {"button": None}

        @viewer.event  # type: ignore
        def on_mouse_press(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                drag_state["button"] = "left"
            elif button == pyglet.window.mouse.RIGHT:
                drag_state["button"] = "right"
            elif button == pyglet.window.mouse.MIDDLE:
                drag_state["button"] = "middle"

        @viewer.event  # type: ignore
        def on_mouse_release(x, y, button, modifiers):
            drag_state["button"] = None

        @viewer.event  # type: ignore
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            btn = drag_state["button"]
            if btn is None:
                return

            if btn == "left":
                # 화면 평면 기준으로 X/Y 방향 평행 이동
                scale = 0.01
                T = np.eye(4, dtype=np.float32)
                T[0, 3] += dx * scale
                T[1, 3] -= dy * scale
                scene.apply_transform(T)

            elif btn == "right":
                # yaw/pitch 회전
                rot_scale = 0.005
                yaw = dx * rot_scale    # 수평 이동 → 수평 회전
                pitch = dy * rot_scale  # 수직 이동 → 수직 회전

                cy, sy = math.cos(yaw), math.sin(yaw)
                cp, sp = math.cos(pitch), math.sin(pitch)

                Ry = np.array(
                    [[cy, 0.0, sy, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [-sy, 0.0, cy, 0.0],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                Rx = np.array(
                    [[1.0, 0.0, 0.0, 0.0],
                     [0.0, cp, -sp, 0.0],
                     [0.0, sp, cp, 0.0],
                     [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                _ = Ry @ Rx


def _sdf_to_trimesh(
    sdf: np.ndarray,
    bound: float,
) -> Optional["trimesh.Trimesh"]:
    """
    SDF voxel grid을 marching_cubes로 메쉬로 변환한다.
    - sdf: (R, R, R) numpy 배열, 음수 영역이 내부.
    - bound: 좌표계를 [-bound, bound]^3 로 매핑.
    """
    if not _HAS_SKIMAGE:
        print("[ImplicitMesh] scikit-image(marching_cubes) 미설치로 메쉬 생성을 건너뜁니다.")
        return None
    if not _HAS_TRIMESH:
        print("[ImplicitMesh] trimesh 미설치로 메쉬 생성을 건너뜁니다.")
        return None

    sdf = sdf.astype(np.float32)
    try:
        verts, faces, normals, _ = marching_cubes(sdf, level=0.0)
    except Exception as e:
        print(f"[ImplicitMesh] marching_cubes 실패: {e}")
        return None

    R = sdf.shape[0]
    if R > 1:
        coords = (verts / (R - 1.0)) * 2.0 - 1.0
    else:
        coords = verts
    coords = coords * float(bound)

    try:
        mesh = trimesh.Trimesh(
            vertices=coords,
            faces=faces.astype(np.int64),
            vertex_normals=normals,
            process=True,
        )
    except Exception as e:
        print(f"[ImplicitMesh] trimesh 메쉬 생성 실패: {e}")
        return None
    return mesh


def _run_single_sample(
    cad_latent: np.ndarray,
    cad_dim: int,
    output_dir: Path,
    prompt: Optional[str] = None,
    grid_res: int = 64,
    hidden_dim: int = 256,
    plane_feat_dim: int = 32,
    use_denoising: bool = True,
    noise_std: float = 0.1,
    interactive: bool = False,
    save_texture: bool = True,
) -> None:
    """
    하나의 cad_latent 벡터에 대해:
    - ImplicitMeshDecoder로 SDF 생성
    - marching_cubes로 메쉬 생성
    - OBJ(+MTL), PNG 텍스처 저장
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cad_latent_t = np.asarray(cad_latent, dtype=np.float32)
    if cad_latent_t.ndim == 1:
        cad_latent_t = cad_latent_t[None, :]
    if cad_latent_t.shape[1] != cad_dim:
        raise ValueError(
            f"cad_latent 차원 불일치: expected {cad_dim}, got {cad_latent_t.shape[1]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    decoder = ImplicitMeshDecoder(
        cad_dim=cad_dim,
        hidden_dim=hidden_dim,
        grid_res=grid_res,
        bound=1.2,
        plane_feat_dim=plane_feat_dim,
        use_denoising=use_denoising,
        noise_std=noise_std,
    ).to(device)
    decoder.eval()

    with torch.no_grad():
        cad_tensor = torch.from_numpy(cad_latent_t).to(device)
        sdf_grid = decoder(cad_tensor)[0].cpu().numpy()

    np.save(output_dir / "sdf_grid.npy", sdf_grid)
    print(f"[ImplicitMesh] SDF grid 저장: {output_dir / 'sdf_grid.npy'}")

    mesh = _sdf_to_trimesh(sdf_grid, bound=decoder.bound)
    if mesh is None:
        return

    if _HAS_TRIMESH:
        mesh_path = output_dir / "implicit_mesh.ply"
        try:
            mesh.export(mesh_path)
            print(f"[ImplicitMesh] 메쉬 저장: {mesh_path}")
        except Exception as e:
            print(f"[ImplicitMesh] 메쉬 저장 실패: {e}")

    if save_texture:
        prompt_text = prompt or "Doosan Text2CAD implicit mesh sample"
        _apply_prompt_based_texture(mesh, prompt_text, output_dir, sample_idx=0)

    if interactive:
        _interactive_preview_mesh(mesh)


def _load_cad_latent(path: Path) -> np.ndarray:
    """
    .npy 또는 .npz 에 저장된 cad_latent 를 로드한다.
    - .npy: (D,) 또는 (B,D)
    - .npz: 'cad_latent' 키를 우선 사용, 없으면 첫 배열 사용
    """
    if not path.is_file():
        raise FileNotFoundError(f"cad_latent 파일을 찾을 수 없습니다: {path}")

    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "cad_latent" in data:
            return data["cad_latent"]
        first_key = list(data.files)[0]
        return data[first_key]
    raise ValueError(f"지원하지 않는 cad_latent 포맷입니다: {path.suffix}")


def main(argv: Optional[List[str]] = None) -> None:
    """
    예시 실행:
      python run_text2cad_implicit_mesh_2.py ^
        --cad-latent-path data/text2cad_sample/cad_latent.npy ^
        --cad-dim 512 ^
        --output-dir outputs/implicit_mesh_sample ^
        --prompt "Doosan compressor head"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Text2CAD latent → Implicit SDF/mesh 생성 스크립트 (run_text2cad_implicit_mesh_2)."
    )
    parser.add_argument(
        "--cad-latent-path",
        type=str,
        required=True,
        help="Text2CAD에서 저장한 cad_latent .npy/.npz 파일 경로",
    )
    parser.add_argument(
        "--cad-dim",
        type=int,
        required=True,
        help="cad_latent 의 차원 D (Text2CAD latent dimension)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/implicit_mesh",
        help="SDF/mesh/텍스처를 저장할 디렉터리",
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=64,
        help="SDF voxel 해상도 R (기본 64)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="텍스처 생성에 사용할 프롬프트 (없으면 기본값 사용)",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="텍스처 생성/OBJ+MTL 저장을 건너뜀",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="trimesh 뷰어를 사용한 인터랙티브 미리보기 실행 (설치된 경우)",
    )

    args = parser.parse_args(argv)

    cad_path = Path(args.cad_latent_path)
    output_dir = Path(args.output_dir)

    cad_latent = _load_cad_latent(cad_path)

    _run_single_sample(
        cad_latent=cad_latent,
        cad_dim=args.cad_dim,
        output_dir=output_dir,
        prompt=args.prompt,
        grid_res=args.grid_res,
        interactive=args.interactive,
        save_texture=not args.no_texture,
    )


if __name__ == "__main__":
    main()

