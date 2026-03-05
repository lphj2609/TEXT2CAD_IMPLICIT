#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_text2cad_implicit_mesh.py

목표:
- OCC / CadSeqProc 없이 Text2CAD + 뉴럴 implicit SDF + Marching Cubes로 메쉬를 생성하는 예시 파이프라인.
- 기존 `run_text2cad.py` 와는 별도의 실험용 스크립트이며, STEP/B-Rep 대신
  voxel SDF → marching_cubes 로 얻은 삼각형 메쉬(PLY)를 출력한다.

주의:
- 여기의 ImplicitMeshDecoder는 단순 MLP 초기화만 되어 있고, 학습되지 않은 상태에서는
  의미 있는 메쉬가 나오지 않는다. 실제 사용을 위해서는 본 스크립트로 만든 데이터셋
  (예: Text2CAD cad_vec + SDF)을 기반으로 별도의 학습 루프를 구성해야 한다.
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

    설계:
        - 좌표 grid (R^3)를 [-1,1]^3 범위로 생성
        - 각 점마다 cad_latent를 concat → MLP → 스칼라 SDF
    """

    def __init__(
        self,
        cad_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        grid_res: int = 64,
        bound: float = 1.2,
    ) -> None:
        super().__init__()
        self.grid_res = grid_res
        self.bound = bound

        layers: List[nn.Module] = []
        in_dim = cad_dim + 3  # [cad_latent, x, y, z]
        dim = hidden_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cad_latent: torch.Tensor) -> torch.Tensor:
        """
        cad_latent: (B, D)
        return: (B, R, R, R) SDF grid
        """
        B, D = cad_latent.shape
        R = self.grid_res

        # 3D 좌표 grid 생성
        lin = torch.linspace(-self.bound, self.bound, R, device=cad_latent.device)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 3)  # (1, N, 3)
        coords = coords.expand(B, -1, -1)  # (B, N, 3)

        # cad_latent 를 모든 좌표에 concat
        cad_expanded = cad_latent.unsqueeze(1).expand(-1, coords.shape[1], -1)  # (B, N, D)
        x = torch.cat([cad_expanded, coords], dim=-1)  # (B, N, D+3)

        x = x.view(B * coords.shape[1], -1)
        sdf = self.mlp(x)  # (B*N, 1)
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
                R = Ry @ Rx
                scene.apply_transform(R)

            elif btn == "middle":
                # 드래그로 줌 (dy 기준)
                zoom_scale = 0.005
                factor = 1.0 - dy * zoom_scale
                factor = max(0.2, min(5.0, factor))
                S = np.eye(4, dtype=np.float32)
                S[0, 0] *= factor
                S[1, 1] *= factor
                S[2, 2] *= factor
                scene.apply_transform(S)

        @viewer.event  # type: ignore
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # 휠 스크롤로도 줌 인/아웃
            zoom_scale = 0.1
            factor = 1.0 + scroll_y * zoom_scale
            factor = max(0.2, min(5.0, factor))
            S = np.eye(4, dtype=np.float32)
            S[0, 0] *= factor
            S[1, 1] *= factor
            S[2, 2] *= factor
            scene.apply_transform(S)

        pyglet.app.run()
    except Exception:
        # 인터랙티브 뷰어는 필수 기능이 아니므로, 실패 시 조용히 무시한다.
        return


def _generate_orbit_video_for_meshes(output_dir: Path, fps: int = 24, num_frames: int = 120) -> None:
    """
    Astra의 비디오 유틸(diffsynth.data.save_video)을 활용해,
    텍스처가 입혀진 메쉬 OBJ들을 카메라 궤도를 따라 렌더링한 영상을 생성한다.

    - 입력: output_dir 안의 sample_XXX.obj + sample_XXX_tex.png
    - 출력: sample_XXX_orbit.mp4 (카메라가 객체 주변을 한 바퀴 도는 비디오)
    """
    obj_files = sorted(output_dir.glob("sample_*.obj"))
    if not obj_files:
        return

    # Astra 비디오 저장 유틸 로드 시도
    save_video_fn = None
    try:
        astra_root = Path(__file__).parent / "model" / "Astra" / "Astra"
        if astra_root.exists():
            if str(astra_root) not in sys.path:
                sys.path.append(str(astra_root))
            from diffsynth.data import save_video as astra_save_video  # type: ignore
            save_video_fn = astra_save_video
    except Exception:
        save_video_fn = None

    # fallback: imageio 직접 사용. codec 명시로 PyAV "expected bytes, NoneType found" 방지
    if save_video_fn is None and _HAS_IMAGEIO:
        def save_video_fn(frames, save_path, fps, quality=9, ffmpeg_params=None):
            path = str(save_path)
            # 1) imageio + codec 명시 (PyAV "expected bytes, NoneType found" 방지)
            try:
                try:
                    writer = imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264")
                except Exception:
                    try:
                        writer = imageio.get_writer(path, fps=fps, codec="libx264")
                    except Exception:
                        writer = imageio.get_writer(path, fps=fps)
                for frame in frames:
                    writer.append_data(np.array(frame))
                writer.close()
                return
            except Exception:
                pass
            # 2) ffmpeg CLI fallback (imageio/PyAV 실패 시)
            import subprocess
            import tempfile
            from PIL import Image as PILImage
            tmpdir = tempfile.mkdtemp()
            try:
                for i, frame in enumerate(frames):
                    p = os.path.join(tmpdir, f"frame_{i:05d}.png")
                    if hasattr(frame, "save"):
                        frame.save(p)
                    else:
                        PILImage.fromarray(np.asarray(frame).astype(np.uint8)).save(p)
                cmd = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            finally:
                for f in Path(tmpdir).glob("*.png"):
                    f.unlink(missing_ok=True)
                Path(tmpdir).rmdir()

    if save_video_fn is None:
        print("[ImplicitMesh] Astra / imageio 비디오 유틸을 찾지 못해 orbit 비디오 생성을 건너뜁니다.")
        return

    if not _HAS_TRIMESH:
        print("[ImplicitMesh] orbit 비디오 생성을 위해 trimesh가 필요합니다. (pip install trimesh)")
        return

    from PIL import Image as PILImage

    for obj_path in obj_files:
        try:
            mesh = trimesh.load_mesh(str(obj_path), process=False)
        except Exception as e:
            print(f"[ImplicitMesh] OBJ 로드 실패({obj_path}): {e}")
            continue

        frames: List[PILImage.Image] = []

        if _HAS_PYRENDER:
            import pyrender  # type: ignore

            scene = pyrender.Scene()
            mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            scene.add(mesh_pr)

            radius = float(np.linalg.norm(mesh.bounding_box.extents)) * 1.5 if mesh.bounding_box is not None else 2.0
            cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
            cam_node = scene.add(cam)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

            renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)

            for t in range(num_frames):
                theta = 2.0 * math.pi * float(t) / float(num_frames)
                cam_pos = np.array([radius * math.cos(theta), radius * 0.2, radius * math.sin(theta)], dtype=np.float32)
                up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                z = -cam_pos / np.linalg.norm(cam_pos)
                x = np.cross(up, z)
                if np.linalg.norm(x) < 1e-5:
                    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = np.stack([x, y, z], axis=1)
                pose[:3, 3] = cam_pos

                scene.set_pose(cam_node, pose)
                light_node = scene.add(light, pose=pose)
                color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                scene.remove_node(light_node)

                frame = PILImage.fromarray(color[..., :3])
                frames.append(frame)

            renderer.delete()

        else:
            # pyrender가 없으면 matplotlib의 3D 렌더링으로 orbit 영상 생성
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.faces)

                radius = float(np.linalg.norm(mesh.bounding_box.extents)) * 1.5 if mesh.bounding_box is not None else 2.0

                for t in range(num_frames):
                    theta = 360.0 * float(t) / float(num_frames)
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111, projection="3d")
                    ax.plot_trisurf(
                        verts[:, 0],
                        verts[:, 1],
                        verts[:, 2],
                        triangles=faces,
                        cmap="viridis",
                        linewidth=0.2,
                        antialiased=True,
                    )
                    ax.view_init(elev=20.0, azim=theta)
                    ax.set_axis_off()
                    fig.tight_layout()

                    fig.canvas.draw()
                    # 최신 Matplotlib에서는 buffer_rgba() 사용이 더 안전함
                    buf = np.asarray(fig.canvas.buffer_rgba(), dtype="uint8")
                    image = buf[..., :3]  # RGB만 사용
                    frames.append(PILImage.fromarray(image))
                    plt.close(fig)
            except Exception as e:
                print(f"[ImplicitMesh] matplotlib orbit 렌더링 실패({e}) → 비디오 생성을 건너뜁니다.")
                continue

        video_path = output_dir / f"{obj_path.stem}_orbit.mp4"
        print(f"[ImplicitMesh] orbit 비디오 저장: {video_path}")
        save_video_fn(frames, str(video_path), fps=fps)


# -----------------------------------------------------------------------------
# Text2CAD + implicit 메쉬 생성 루틴
# -----------------------------------------------------------------------------


def _load_doosan_text2cad(config: dict, device: str):
    """train_pipeline.py 와 동일한 방식으로 DoosanText2CAD 인스턴스를 만든다."""
    from train_pipeline import load_config  # type: ignore[attr-defined]
    from src.models.doosan_text2cad import DoosanText2CAD, _get_text2cad_config

    project_root = Path(__file__).parent
    t2c = config.get("text2cad", {})

    t2c_config_path = project_root / t2c.get("config_path", "src/models/Text2CAD/Cad_VLM/config/trainer.yaml")
    full_cfg = _get_text2cad_config(str(t2c_config_path))
    text_config = full_cfg.get("text_encoder", {})
    cad_config = full_cfg.get("cad_decoder", {})
    model_cfg = config.get("model", {})

    source_dir = config.get("data", {}).get("pdf_source_dir", "source")
    if not Path(source_dir).exists():
        source_dir = project_root / "source"
    source_dir = str(Path(source_dir).resolve())

    model = DoosanText2CAD(
        text_config=text_config,
        cad_config=cad_config,
        source_dir=source_dir,
        num_experts=12 if model_cfg.get("use_3d_experts", True) else 8,
        use_3d_experts=model_cfg.get("use_3d_experts", True),
        expert_dim=4096,
        moe_top_k=2,
        text_dim=1024,
        freeze_base_embedder=True,
    ).to(device)

    # 기존 Text2CAD 체크포인트가 있으면 로드
    ckpt_path = project_root / t2c.get("checkpoint_dir", "checkpoints/text2cad") / "doosan_text2cad_moe.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"[ImplicitMesh] Loaded Text2CAD checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[ImplicitMesh] Warning: failed to load checkpoint ({e}), using randomly initialized model.")

    return model


def _extract_prompts_from_pdfs(config: dict) -> List[str]:
    """PDF에서 간단한 프롬프트 몇 개를 추출한다."""
    from src.data.pdf_loader import PDFKnowledgeLoader

    project_root = Path(__file__).parent
    source_dir = project_root / config.get("data", {}).get("pdf_source_dir", "source")
    if not source_dir.exists():
        source_dir = project_root / "source"
    if not source_dir.exists():
        print("[ImplicitMesh] PDF source directory not found, using default toy prompts.")
        return ["a simple rectangular block", "a cylinder shape"]

    loader = PDFKnowledgeLoader(source_dir=str(source_dir))
    chunks = loader.load_pdfs()
    prompts: List[str] = [
        c["text"][:200] for c in chunks[:5] if c.get("text") and isinstance(c["text"], str) and len(c["text"]) > 20
    ]
    if not prompts:
        prompts = ["a simple rectangular block", "a cylinder shape"]
    return prompts


def generate_implicit_meshes_from_text(
    model: torch.nn.Module,
    prompts: List[str],
    output_dir: Path,
    device: str,
    grid_res: int = 64,
) -> None:
    """
    Text2CAD 모델과 ImplicitMeshDecoder를 사용해
    - 각 텍스트 프롬프트에 대해 cad_vec 생성
    - cad_vec → SDF grid → marching_cubes → mesh.ply 저장
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # cad_vec 차원은 Text2CAD의 test_decode 결과에서 확인 (샘플 하나 forward)
    model.eval()

    # 우선 cad_vec 크기를 알아내기 위해 한 번만 forward
    with torch.no_grad():
        tmp_prompt = prompts[0]
        pred = model.test_decode(
            texts=[tmp_prompt],
            maxlen=256,
            nucleus_prob=0.0,
            topk_index=1,
            device=device,
        )
        cad_vec_sample = pred["cad_vec"][0].to(device)
        cad_dim = cad_vec_sample.numel()

    print(f"[ImplicitMesh] Using cad_vec dimension: {cad_dim}")

    decoder = ImplicitMeshDecoder(cad_dim=cad_dim, grid_res=grid_res).to(device)

    if not _HAS_SKIMAGE:
        print(
            "[ImplicitMesh] Warning: scikit-image가 없어 marching_cubes를 사용할 수 없습니다.\n"
            "               메쉬 파일은 생성되지 않고 SDF grid만 .npy 로 저장됩니다.\n"
            "               (pip install scikit-image 로 설치 가능)"
        )

    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or len(prompt.strip()) < 3:
            continue
        prompt_text = prompt.strip()[:200]
        print(f"[ImplicitMesh] [{idx+1}/{len(prompts)}] prompt: {prompt_text[:60]}...")

        with torch.no_grad():
            out = model.test_decode(
                texts=[prompt_text],
                maxlen=256,
                nucleus_prob=0.0,
                topk_index=1,
                device=device,
            )
            cad_vec = out["cad_vec"][0].to(device).view(1, -1)  # (1, D)
            sdf = decoder(cad_vec)  # (1, R, R, R)

        sdf_np = sdf[0].cpu().numpy()
        np.save(output_dir / f"sample_{idx:03d}_sdf.npy", sdf_np)

        if not _HAS_SKIMAGE:
            continue

        # marching_cubes는 level 값이 [min, max] 범위 안에 있어야 한다.
        # 학습되지 않은 SDF의 경우 전체가 양수/음수 한쪽으로 치우쳐 있을 수 있으므로
        # 0 레벨셋이 항상 범위 안에 들어가도록 간단한 정규화를 수행한다.
        sdf_min, sdf_max = float(sdf_np.min()), float(sdf_np.max())
        if not (sdf_min < 0.0 < sdf_max):
            # 평균을 빼서 대략 0을 중앙으로 이동
            sdf_np = sdf_np - (sdf_min + sdf_max) * 0.5
            sdf_min, sdf_max = float(sdf_np.min()), float(sdf_np.max())

        # 여전히 상수 필드(=표면 없음)라면 marching_cubes를 건너뛴다.
        if not (sdf_min < 0.0 < sdf_max):
            print(
                "[ImplicitMesh] marching_cubes skipped: SDF field has no sign change "
                f"(min={sdf_min:.4f}, max={sdf_max:.4f})"
            )
            continue

        try:
            verts, faces, _, _ = marching_cubes(sdf_np, level=0.0)
        except Exception as e:
            print(f"[ImplicitMesh] marching_cubes failed: {e}")
            continue

        # 좌표를 [-1,1] 박스로 스케일 (ImplicitMeshDecoder의 bound 사용)
        bound = decoder.bound
        scale = (2 * bound) / (grid_res - 1)
        verts = verts * scale - bound

        if _HAS_TRIMESH:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False)
            mesh_path = output_dir / f"sample_{idx:03d}.ply"
            mesh.export(mesh_path)
            print(f"  → mesh saved: {mesh_path}")

            # 프롬프트 기반 텍스처 적용 (OBJ + MTL + PNG 생성)
            _apply_prompt_based_texture(mesh, prompt_text, output_dir, idx)

            # 사용자가 마우스로 회전/이동해 볼 수 있도록 단순 뷰어 호출 (옵션)
            _interactive_preview_mesh(mesh)
        else:
            # trimesh 가 없으면 간단한 npz로 저장
            np.savez(output_dir / f"sample_{idx:03d}_mesh.npz", vertices=verts, faces=faces.astype(np.int64))
            print(
                "  → trimesh 미설치로 .ply 대신 vertices/faces를 .npz로 저장했습니다."
                " (pip install trimesh 로 PLY 저장 가능)"
            )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    from train_pipeline import load_config  # type: ignore[attr-defined]

    project_root = Path(__file__).parent
    config_path = project_root / "config" / "model_config.yaml"
    config = load_config(str(config_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ImplicitMesh] Using device: {device}")

    model = _load_doosan_text2cad(config, device)
    prompts = _extract_prompts_from_pdfs(config)

    out_dir = project_root / "checkpoints" / "text2cad_implicit_mesh"
    generate_implicit_meshes_from_text(
        model=model,
        prompts=prompts,
        output_dir=out_dir,
        device=device,
        grid_res=64,
    )

    # implicit 메쉬와 텍스처 생성 후, Astra 유틸을 활용해 카메라가 메쉬 주변을 도는 비디오 생성
    _generate_orbit_video_for_meshes(out_dir, fps=24, num_frames=120)

    print(f"[ImplicitMesh] Done. Outputs are under: {out_dir}")


if __name__ == "__main__":
    main()

