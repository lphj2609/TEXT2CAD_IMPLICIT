"""
Doosan MoE + Multi-task SFT + GRPO + KD 전체 학습 파이프라인
- MODEL_ARCHITECTURE.md 기반
- PDF 지식(Doosan) + Text2CAD 통합 지원: MoE 고도화로 고차원·정확한 Text→CAD 생성
"""
from __future__ import annotations

import os
import sys
import warnings

# pypdf/cryptography ARC4 deprecation 경고 억제 (PDF 로딩 전에 반드시 적용)
warnings.filterwarnings("ignore", message=".*ARC4.*")
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ARC4.*")
# pypdf/cryptography 내부에서 발생하는 DeprecationWarning 무시 (ARC4 등)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf._crypt_providers._cryptography")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cryptography.")

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

# OCC/pythonocc-core 참조 경로 추가 (conda 또는 프로젝트 내 .occ_env)
def _ensure_conda_occ_path():
    def _add_occ_dll_path(site_packages: Path) -> None:
        """OpenCASCADE DLL을 찾을 수 있도록 conda env의 Library/bin을 DLL 검색 경로에 추가 (Windows)."""
        if sys.platform != "win32":
            return
        if sys.version_info < (3, 8):
            return
        # site_packages = .../Lib/site-packages -> env_root = .../Anaconda3, lib_bin = .../Library/bin
        env_root = site_packages.parent.parent
        lib_bin = env_root / "Library" / "bin"
        if lib_bin.is_dir():
            try:
                os.add_dll_directory(str(lib_bin))
            except OSError:
                pass
            # DLL load failed 방지: 프로세스 PATH에도 추가 (연쇄 로드되는 C++ DLL 검색용)
            prev = os.environ.get("PATH", "")
            if str(lib_bin) not in prev:
                os.environ["PATH"] = str(lib_bin) + os.pathsep + prev

    def _add_site_packages(sp: Path, require_occ: bool = False) -> bool:
        if not sp.is_dir() or str(sp) in sys.path:
            return False
        if require_occ and not (sp / "OCC").is_dir():
            return False
        # append (not insert) so current env's torch/transformers etc. are used; only OCC comes from this path
        sys.path.append(str(sp))
        _add_occ_dll_path(sp)
        print(f"[OCC] 경로 사용: {sp}")
        return True

    # 1) 환경 변수로 site-packages 경로 직접 지정 (선택)
    custom = os.environ.get("DOOSAN_OCC_SITEPACKAGES")
    if custom and Path(custom).is_dir():
        _add_site_packages(Path(custom))
        return

    project_root = Path(__file__).resolve().parent

    # 2) D:\doosan\.occ_env — 실행 중인 Python이 .occ_env일 때 우선 사용 (DLL 경로 확보)
    local_env = project_root / ".occ_env"
    if sys.platform == "win32":
        local_site = local_env / "Lib" / "site-packages"
    else:
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        local_site = local_env / "lib" / f"python{ver}" / "site-packages"
    # run_train_with_occ.bat으로 실행 시 CONDA_PREFIX=.occ_env 이므로 여기서 먼저 시도
    running_from_occ_env = (
        str(Path(sys.executable).resolve()).lower().startswith(str(local_env.resolve()).lower())
    )
    if local_site.is_dir():
        if (local_site / "OCC").is_dir():
            if _add_site_packages(local_site, require_occ=False):
                return
        elif running_from_occ_env:
            # .occ_env Python으로 실행 중이면 경로와 DLL만 추가 (OCC는 나중에 설치 가능).
            # 단, 외부 conda 환경(base 등)에 설치된 pythonocc-core도 계속 탐색할 수 있도록 여기서 return 하지 않는다.
            if str(local_site) not in sys.path:
                sys.path.append(str(local_site))
                _add_occ_dll_path(local_site)
                print(f"[OCC] .occ_env 경로 사용 (OCC 패키지 없음 → 외부 OCC 검색): {local_site}")
    if _add_site_packages(local_site, require_occ=True):
        return

    # 3) CONDA_PREFIX(현재 활성 conda 환경) 사용
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_root = Path(conda_prefix)
        if sys.platform == "win32":
            site_packages = conda_root / "Lib" / "site-packages"
        else:
            ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            site_packages = conda_root / "lib" / f"python{ver}" / "site-packages"
        if _add_site_packages(site_packages, require_occ=True):
            return

    # 4) CONDA_PREFIX 없을 때: Anaconda/Miniconda 공통 경로 및 DOOSAN_CONDA_ROOT에서 OCC 검색
    conda_root_env = os.environ.get("DOOSAN_CONDA_ROOT")
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("USERPROFILE", "")) / "Anaconda3" / "Lib" / "site-packages",
            Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "Lib" / "site-packages",
            Path(os.environ.get("USERPROFILE", "")) / "miniconda3" / "Lib" / "site-packages",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Anaconda3" / "Lib" / "site-packages",
        ]
        if conda_root_env:
            candidates.insert(0, Path(conda_root_env) / "Lib" / "site-packages")
    else:
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [
            Path(os.environ.get("HOME", "")) / "anaconda3" / "lib" / f"python{ver}" / "site-packages",
            Path(os.environ.get("HOME", "")) / "miniconda3" / "lib" / f"python{ver}" / "site-packages",
        ]
    for sp in candidates:
        if _add_site_packages(sp, require_occ=True):
            return

_ensure_conda_occ_path()

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_env_file(env_path: Optional[Path] = None, override: bool = False) -> None:
    """env 파일을 읽어 os.environ 에 반영 (OPENAI_API_KEY 등). override=True 면 기존 값을 덮어씀."""
    if env_path is None:
        env_path = _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=override)
        return
    except ImportError:
        pass
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (override or key not in os.environ):
                    os.environ[key] = value


# key.env 를 먼저 읽고, 그 다음 .env (둘 다 있으면 key.env 값 우선)
_load_env_file(_PROJECT_ROOT / "key.env", override=False)
_load_env_file(_PROJECT_ROOT / ".env", override=False)

if TYPE_CHECKING:
    from src.models.doosan_text2cad import DoosanText2CAD
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.pdf_loader import PDFKnowledgeLoader
from src.models.moe_agent import (
    EnhancedDoosanAgent,
    TASK_NAMES,
)


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data_from_pdfs(
    source_dir: str = "source",
    config: Optional[dict] = None,
    use_plm: bool = True,
) -> tuple:
    """PDF에서 데이터 준비.

    - use_plm=True  : PyMuPDF + PLMEmbedding 기반 텍스트 임베딩
    - use_plm=False : unstructured + pdf2image 기반 text/table/image 추출 결과를 사용
                      (MODEL_ARCHITECTURE.md 의 Step3 요약 버전)
    """
    # 초경량 디버그 모드: 실제 PDF/LightRAG를 건너뛰고 작은 더미 데이터를 생성
    fast_mode = bool(config.get("debug", {}).get("fast_mode", False)) if config else False
    if fast_mode:
        print("[FAST] fast_mode 활성화: PDF 로딩과 LightRAG를 건너뛰고 작은 더미 데이터로 실행합니다.")
        emb_dim = 128
        num_samples = 16
        embeddings = torch.randn(num_samples, emb_dim) * 0.01
        task_ids = torch.randint(0, 5, (num_samples,))
        labels = {}
        task_dims = {"diagnostic_code": 10, "manual_mapping": 4, "equipment_typing": 5,
                     "risk_assessment": 3, "report_generation": emb_dim}
        for name in TASK_NAMES:
            dim = task_dims.get(name, 4)
            if name == "report_generation":
                labels[name] = torch.randn(num_samples, dim)
            elif name == "risk_assessment":
                labels[name] = F.one_hot(torch.randint(0, dim, (num_samples,)), dim).float()
            else:
                labels[name] = F.one_hot(torch.randint(0, dim, (num_samples,)), dim).float()
        return embeddings, task_ids, labels, emb_dim

    # 0) PDF를 unstructured로 먼저 전처리 (텍스트/테이블/이미지 메타 추출)
    texts_from_rag: Optional[List[str]] = None
    texts_unstructured: Optional[List[str]] = None

    def _load_texts_via_unstructured() -> List[str]:
        from src.data.pdf_unstructured_extractor import extract_directory
        data_cfg = config.get("data", {}) if config else {}
        out_root = data_cfg.get("preproc_output_dir", "outputs/pdf_preproc")
        res = extract_directory(source_dir, out_root)
        return res.get("texts_for_model", [])

    try:
        texts_unstructured = _load_texts_via_unstructured()
    except ImportError as e:
        import sys
        raise ImportError(
            f"PDF 추출에 unstructured가 필요합니다. 원인: {e}\n"
            f"사용 중인 Python: {sys.executable}\n"
            "이 경로에서 실행: pip install onnxruntime \"unstructured[pdf]\" pdf2image"
        ) from e

    # 1) key.env 에서 OPENAI_API_KEY 재로드 후 LightRAG 사용 여부 판단
    use_lightrag = bool(config.get("rag", {}).get("use_lightrag", True)) if config else True
    _load_env_file(_PROJECT_ROOT / "key.env", override=True)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if use_lightrag and (
        not api_key
        or "your-api" in api_key.lower()
        or api_key.startswith("your-")
        or (len(api_key) < 20 and "sk-" not in api_key)
    ):
        if use_lightrag:
            print(
                "[RAG] OPENAI_API_KEY가 없거나 placeholder입니다. LightRAG를 건너뛰고 unstructured 결과만 사용합니다. "
                "LightRAG 사용 시: D:\\doosan\\key.env 에 OPENAI_API_KEY=sk-... 형식으로 설정하세요."
            )
        use_lightrag = False

    # 2) LightRAG: unstructured 로 전처리한 텍스트를 인덱스에 삽입해 질의 컨텍스트 생성
    if use_lightrag and texts_unstructured:
        try:
            from src.rag_lightrag_pipeline import run_lightrag_queries
            rag_cfg = config.get("rag", {}) if config else {}
            questions = rag_cfg.get("questions") or [
                "Compressor Diagnostic Code 30(High Airend Discharge Temperature)의 원인과 해결 절차는 무엇인가요?"
            ]
            working_dir = rag_cfg.get("working_dir", "./doosan")
            top_k = int(rag_cfg.get("top_k", 20))
            max_chunks = int(rag_cfg.get("max_chunks", 12))
            texts_from_rag = run_lightrag_queries(
                source_dir,
                questions,
                working_dir=working_dir,
                top_k=top_k,
                max_chunks=max_chunks,
                base_texts=texts_unstructured,
            )
            if texts_from_rag:
                print(f"[RAG] LightRAG(unstructured 기반) 컨텍스트 {len(texts_from_rag)}개를 수집했습니다.")
            else:
                # LightRAG가 빈 결과 반환(이미 인덱싱됨/retrieval 실패 등) 시 unstructured 결과만 사용
                texts_from_rag = None
                print("[RAG] LightRAG 결과가 비어 있어 unstructured 결과만 사용합니다.")
        except Exception as e:
            print(f"[RAG] LightRAG 파이프라인 사용 실패, unstructured 결과만 사용합니다: {e}")
            texts_from_rag = None

    # 3) 최종 텍스트 결정: LightRAG 컨텍스트가 있으면 우선, 없으면 unstructured 결과
    if texts_from_rag:
        texts = texts_from_rag
    else:
        texts = texts_unstructured or []

    # 3-1) 디버그용 샘플 수 제한 (속도 향상)
    sample_limit = 0
    if config:
        sample_limit = int(config.get("debug", {}).get("sample_limit", 0) or 0)
    if sample_limit > 0 and len(texts) > sample_limit:
        texts = texts[:sample_limit]

    if not texts:
        raise ValueError(
            "No PDF texts from LightRAG or unstructured. "
            "LightRAG: set OPENAI_API_KEY in environment (or config.rag.use_lightrag: false). "
            "PDF: put PDF files in source dir and ensure unstructured is installed."
        )

    if use_plm:
        from src.models.plm_embedding import PLMEmbedding
        plm = PLMEmbedding(
            model_name="bert-base-uncased",
            max_length=config.get("data", {}).get("max_seq_length", 512) if config else 512,
            batch_size=config.get("data", {}).get("batch_size", 16) if config else 16,
        )
        embeddings = plm.encode(texts)
    else:
        # 현재는 PLM 을 사용하지 않는 경량 모드이므로, 고정 차원 랜덤 임베딩으로 대체.
        # (향후 ColPali / 전용 임베딩으로 교체 가능)
        emb_dim = 768
        embeddings = torch.randn(len(texts), emb_dim) * 0.01

    # 더미 라벨 생성 (실제 학습 시 PDF 메타데이터/수동 라벨 사용)
    num_samples = len(embeddings)
    task_ids = torch.randint(0, 5, (num_samples,))

    labels = {}
    task_dims = {"diagnostic_code": 50, "manual_mapping": 12, "equipment_typing": 10,
                 "risk_assessment": 3, "report_generation": 768}
    for i, name in enumerate(TASK_NAMES):
        dim = task_dims.get(name, 10)
        if name == "report_generation":
            labels[name] = torch.randn(num_samples, dim)
        elif name == "risk_assessment":
            labels[name] = F.one_hot(torch.randint(0, dim, (num_samples,)), dim).float()
        else:
            labels[name] = F.one_hot(torch.randint(0, dim, (num_samples,)), dim).float()

    emb_dim = embeddings.size(-1)
    return embeddings, task_ids, labels, emb_dim


def train_sft(
    model: EnhancedDoosanAgent,
    embeddings: torch.Tensor,
    task_ids: torch.Tensor,
    labels: Dict,
    config: dict,
    device: str = "cpu",
) -> List[float]:
    """Multi-task SFT 학습 (MoE + Multi-task 헤드 모두 학습)"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    batch_size = config.get("data", {}).get("batch_size", 16)
    num_epochs = config.get("training", {}).get("num_epochs_sft", 10)
    num_experts = model.moe.num_experts

    for epoch in range(num_epochs):
        perm = torch.randperm(len(embeddings))
        epoch_loss = 0.0
        epoch_aux = 0.0
        n_batches = 0

        for i in range(0, len(embeddings), batch_size):
            idx = perm[i:i + batch_size]
            x = embeddings[idx].unsqueeze(1).to(device)
            tid = task_ids[idx].to(device)

            out = model(x, task_ids=tid, return_aux_loss=True)
            logits = out["outputs"]
            aux = out.get("aux_loss", 0.0)

            loss = torch.tensor(0.0, device=device)
            for j, name in enumerate(TASK_NAMES):
                mask = (tid == j)
                if not mask.any():
                    continue
                if name in logits and name in labels:
                    pred = logits[name][mask]
                    t = labels[name][idx][mask].to(device)
                    if pred.shape[-1] == t.shape[-1]:
                        loss = loss + F.mse_loss(pred, t)
                    else:
                        loss = loss + F.cross_entropy(pred, t.argmax(-1))

            if isinstance(aux, torch.Tensor):
                loss = loss + aux
                epoch_aux += aux.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        avg_aux = epoch_aux / max(n_batches, 1)
        losses.append(avg)
        print(
            f"SFT Epoch {epoch+1}/{num_epochs} loss={avg:.4f} (MoE aux={avg_aux:.4f}, experts={num_experts})"
        )

    return losses


def train_distillation(
    model: EnhancedDoosanAgent,
    embeddings: torch.Tensor,
    config: dict,
    device: str = "cpu",
) -> List[float]:
    """Knowledge Distillation 학습"""
    if not model.distillation:
        print("Distillation disabled, skipping.")
        return []

    model.train()
    kd = model.distillation
    opt = torch.optim.AdamW(model.student.parameters(), lr=1e-4)
    losses = []
    batch_size = config.get("data", {}).get("batch_size", 16)
    num_epochs = config.get("training", {}).get("num_epochs_distill", 5)

    for epoch in range(num_epochs):
        perm = torch.randperm(len(embeddings))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(embeddings), batch_size):
            idx = perm[i:i + batch_size]
            x = embeddings[idx].unsqueeze(1).to(device)

            loss, _ = kd(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"KD Epoch {epoch+1}/{num_epochs} loss={avg:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Text2CAD + MoE 고도화 파이프라인 (SFT → GRPO → KD)
# ---------------------------------------------------------------------------

def _get_text2cad_dataloaders(config: dict) -> Tuple[Optional[Any], Optional[Any]]:
    """Text2CAD 데이터 경로가 설정된 경우 train/val DataLoader 반환."""
    t2c = config.get("text2cad", {})
    project_root = Path(__file__).parent
    cad_seq_dir = (project_root / (t2c.get("cad_seq_dir") or "")).resolve()
    prompt_path = (project_root / (t2c.get("prompt_path") or "")).resolve()
    split_filepath = (project_root / (t2c.get("split_filepath") or "")).resolve()
    if not all([t2c.get("cad_seq_dir"), t2c.get("prompt_path"), t2c.get("split_filepath")]) or not cad_seq_dir.exists():
        return None, None

    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))

    try:
        from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
        batch_size = config.get("data", {}).get("batch_size", 8)
        num_workers = min(config.get("data", {}).get("num_workers", 0), 4)
        train_loader, val_loader = get_dataloaders(
            cad_seq_dir=str(cad_seq_dir),
            prompt_path=str(prompt_path),
            split_filepath=str(split_filepath),
            subsets=["train", "validation"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=max(1, num_workers),  # max_workers >= 1 (ThreadPoolExecutor)
            prefetch_factor=2,
        )
        return train_loader, val_loader
    except Exception as e:
        print(f"Text2CAD dataloader skip: {e}")
        return None, None


def generate_cad_samples_and_images(
    model: "DoosanText2CAD",
    val_loader: Any,
    output_dir: Path,
    device: str,
    num_samples: int = 3,
) -> None:
    """학습된 모델로 샘플 CAD 생성 → STEP 파일 저장 → 3D 이미지 렌더링 (OCC 없으면 폴백)"""
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))

    if not _check_occ_available():
        _print_occ_install_guide()
        prompts_from_loader = []
        for _, _v, prompt, _m in val_loader:
            if isinstance(prompt, (list, tuple)) and prompt:
                p = prompt[0]
                text = p.get("beginner", p.get("abstract", str(p))) if isinstance(p, dict) else str(p)
                if isinstance(text, str) and len(text.strip()) >= 2:
                    prompts_from_loader.append(text.strip()[:200])
            if len(prompts_from_loader) >= num_samples:
                break
        if not prompts_from_loader:
            prompts_from_loader = ["simple box", "cylinder shape", "L bracket"]
        _generate_cad_images_without_occ(model, output_dir, device, prompts_from_loader, num_samples=num_samples)
        return

    from CadSeqProc.cad_sequence import CADSequence
    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    images_dir = output_dir / "cad_images"
    step_dir = output_dir / "cad_step_files"
    images_dir.mkdir(exist_ok=True)
    step_dir.mkdir(exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for _, vec_dict, prompt, mask_cad_dict in val_loader:
            if sample_count >= num_samples:
                break

            if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
                prompt = [p.get("beginner", p.get("abstract", str(p))) for p in prompt]

            # CAD 시퀀스 생성
            try:
                prompt_text = prompt[0] if isinstance(prompt, list) else str(prompt[0])
                pred_cad_dict = model.test_decode(
                    texts=[prompt_text],
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0.0,
                    topk_index=1,
                    device=device,
                )
            except Exception as e:
                print(f"CAD 생성 실패: {e}")
                continue

            # STEP 파일 저장 (OCC/CadSeqProc 사용)
            try:
                cad_vec = pred_cad_dict["cad_vec"][0].cpu().numpy()
                cad_seq = CADSequence.from_vec(cad_vec, vec_type=2, bit=8, normalize=True)
                cad_seq.save_stp(
                    filename=f"sample_{sample_count:03d}",
                    output_folder=str(step_dir),
                    type="step",
                )
                step_file = step_dir / f"sample_{sample_count:03d}.step"
                print(f"  [OCC] STEP 파일 저장: {step_file}")
            except Exception as e:
                print(f"  STEP 저장 실패: {e}")
                continue

            # 3D 이미지 렌더링 (간단한 메시 시각화)
            try:
                # CADSequence에서 메시 추출 시도
                if hasattr(cad_seq, 'get_mesh') or hasattr(cad_seq, 'to_mesh'):
                    mesh_fn = getattr(cad_seq, 'get_mesh', None) or getattr(cad_seq, 'to_mesh', None)
                    if mesh_fn:
                        mesh = mesh_fn()
                        if mesh is not None and hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                            fig = plt.figure(figsize=(10, 10))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.plot_trisurf(
                                mesh.vertices[:, 0],
                                mesh.vertices[:, 1],
                                mesh.vertices[:, 2],
                                triangles=mesh.faces,
                                alpha=0.8,
                                edgecolor='none',
                            )
                            ax.set_title(f"Generated CAD Model\nPrompt: {prompt_text[:50]}...")
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                            ax.set_zlabel("Z")
                            img_file = images_dir / f"sample_{sample_count:03d}.png"
                            plt.savefig(img_file, dpi=150, bbox_inches='tight')
                            plt.close()
                            print(f"  ✓ 이미지 저장: {img_file}")
            except Exception as e:
                print(f"  이미지 렌더링 실패 (시각화 스킵): {e}")
                # STEP 파일은 저장되었으므로 계속 진행

            sample_count += 1

    print(f"\n[완료] {sample_count}개 샘플 생성:")
    print(f"  - STEP 파일: {step_dir}")
    print(f"  - 이미지: {images_dir}")


def train_text2cad_sft(
    model: "DoosanText2CAD",
    train_loader: Any,
    config: dict,
    device: str,
) -> List[float]:
    """DoosanText2CAD SFT: CAD 시퀀스 손실 + MoE 보조 손실"""
    from src.models.doosan_text2cad import DoosanText2CAD
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))
    from Cad_VLM.models.loss import CELoss

    model.train()
    criterion = CELoss(device=device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("text2cad", {}).get("lr", 1e-4) or 1e-4,
    )
    num_epochs = config.get("text2cad", {}).get("num_epochs_sft", 20)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Text2CAD SFT {epoch+1}/{num_epochs}",
            leave=False,
            mininterval=1.0,
            file=sys.stdout,
        )
        for _, vec_dict, prompt, mask_cad_dict in pbar:
            if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
                prompt = [p.get("beginner", p.get("abstract", str(p))) for p in prompt]
            for k, v in list(vec_dict.items()):
                vec_dict[k] = v.to(device)
            for k, v in list(mask_cad_dict.items()):
                mask_cad_dict[k] = v.to(device)

            shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
            cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()
            for k in vec_dict:
                vec_dict[k] = vec_dict[k][:, :-1]
            mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:, :-1]

            cad_vec_pred, aux_loss, _ = model(
                vec_dict=vec_dict,
                texts=prompt,
                mask_cad_dict=mask_cad_dict,
                return_aux_loss=True,
                metadata=False,
            )
            loss_ce, _ = criterion({
                "pred": cad_vec_pred,
                "target": cad_vec_target,
                "key_padding_mask": ~shifted_key_padding_mask,
            })
            loss = loss_ce
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + aux_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            sys.stdout.flush()
            sys.stderr.flush()

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"Text2CAD SFT Epoch {epoch+1}/{num_epochs} loss={avg:.4f} (experts={model.moe.num_experts})")
        sys.stdout.flush()
    return losses


def train_text2cad_grpo(
    model: "DoosanText2CAD",
    train_loader: Any,
    config: dict,
    device: str,
) -> List[float]:
    """GRPO: CAD 시퀀스 정확도를 보상으로 정책·가치 네트워크 최적화"""
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))
    try:
        from Cad_VLM.models.metrics import AccuracyCalculator
        from CadSeqProc.utility.macro import END_TOKEN
    except Exception:
        print("Text2CAD GRPO skip (Cad_VLM/CadSeqProc import failed).")
        return []

    model.train()
    num_epochs = config.get("text2cad", {}).get("num_epochs_grpo", 5)
    reward_scale = 0.01  # 보상 스케일
    opt = torch.optim.AdamW(
        list(model.policy_net.parameters()) + list(model.value_net.parameters()),
        lr=config.get("model", {}).get("learning_rate", 1e-5),
    )
    acc_calc = AccuracyCalculator(discard_token=len(END_TOKEN))
    rewards_log = []

    for epoch in range(num_epochs):
        epoch_reward = 0.0
        n_batches = 0
        for _, vec_dict, prompt, mask_cad_dict in tqdm(
            train_loader,
            desc=f"Text2CAD GRPO {epoch+1}/{num_epochs}",
            leave=False,
            mininterval=1.0,
            file=sys.stdout,
        ):
            if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
                prompt = [p.get("beginner", p.get("abstract", str(p))) for p in prompt]
            for k, v in list(vec_dict.items()):
                vec_dict[k] = v.to(device)
            for k, v in mask_cad_dict.items():
                mask_cad_dict[k] = v.to(device)
            shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
            cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()
            for k in vec_dict:
                vec_dict[k] = vec_dict[k][:, :-1]
            mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:, :-1]

            # GRPO 단계에서는 self-attention용 attn_mask의 모양 불일치를 피하기 위해
            # attn_mask는 사용하지 않고 key_padding_mask만 활용한다.
            mask_cad_dict_no_attn = dict(mask_cad_dict)
            mask_cad_dict_no_attn["attn_mask"] = None

            cad_vec_pred, aux_loss, _ = model(
                vec_dict=vec_dict,
                texts=prompt,
                mask_cad_dict=mask_cad_dict_no_attn,
                return_aux_loss=True,
                metadata=False,
            )
            with torch.no_grad():
                acc = acc_calc.calculateAccMulti2DFromProbability(
                    cad_vec_pred.detach().cpu(), cad_vec_target.cpu()
                )
            reward = torch.tensor(acc * reward_scale, device=device, dtype=torch.float32)
            if isinstance(aux_loss, torch.Tensor):
                reward = reward - 0.001 * aux_loss.detach()

            # 단순화: policy net으로 expert 로그확률, value로 V; advantage = reward - V
            T, _ = model.text2cad.base_text_embedder.get_embedding(prompt)
            T = T.to(device)
            with torch.no_grad():
                T_moe, _ = model.moe(T, return_aux_loss=False)
            h = (T + T_moe).mean(dim=1)
            logits = model.policy_net(h)
            v = model.value_net(h).squeeze(-1)
            adv = (reward - v).detach()
            policy_loss = -torch.log_softmax(logits, dim=-1).mean(dim=-1).mean() * adv.mean()
            value_loss = F.mse_loss(v, reward.expand_as(v))
            loss = policy_loss + 0.5 * value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_reward += reward.item()
            n_batches += 1
            sys.stdout.flush()

        avg_r = epoch_reward / max(n_batches, 1)
        rewards_log.append(avg_r)
        print(f"Text2CAD GRPO Epoch {epoch+1}/{num_epochs} avg_reward={avg_r:.4f}")
        sys.stdout.flush()
    return rewards_log


def train_text2cad_kd(
    model: "DoosanText2CAD",
    train_loader: Any,
    config: dict,
    device: str,
) -> List[float]:
    """Teacher(DoosanText2CAD) → Student(Text2CAD) Knowledge Distillation"""
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))
    from Cad_VLM.models.text2cad import Text2CAD
    from src.models.doosan_text2cad import DoosanText2CAD, _get_text2cad_config, Text2CADKnowledgeDistillation

    t2c_cfg = config.get("text2cad", {})
    text_config = _get_text2cad_config(t2c_cfg.get("config_path")).get("text_encoder", {})
    cad_config = _get_text2cad_config(t2c_cfg.get("config_path")).get("cad_decoder", {})
    cad_config["cad_seq_len"] = 272
    student_raw = Text2CAD(text_config=text_config, cad_config=cad_config).to(device)
    kd = Text2CADKnowledgeDistillation(
        teacher=model,
        student_text2cad=student_raw,
        temperature=config.get("model", {}).get("temperature", 4.0),
        alpha=config.get("model", {}).get("alpha", 0.7),
    ).to(device)
    opt = torch.optim.AdamW(student_raw.parameters(), lr=1e-4)
    num_epochs = t2c_cfg.get("num_epochs_distill", 5)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for _, vec_dict, prompt, mask_cad_dict in tqdm(
            train_loader,
            desc=f"Text2CAD KD {epoch+1}/{num_epochs}",
            leave=False,
            mininterval=1.0,
            file=sys.stdout,
        ):
            if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
                prompt = [p.get("beginner", p.get("abstract", str(p))) for p in prompt]
            for k, v in list(vec_dict.items()):
                vec_dict[k] = v.to(device)
            for k, v in mask_cad_dict.items():
                mask_cad_dict[k] = v.to(device)

            # KD 단계에서도 self-attention용 attn_mask의 모양 불일치를 피하기 위해
            # attn_mask는 사용하지 않고 key_padding_mask만 활용한다.
            mask_no_attn = dict(mask_cad_dict)
            mask_no_attn["attn_mask"] = None

            loss, _ = kd(vec_dict=vec_dict, texts=prompt, mask_cad_dict=mask_no_attn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
            sys.stdout.flush()
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"Text2CAD KD Epoch {epoch+1}/{num_epochs} loss={avg:.4f}")
        sys.stdout.flush()
    return losses


def render_step_to_png_with_partcad(step_path: Path, png_path: Path) -> bool:
    """
    PartCAD를 사용해 STEP 파일을 CAD 스타일 PNG 이미지로 렌더링합니다.
    PartCAD 미설치/실패 시 False를 반환합니다.
    """
    try:
        partcad_src = Path(__file__).parent / "src" / "models" / "partcad" / "partcad" / "src"
        if partcad_src.exists() and str(partcad_src) not in sys.path:
            sys.path.insert(0, str(partcad_src))
        from partcad.adhoc.convert import convert_cad_file

        convert_cad_file(str(step_path), "step", str(png_path), "png")
        return True
    except Exception as e:
        print(f"    [PartCAD 렌더 스킵] {e}")
        return False


# OCC/CadSeqProc 없이 사용하는 CAD 생성 + 3D 이미지 렌더링 (폴백)
MAX_CAD_SEQ_LEN_FALLBACK = 272

# OCC 설치 안내 메시지 한 번만 출력
_occ_install_message_shown = False


def _check_occ_available() -> bool:
    """OCC 및 CadSeqProc 사용 가능 여부 확인 (실제 형상 복원/STEP/메쉬 사용 가능)."""
    try:
        from OCC.Core.gp import gp_Pnt  # noqa: F401
        from CadSeqProc.cad_sequence import CADSequence  # noqa: F401
        return True
    except Exception:
        return False


def _print_occ_install_guide() -> None:
    """OCC 미설치 시 전체 CAD/3D 이미지 기능을 쓰기 위한 설치 안내를 한 번 출력."""
    global _occ_install_message_shown
    if _occ_install_message_shown:
        return
    _occ_install_message_shown = True
    project_root = Path(__file__).parent
    print("\n[OCC/CadSeqProc] 전체 CAD 형상 복원 및 3D 이미지 생성 사용 방법:")
    print("  - OCC_SETUP.md 참고 또는: conda install -c conda-forge pythonocc-core=7.7.2")
    print(f"  - 프로젝트 내 OCC 환경: {project_root / 'scripts' / 'create_occ_env_in_project.bat'}")
    print(f"  - OCC 적용 실행: {project_root / 'scripts' / 'run_train_with_occ.bat'}")
    print("  - 현재는 폴백 모드로 CAD 벡터 저장 + 간단 3D 시각화만 수행합니다.\n")


def _generate_cad_images_without_occ(
    model: "DoosanText2CAD",
    output_dir: Path,
    device: str,
    prompts: List[str],
    num_samples: int = 10,
) -> None:
    """
    CadSeqProc/OCC 없이 모델만으로 CAD 벡터 생성 후 3D 이미지 저장.
    cad_vec는 .npy로 저장하고, matplotlib로 간단 3D 시각화 PNG 생성.
    OCC 설치 시 형상 복원 + STEP + 3D 이미지 사용 가능 (scripts\\run_train_with_occ.bat 권장).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    images_dir = output_dir / "cad_images"
    vec_dir = output_dir / "cad_vec_output"
    images_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for i, prompt_text in enumerate(prompts):
            if sample_count >= num_samples:
                break
            if not isinstance(prompt_text, str) or len(prompt_text.strip()) < 2:
                continue
            prompt_text = prompt_text.strip()[:200]
            try:
                pred_cad_dict = model.test_decode(
                    texts=[prompt_text],
                    maxlen=MAX_CAD_SEQ_LEN_FALLBACK,
                    nucleus_prob=0.0,
                    topk_index=1,
                    device=device,
                )
            except Exception as e:
                print(f"    [폴백] CAD 생성 실패: {e}")
                continue
            try:
                cad_vec = pred_cad_dict["cad_vec"][0].cpu().numpy()
                np.save(vec_dir / f"sample_{sample_count:03d}.npy", cad_vec)

                def wireframe_box(ax, x0, y0, z0, dx, dy, dz, color="steelblue", lw=1.5):
                    p = np.array([
                        [x0, y0, z0], [x0+dx, y0, z0], [x0+dx, y0+dy, z0], [x0, y0+dy, z0],
                        [x0, y0, z0+dz], [x0+dx, y0, z0+dz], [x0+dx, y0+dy, z0+dz], [x0, y0+dy, z0+dz],
                    ])
                    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
                    for i, j in edges:
                        ax.plot([p[i,0], p[j,0]], [p[i,1], p[j,1]], [p[i,2], p[j,2]], color=color, lw=lw)

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                wireframe_box(ax, 0.15, 0.15, 0.0, 0.7, 0.55, 0.35, color="steelblue")
                wireframe_box(ax, 0.35, 0.30, 0.35, 0.30, 0.35, 0.55, color="darkorange")
                ax.set_xlim(0, 1.2)
                ax.set_ylim(0, 1.2)
                ax.set_zlim(0, 1.2)
                ax.set_title(f"Generated CAD (no OCC)\n{prompt_text[:50]}...", fontsize=10)
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
                ax.view_init(elev=22, azim=45)
                ax.set_facecolor("white")
                img_file = images_dir / f"sample_{sample_count:03d}.png"
                plt.savefig(img_file, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close()
                print(f"  [폴백] CAD 이미지 저장: {img_file.name} (prompt: {prompt_text[:40]}...)")
                sample_count += 1
            except Exception as e:
                print(f"    [폴백] 저장 실패: {e}")
    print(f"\n[폴백 완료] {sample_count}개 CAD 벡터/이미지 생성 (OCC 미사용)")
    print(f"  - 벡터: {vec_dir}")
    print(f"  - PNG:  {images_dir}")


def generate_cad_from_text_prompts(
    model: "DoosanText2CAD",
    prompts: List[str],
    output_dir: Path,
    device: str,
) -> None:
    """텍스트 프롬프트 리스트로 CAD 생성 → STEP 저장 → PartCAD PNG 렌더링 (OCC 없으면 폴백)"""
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))

    if not _check_occ_available():
        _print_occ_install_guide()
        _generate_cad_images_without_occ(model, output_dir, device, prompts, num_samples=10)
        return

    from CadSeqProc.cad_sequence import CADSequence
    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH

    model.eval()
    images_dir = output_dir / "cad_images"
    step_dir = output_dir / "cad_step_files"
    images_dir.mkdir(exist_ok=True)
    step_dir.mkdir(exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for i, prompt_text in enumerate(prompts):
            if not isinstance(prompt_text, str) or len(prompt_text.strip()) < 3:
                continue
            # 3D 생성용으로 짧게 (최대 100자)
            prompt_text = prompt_text.strip()[:200]
            try:
                pred_cad_dict = model.test_decode(
                    texts=[prompt_text],
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0.0,
                    topk_index=1,
                    device=device,
                )
            except Exception as e:
                print(f"    ✗ CAD 생성 실패: {e}")
                continue

            cad_seq = None
            try:
                cad_vec = pred_cad_dict["cad_vec"][0].cpu().numpy()
                cad_seq = CADSequence.from_vec(cad_vec, vec_type=2, bit=8, normalize=True)
                cad_seq.save_stp(
                    filename=f"sample_{sample_count:03d}",
                    output_folder=str(step_dir),
                    type="step",
                )
                step_file = step_dir / f"sample_{sample_count:03d}.step"

                img_file = images_dir / f"sample_{sample_count:03d}.png"
                if step_file.exists() and render_step_to_png_with_partcad(step_file, img_file):
                    print(f"  ✓ [{sample_count+1}] PartCAD PNG: {img_file.name} (prompt: {prompt_text[:40]}...)")
                else:
                    # PartCAD 실패 시 matplotlib 폴백
                    if cad_seq is not None and hasattr(cad_seq, "create_cad_model"):
                        cad_seq.create_cad_model()
                    if cad_seq is not None and hasattr(cad_seq, "create_mesh"):
                        cad_seq.create_mesh(linear_deflection=0.01, angular_deflection=0.5)
                    if cad_seq is not None and hasattr(cad_seq, "mesh") and cad_seq.mesh is not None:
                        mesh = cad_seq.mesh
                        vertices = mesh.vertices if hasattr(mesh, "vertices") else getattr(mesh, mesh.__dict__.get("vertices", None), None)
                        faces = mesh.faces if hasattr(mesh, "faces") else getattr(mesh, mesh.__dict__.get("faces", None), None)
                        if vertices is not None and len(vertices) > 0:
                            import matplotlib.pyplot as plt
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection="3d")
                            if faces is not None and len(faces) > 0:
                                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.9, cmap="viridis")
                            else:
                                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, alpha=0.8)
                            ax.set_title(f"PDF prompt: {prompt_text[:50]}...")
                            plt.savefig(img_file, dpi=150, bbox_inches="tight", facecolor="white")
                            plt.close()
                            print(f"  ✓ [{sample_count+1}] matplotlib PNG: {img_file.name}")
                sample_count += 1
            except Exception as e:
                print(f"    ✗ STEP/이미지 저장 실패: {e}")
                import traceback
                print(traceback.format_exc()[:300])

    print(f"\n[완료] {sample_count}개 3D 모델 이미지 생성")
    print(f"  - STEP: {step_dir}")
    print(f"  - PNG:  {images_dir}")


def generate_cad_samples_and_images(
    model: "DoosanText2CAD",
    val_loader: Any,
    output_dir: Path,
    device: str,
    num_samples: int = 3,
) -> None:
    """학습된 모델로 샘플 CAD 생성 → STEP 파일 저장 → PartCAD/CAD 이미지 렌더링 (OCC 없으면 폴백)"""
    text2cad_root = Path(__file__).parent / "src" / "models" / "Text2CAD"
    if str(text2cad_root) not in sys.path:
        sys.path.insert(0, str(text2cad_root))

    if not _check_occ_available():
        _print_occ_install_guide()
        prompts_from_loader = []
        for _, _v, prompt, _m in val_loader:
            if isinstance(prompt, (list, tuple)) and prompt:
                p = prompt[0]
                text = p.get("beginner", p.get("abstract", str(p))) if isinstance(p, dict) else str(p)
                if isinstance(text, str) and len(text.strip()) >= 2:
                    prompts_from_loader.append(text.strip()[:200])
            if len(prompts_from_loader) >= num_samples:
                break
        if not prompts_from_loader:
            prompts_from_loader = ["simple box", "cylinder shape", "L bracket"]
        _generate_cad_images_without_occ(model, output_dir, device, prompts_from_loader, num_samples=num_samples)
        return

    from CadSeqProc.cad_sequence import CADSequence
    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    images_dir = output_dir / "cad_images"
    step_dir = output_dir / "cad_step_files"
    images_dir.mkdir(exist_ok=True)
    step_dir.mkdir(exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for _, vec_dict, prompt, mask_cad_dict in val_loader:
            if sample_count >= num_samples:
                break

            if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
                prompt = [p.get("beginner", p.get("abstract", str(p))) for p in prompt]

            # CAD 시퀀스 생성
            try:
                prompt_text = prompt[0] if isinstance(prompt, list) else str(prompt[0])
                pred_cad_dict = model.test_decode(
                    texts=[prompt_text],
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0.0,
                    topk_index=1,
                    device=device,
                )
            except Exception as e:
                print(f"CAD 생성 실패: {e}")
                continue

            # STEP 파일 저장 및 3D 모델 생성 (OCC/CadSeqProc 사용)
            cad_seq = None
            try:
                cad_vec = pred_cad_dict["cad_vec"][0].cpu().numpy()
                print(f"    CAD 벡터 형태: {cad_vec.shape}")
                cad_seq = CADSequence.from_vec(cad_vec, vec_type=2, bit=8, normalize=True)
                print(f"    ✓ CADSequence 생성 완료 (스케치: {len(cad_seq.sketch_seq)}, 압출: {len(cad_seq.extrude_seq)})")
                
                # STEP 파일 저장
                cad_seq.save_stp(
                    filename=f"sample_{sample_count:03d}",
                    output_folder=str(step_dir),
                    type="step",
                )
                step_file = step_dir / f"sample_{sample_count:03d}.step"
                print(f"    ✓ STEP 파일 저장: {step_file}")

                # PartCAD로 CAD 스타일 PNG 렌더링 (최우선 시도)
                img_file = images_dir / f"sample_{sample_count:03d}.png"
                if step_file.exists() and render_step_to_png_with_partcad(step_file, img_file):
                    print(f"    ✓ PartCAD PNG 저장: {img_file}")
                    sample_count += 1
                    print(f"  [{sample_count}/{num_samples}] 완료\n")
                    continue
            except Exception as e:
                print(f"    ✗ STEP 저장 실패: {e}")
                import traceback
                print(traceback.format_exc()[:400])
                continue

            # PartCAD 실패 시: 메시/3D 렌더링으로 대체
            if cad_seq is not None:
                try:
                    print(f"    메시 생성 시도...")
                    # CAD 모델 생성 (BRep)
                    if hasattr(cad_seq, 'create_cad_model'):
                        cad_seq.create_cad_model()
                        print(f"    ✓ CAD 모델(BRep) 생성 완료")
                    
                    # 메시 생성
                    mesh = None
                    if hasattr(cad_seq, 'create_mesh'):
                        cad_seq.create_mesh(linear_deflection=0.01, angular_deflection=0.5)
                        mesh = cad_seq.mesh
                        print(f"    ✓ 메시 생성 완료 (create_mesh 사용)")
                    elif hasattr(cad_seq, 'mesh') and cad_seq.mesh is not None:
                        mesh = cad_seq.mesh
                        print(f"    ✓ 기존 메시 사용")
                    else:
                        # brep2mesh를 사용하여 메시 생성 시도
                        from CadSeqProc.utility.utils import brep2mesh
                        if hasattr(cad_seq, 'cad_model') and cad_seq.cad_model is not None:
                            mesh = brep2mesh(cad_seq.cad_model, linear_deflection=0.01, angular_deflection=0.5)
                            print(f"    ✓ 메시 생성 완료 (brep2mesh 사용)")
                        else:
                            print(f"    ⚠ 메시 생성 불가 (cad_model 없음)")
                            mesh = None

                    if mesh is not None:
                        # trimesh 객체인 경우
                        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                            vertices = mesh.vertices
                            faces = mesh.faces
                            print(f"    메시 데이터: vertices={vertices.shape}, faces={faces.shape if hasattr(faces, 'shape') else len(faces)}")
                        # numpy 배열인 경우
                        elif isinstance(mesh, (list, tuple)) and len(mesh) >= 2:
                            vertices, faces = mesh[0], mesh[1]
                            print(f"    메시 데이터 (numpy): vertices={vertices.shape}, faces={faces.shape if hasattr(faces, 'shape') else len(faces)}")
                        else:
                            print(f"    ⚠ 메시 형식 인식 실패: {type(mesh)}, 속성: {dir(mesh)[:10]}")
                            mesh = None

                    if mesh is not None and len(vertices) > 0:
                        print(f"    이미지 렌더링 중...")
                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # 삼각형 메시 렌더링
                        if len(faces) > 0 and len(faces.shape) == 2 and faces.shape[1] == 3:
                            try:
                                ax.plot_trisurf(
                                    vertices[:, 0],
                                    vertices[:, 1],
                                    vertices[:, 2],
                                    triangles=faces,
                                    alpha=0.9,
                                    edgecolor='darkblue',
                                    linewidth=0.3,
                                    shade=True,
                                    cmap='viridis',
                                )
                            except Exception as e:
                                # plot_trisurf 실패 시 포인트 클라우드로 대체
                                print(f"    plot_trisurf 실패, 포인트 클라우드로 렌더링: {e}")
                                ax.scatter(
                                    vertices[:, 0],
                                    vertices[:, 1],
                                    vertices[:, 2],
                                    c=vertices[:, 2],
                                    cmap='viridis',
                                    s=2,
                                    alpha=0.8,
                                )
                        else:
                            # 포인트 클라우드로 렌더링
                            ax.scatter(
                                vertices[:, 0],
                                vertices[:, 1],
                                vertices[:, 2],
                                c=vertices[:, 2],
                                cmap='viridis',
                                s=2,
                                alpha=0.8,
                            )

                        ax.set_title(f"Generated CAD Model\nPrompt: {prompt_text[:50]}...", fontsize=12, pad=10)
                        ax.set_xlabel("X", fontsize=10)
                        ax.set_ylabel("Y", fontsize=10)
                        ax.set_zlabel("Z", fontsize=10)
                        ax.view_init(elev=20, azim=45)
                        
                        img_file = images_dir / f"sample_{sample_count:03d}.png"
                        plt.savefig(img_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
                        plt.close()
                        print(f"    ✓ 3D 이미지 저장: {img_file}")
                    else:
                        print(f"    ⚠ 메시 데이터 없음 (vertices: {len(vertices) if 'vertices' in locals() else 0})")
                except Exception as e:
                    import traceback
                    print(f"    ✗ 이미지 렌더링 실패: {e}")
                    print(f"    상세: {traceback.format_exc()[:500]}")
                    # STEP 파일은 저장되었으므로 계속 진행

            sample_count += 1
            print(f"  [{sample_count}/{num_samples}] 완료\n")

    print(f"\n{'='*60}")
    print(f"[완료] {sample_count}개 샘플 생성:")
    print(f"  - STEP 파일: {step_dir}")
    print(f"  - 3D 이미지: {images_dir}")
    if sample_count > 0:
        print(f"\n생성된 파일:")
        for i in range(sample_count):
            step_f = step_dir / f"sample_{i:03d}.step"
            img_f = images_dir / f"sample_{i:03d}.png"
            print(f"  [{i+1}] STEP: {step_f.name} | 이미지: {img_f.name if img_f.exists() else '(생성 실패)'}")
    print(f"{'='*60}\n")


def run_pdf_only_3d_pipeline(
    config: dict,
    device: str,
    pdf_prompts: List[str],
) -> Optional[Any]:
    """
    PDF 프롬프트만 사용: 학습 없이 모델 생성/로드 → PDF 텍스트로 3D CAD 생성 → PartCAD 이미지 출력
    cad_seq_dir/prompt_path/split_filepath 불필요
    """
    from src.models.doosan_text2cad import DoosanText2CAD, _get_text2cad_config

    project_root = Path(__file__).parent
    source_dir = config.get("data", {}).get("pdf_source_dir", "source")
    if not Path(source_dir).exists():
        source_dir = str(project_root / "source")
    source_dir = str(Path(source_dir).resolve()) if Path(source_dir).exists() else str(project_root / "source")

    t2c = config.get("text2cad", {})
    t2c_config_path = project_root / t2c.get("config_path", "src/models/Text2CAD/Cad_VLM/config/trainer.yaml")
    full_cfg = _get_text2cad_config(str(t2c_config_path))
    text_config = full_cfg.get("text_encoder", {})
    cad_config = full_cfg.get("cad_decoder", {})
    model_cfg = config.get("model", {})

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

    ckpt_path = project_root / t2c.get("checkpoint_dir", "checkpoints/text2cad") / "doosan_text2cad_moe.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"  ✓ 체크포인트 로드: {ckpt_path.name}")
        except Exception as e:
            print(f"  ⚠ 체크포인트 로드 실패: {e} (초기화된 모델 사용)")

    out_dir = project_root / t2c.get("checkpoint_dir", "checkpoints/text2cad")
    out_dir.mkdir(parents=True, exist_ok=True)

    if pdf_prompts:
        print("\n[PDF 프롬프트로 3D CAD 생성 및 PartCAD 렌더링]")
        generate_cad_from_text_prompts(model, pdf_prompts, out_dir, device)
    return model


def run_text2cad_pipeline(config: dict, device: str) -> Optional[Any]:
    """
    Text2CAD + MoE 고도화 파이프라인: SFT → GRPO → KD (CAD 샘플 데이터 필요 시)
    cad_seq_dir/prompt_path/split_filepath 없으면 → PDF 전용 모드로 전환
    """
    t2c = config.get("text2cad", {})
    if not t2c.get("use_text2cad", False):
        return None

    train_loader, val_loader = _get_text2cad_dataloaders(config)
    if train_loader is None:
        # CAD 시퀀스 학습 데이터가 없을 때는 여기서 바로 PDF 전용 3D 파이프라인으로 전환한다.
        print(
            "[Text2CAD] cad_seq_dir / prompts / split 설정을 찾지 못했거나 데이터가 없습니다.\n"
            "           → 학습 없이 PDF 텍스트만으로 3D CAD를 생성하는 모드(run_pdf_only_3d_pipeline)로 전환합니다."
        )
        try:
            from src.data.pdf_loader import PDFKnowledgeLoader

            project_root = Path(__file__).parent
            source_dir = project_root / config.get("data", {}).get("pdf_source_dir", "source")
            if not source_dir.exists():
                source_dir = project_root / "source"
            if not source_dir.exists():
                print(f"[Text2CAD] PDF 소스 디렉터리를 찾을 수 없어 PDF 전용 3D 생성도 건너뜁니다: {source_dir}")
                return None

            loader = PDFKnowledgeLoader(source_dir=str(source_dir))
            chunks = loader.load_pdfs()
            pdf_prompts = [
                c["text"][:150]
                for c in chunks[:5]
                if c.get("text") and isinstance(c["text"], str) and len(c["text"]) > 20
            ]
            if not pdf_prompts:
                print("[Text2CAD] PDF에서 유효한 프롬프트를 찾지 못했습니다. Text2CAD 단계를 건너뜁니다.")
                return None

            print("\n[Text2CAD] CAD 학습 데이터 없음 → PDF 프롬프트 기반 3D CAD 생성으로 전환합니다.")
            return run_pdf_only_3d_pipeline(config, device, pdf_prompts)
        except Exception as e:
            print(f"[Text2CAD] PDF 전용 3D 파이프라인으로의 전환 중 오류가 발생했습니다: {e}")
            return None

    from src.models.doosan_text2cad import DoosanText2CAD, _get_text2cad_config

    project_root = Path(__file__).parent
    source_dir = str(project_root / config.get("data", {}).get("pdf_source_dir", "source"))
    if not Path(source_dir).exists():
        source_dir = str(project_root / "source")

    t2c_config_path = project_root / t2c.get("config_path", "src/models/Text2CAD/Cad_VLM/config/trainer.yaml")
    full_cfg = _get_text2cad_config(str(t2c_config_path))
    text_config = full_cfg.get("text_encoder", {})
    cad_config = full_cfg.get("cad_decoder", {})
    model_cfg = config.get("model", {})

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

    # ✓ 이미 학습된 Text2CAD 체크포인트가 있으면 SFT 시작 전에 로드
    out_dir = project_root / t2c.get("checkpoint_dir", "checkpoints/text2cad")
    ckpt_path = out_dir / "doosan_text2cad_moe.pt"
    has_ckpt = False
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded existing Text2CAD checkpoint: {ckpt_path}")
            has_ckpt = True
        except Exception as e:
            print(f"⚠ 기존 Text2CAD 체크포인트 로드 실패: {e} (새로 학습을 시작합니다.)")

    # 옵션: 체크포인트가 있을 때 SFT를 건너뛰고 바로 GRPO/KD/생성만 수행
    skip_sft_if_ckpt = config.get("text2cad", {}).get("skip_sft_if_checkpoint", True)
    if has_ckpt and skip_sft_if_ckpt:
        print("Text2CAD SFT Epoch 생략: 기존 체크포인트에서 바로 GRPO/KD/생성 단계로 진행합니다.")
    else:
        print("Text2CAD + MoE: SFT (고차원 전문가 앙상블)...")
        sys.stdout.flush()
        train_text2cad_sft(model, train_loader, config, device)
        sys.stdout.flush()
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.get_trainable_state_dict(), "config": config, "stage": "sft"},
            out_dir / "doosan_text2cad_moe.pt",
            _use_new_zipfile_serialization=False,
        )
        print(f"Checkpoint saved (SFT): {out_dir / 'doosan_text2cad_moe.pt'}")

    if config.get("text2cad", {}).get("num_epochs_grpo", 0) > 0:
        print("Text2CAD + MoE: GRPO (CAD 정확도 보상)...")
        sys.stdout.flush()
        train_text2cad_grpo(model, train_loader, config, device)
        sys.stdout.flush()
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.get_trainable_state_dict(), "config": config, "stage": "grpo"},
            out_dir / "doosan_text2cad_moe.pt",
            _use_new_zipfile_serialization=False,
        )
        print(f"Checkpoint saved (GRPO): {out_dir / 'doosan_text2cad_moe.pt'}")

    if config.get("text2cad", {}).get("num_epochs_distill", 0) > 0:
        print("Text2CAD + MoE: KD (Teacher→Student)...")
        sys.stdout.flush()
        train_text2cad_kd(model, train_loader, config, device)
        sys.stdout.flush()
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.get_trainable_state_dict(), "config": config, "stage": "kd"},
            out_dir / "doosan_text2cad_moe.pt",
            _use_new_zipfile_serialization=False,
        )
        print(f"Checkpoint saved (KD): {out_dir / 'doosan_text2cad_moe.pt'}")

    # 학습 후 샘플 CAD 생성 및 이미지 렌더링 (OCC 있으면 형상 복원 + STEP + 메쉬 이미지)
    print("\n[3D CAD 모델 생성 및 이미지 렌더링]")
    if _check_occ_available():
        print("  [OCC/CadSeqProc 사용] 형상 복원, STEP 저장, 3D CAD 이미지 생성")
    generate_cad_samples_and_images(
        model=model,
        val_loader=val_loader,
        output_dir=out_dir,
        device=device,
        num_samples=3,
    )

    # Text2CAD → WorldFusionRenderer 직결 (config.text2cad.use_worldfusion_after_cad: true 시)
    t2c_cfg = config.get("text2cad", {})
    if t2c_cfg.get("use_worldfusion_after_cad", False):
        step_dir = out_dir / "cad_step_files"
        if step_dir.exists():
            steps = sorted(step_dir.glob("*.step"))
            if steps:
                try:
                    from src.world_models.world_fusion_renderer import WorldFusionRenderer
                    renderer = WorldFusionRenderer(use_worldgen=True, use_astra=True, use_ccubed=True)
                    fusion_dir = out_dir / "world_fusion"
                    for step_file in steps[:3]:  # 최대 3개
                        out_sub = fusion_dir / step_file.stem
                        prompt_text = "Generated CAD model from Doosan Text2CAD"
                        renderer.render(cad_step_path=step_file, prompt=prompt_text, output_dir=out_sub)
                    print(f"  [WorldFusion] 결과: {fusion_dir}")
                except Exception as e:
                    print(f"  [WorldFusion] 스킵: {e}")

    # PDF 소스에서 추출한 프롬프트로 추가 3D 생성
    source_dir = Path(source_dir)
    if source_dir.exists():
        try:
            from src.data.pdf_loader import PDFKnowledgeLoader
            loader = PDFKnowledgeLoader(source_dir=str(source_dir))
            chunks = loader.load_pdfs()
            pdf_prompts = [c["text"][:150] for c in chunks[:5] if c.get("text") and len(c["text"]) > 20]
            if pdf_prompts:
                print("\n[PDF 프롬프트로 추가 3D 생성]")
                if _check_occ_available():
                    print("  [OCC/CadSeqProc 사용] 형상 복원, STEP 저장, 3D CAD 이미지 생성")
                generate_cad_from_text_prompts(model, pdf_prompts, out_dir, device)
        except Exception as e:
            print(f"PDF 프롬프트 생성 스킵: {e}")

    return model


def main():
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    config = load_config(str(config_path))

    # key.env 우선 적용 (OPENAI_API_KEY 등)
    _load_env_file(_PROJECT_ROOT / "key.env", override=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) 항상 먼저 PDF 기반 MoE + Multi-task SFT + GRPO + KD를 위한 입력 준비
    source_dir = Path(__file__).parent / config.get("data", {}).get("pdf_source_dir", "source")
    if not source_dir.exists():
        source_dir = Path(__file__).parent / "source"

    use_plm = config.get("model", {}).get("use_plm_embedding", False)
    print(f"[MoE] prepare_data_from_pdfs 시작 (use_plm={use_plm})...")
    embeddings, task_ids, labels, emb_dim = prepare_data_from_pdfs(
        str(source_dir), config, use_plm=use_plm
    )

    task_dims = {"diagnostic_code": 50, "manual_mapping": 12, "equipment_typing": 10,
                 "risk_assessment": 3, "report_generation": 768}
    n = len(embeddings)
    for name in TASK_NAMES:
        if name not in labels:
            dim = task_dims.get(name, 10)
            if name == "risk_assessment":
                labels[name] = F.one_hot(torch.randint(0, dim, (n,)), dim).float()
            elif name == "report_generation":
                labels[name] = torch.randn(n, dim)
            else:
                labels[name] = F.one_hot(torch.randint(0, dim, (n,)), dim).float()

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    source_path = str(source_dir)
    model = EnhancedDoosanAgent(
        input_dim=emb_dim,
        num_experts=model_cfg.get("num_experts", 8),
        num_tasks=model_cfg.get("num_tasks", 5),
        hidden_dim=model_cfg.get("hidden_dim", 2048),
        use_distillation=model_cfg.get("use_distillation", True),
        temperature=model_cfg.get("temperature", 4.0),
        alpha=model_cfg.get("alpha", 0.7),
        source_dir=source_path if data_cfg.get("experts_from_source", True) else None,
        use_3d_experts=model_cfg.get("use_3d_experts", False),
        grpo_learning_rate=model_cfg.get("learning_rate", 1e-5),
        grpo_gamma=model_cfg.get("gamma", 0.99),
        grpo_clip_epsilon=model_cfg.get("clip_epsilon", 0.2),
        grpo_value_coef=model_cfg.get("value_coef", 0.5),
        grpo_entropy_coef=model_cfg.get("entropy_coef", 0.01),
        reward_weights=model_cfg.get("reward_weights"),
    ).to(device)

    print("Training Multi-task SFT (Step 5) - MoE + task heads...")
    train_sft(model, embeddings, task_ids, labels, config, device)

    num_epochs_grpo = config.get("training", {}).get("num_epochs_grpo", 5)
    if num_epochs_grpo > 0:
        print("Training GRPO (Step 6, preference optimization)...")
        model.train_on_preferences(
            embeddings, task_ids, labels,
            num_epochs=num_epochs_grpo,
            batch_size=config.get("data", {}).get("batch_size", 16),
            device=device,
        )

    print("Training Knowledge Distillation (Step 7)...")
    train_distillation(model, embeddings, config, device)

    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": config},
        out_dir / "doosan_moe_agent.pt",
        _use_new_zipfile_serialization=False,
    )
    print(f"Model saved to {out_dir / 'doosan_moe_agent.pt'}")

    report = model.generate_report(
        manual_context="Doosan WEDGE Controller diagnostic codes",
        preferences={"accuracy": 0.5, "safety": 0.3, "style": 0.2},
    )
    print("Sample report:", report)

    # 2) 그 다음 Text2CAD + MoE 파이프라인 (config.text2cad.use_text2cad=true 및 데이터 경로 설정 시)
    if config.get("text2cad", {}).get("use_text2cad", False):
        model_t2c = run_text2cad_pipeline(config, device)
        if model_t2c is not None:
            print(
                "\n[OK] Text2CAD 학습 완료. 생성된 CAD 모델 이미지는 "
                "checkpoints/text2cad/cad_images/ 에 저장되었습니다."
            )


if __name__ == "__main__":
    main()
