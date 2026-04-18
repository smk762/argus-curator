"""CLIP and DINOv2 embedding extraction for semantic diversity clustering.

Both models are lazy-loaded on first use and cached for the session.
All operations are batched for GPU efficiency.

Aesthetic scoring reuses the loaded CLIP model — no extra model required.
Two contrasting text prompts are embedded at init time; each image's
cosine similarity to the positive prompt minus the negative gives a
lightweight aesthetic proxy that correlates well with human preference
without needing a separate aesthetic predictor head.
"""

from __future__ import annotations

import structlog

from PIL import Image

logger = structlog.get_logger()

# Aesthetic proxy prompts — embedded once at CLIP load time.
_AESTHETIC_POS = "professional photograph, sharp focus, beautiful composition, high quality"
_AESTHETIC_NEG = "blurry photo, low quality, noise, compression artifacts, watermark, text overlay"


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CLIPEmbedder:
    """Extracts CLIP image embeddings and aesthetic scores.

    Usage::

        embedder = CLIPEmbedder("openai/clip-vit-large-patch14", device="cuda")
        embeddings = embedder.embed(images)          # list[list[float]]
        aesthetic   = embedder.aesthetic(images)     # list[float]
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP embedding requires 'transformers' and 'torch'. "
                "Install with: pip install 'argus-curator[gpu]'"
            ) from exc

        self._device = _resolve_device(device)
        logger.info("clip_load_start", model=model_name, device=self._device)
        self._proc = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        import torch
        with torch.no_grad():
            text_inputs = self._proc(
                text=[_AESTHETIC_POS, _AESTHETIC_NEG],
                return_tensors="pt", padding=True,
            )
            text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
            tf = self._model.get_text_features(**text_inputs)
            self._pos_feat = tf[0:1] / tf[0:1].norm(dim=-1, keepdim=True)
            self._neg_feat = tf[1:2] / tf[1:2].norm(dim=-1, keepdim=True)

        logger.info("clip_load_done", model=model_name)

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        """Return normalised CLIP image embeddings (one per image)."""
        import torch
        inputs = self._proc(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()

    def aesthetic(self, images: list[Image.Image]) -> list[float]:
        """Return aesthetic score ∈ [0, 1] for each image.

        Computed as (cos_sim_positive - cos_sim_negative + 1) / 2,
        clipped to [0, 1].
        """
        import torch
        embeddings = self.embed(images)
        emb_t = torch.tensor(embeddings, dtype=torch.float32)
        pos_sim = (emb_t @ self._pos_feat.T.cpu()).squeeze(-1)
        neg_sim = (emb_t @ self._neg_feat.T.cpu()).squeeze(-1)
        scores = ((pos_sim - neg_sim + 1.0) / 2.0).clamp(0.0, 1.0)
        return scores.tolist()

    def embed_and_score(
        self, images: list[Image.Image]
    ) -> tuple[list[list[float]], list[float]]:
        """Single-pass: returns (embeddings, aesthetic_scores)."""
        import torch
        inputs = self._proc(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        feats_cpu = feats.cpu().float()
        pos_sim = (feats_cpu @ self._pos_feat.T.cpu()).squeeze(-1)
        neg_sim = (feats_cpu @ self._neg_feat.T.cpu()).squeeze(-1)
        aesthetic = ((pos_sim - neg_sim + 1.0) / 2.0).clamp(0.0, 1.0).tolist()
        return feats_cpu.tolist(), aesthetic


class DINOEmbedder:
    """Extracts DINOv2 image embeddings for structural diversity clustering.

    DINOv2 captures local/structural features that CLIP sometimes misses
    (e.g. image background, texture, spatial layout) making it a useful
    complement for composition-diversity clustering.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv2 embedding requires 'transformers' and 'torch'."
            ) from exc

        self._device = _resolve_device(device)
        logger.info("dino_load_start", model=model_name, device=self._device)
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        logger.info("dino_load_done", model=model_name)

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        import torch
        inputs = self._proc(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            feats = out.last_hidden_state[:, 0]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()


class EmbeddingPool:
    """Orchestrates CLIP and optional DINOv2 in batches over a list of images.

    Returns per-image (clip_embedding, dino_embedding, aesthetic_score).
    dino_embedding is None when DINOv2 is disabled.
    """

    def __init__(
        self,
        clip_model: str,
        dino_model: str | None,
        batch_size: int,
        device: str = "auto",
    ) -> None:
        self._batch = batch_size
        self._clip = CLIPEmbedder(clip_model, device=device)
        self._dino = DINOEmbedder(dino_model, device=device) if dino_model else None

    def run(
        self, images: list[Image.Image]
    ) -> list[tuple[list[float], list[float] | None, float]]:
        """Process all images in batches.

        Returns list of (clip_emb, dino_emb_or_None, aesthetic_score).
        """
        n = len(images)
        clip_embs: list[list[float]] = []
        aesthetics: list[float] = []
        dino_embs: list[list[float] | None] = []

        for start in range(0, n, self._batch):
            batch = images[start: start + self._batch]
            logger.info(
                "embedding_batch",
                start=start, end=start + len(batch), total=n,
            )
            embs, aes = self._clip.embed_and_score(batch)
            clip_embs.extend(embs)
            aesthetics.extend(aes)

            if self._dino is not None:
                dino_embs.extend(self._dino.embed(batch))
            else:
                dino_embs.extend([None] * len(batch))

        return list(zip(clip_embs, dino_embs, aesthetics))


def availability() -> dict[str, object]:
    """Report which embedding deps are installed."""
    info: dict[str, object] = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
        info["cuda"] = False
    for name, pkg in [("clip", "transformers"), ("dino", "transformers")]:
        try:
            __import__(pkg)
            info[name] = True
        except ImportError:
            info[name] = False
    return info
