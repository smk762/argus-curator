# argus-curator

Dataset curation for LoRA training — quality filtering, embedding-based diversity clustering, and optimal subset selection.

Replaces and extends the `gallery/image_scanner.py` scanner from imogen with a standalone, pip-installable package.

## Pipeline

```
Phase 1 (CPU)   Resolution / aspect / blur / artifact filtering + pHash
Phase 2 (GPU)   CLIP + optional DINOv2 embeddings + aesthetic scoring
Phase 2b (GPU)  YOLO person detection + MTCNN face detection
Phase 3 (CPU)   De-duplication → clustering → tag-balance boost → selection
```

## Quick start

```bash
pip install "argus-curator[server,gpu]"

# Scan a folder, write results to JSON
argus-curator scan /path/to/images --objective identity --output results.json

# Copy selected images to training folder
argus-curator export results.json /path/to/training_subset

# Start the API server
argus-curator serve --port 8101 --cors
```

## Training objectives

| Objective | Focus | Key changes |
|-----------|-------|-------------|
| `identity` | Person/character LoRA | MTCNN+YOLO enabled, strict blur, face-count penalty |
| `style` | Style/aesthetic LoRA | High aesthetic weight, max diversity |
| `wardrobe` | Clothing/outfit LoRA | Full-body preference, YOLO enabled |
| `concept` | Object/concept LoRA | Coverage-maximising clustering |

## License

MIT
