# MCCAG: Medical Concept Cross-Attention Graph

A research implementation of **MCCAG**, an architecture for automatic chest
X-ray radiology report generation on the IU X-Ray dataset. A ViT visual
encoder, a medical-concept graph pathway, triangular attention fusion, a
multimodal memory bank, and a transformer decoder are trained jointly and
evaluated with beam search against BLEU-1..4, ROUGE-L, METEOR, and CIDEr.

## Architecture

| # | Component | File |
|---|---|---|
| 1 | ViT-B/16 visual encoder (last 4/12 blocks fine-tuned), 196 patch + 1 CLS = 197 tokens | `src/models/vit_encoder.py` |
| 2 | Concept extractor (pads/projects cached concept embeddings to `(B,50,768)`) | `src/models/concept_extractor.py` |
| 3 | Concept graph builder (dense `(K,K)` edge weights, Laplacian structural positional encoding) | `src/models/graph_builder.py` |
| 4 | Graph-masked transformer encoder, 6 layers | `src/models/graph_transformer.py` |
| 5 | Triangular attention fusion (visual / concept / 32 learnable memory tokens, 278-token fused sequence) | `src/models/triangular_attention.py` |
| 6 | Multimodal memory: FIFO bank (size 512), top-32 attention, concept-coverage tracking | `src/models/memory_module.py` |
| 7 | 6-layer causal decoder with FINDINGS/IMPRESSION section embedding | `src/models/decoder.py` |
| 8 | `MCCAG`: wires 1-7 together, ablation flags select variants via config | `src/models/mccag.py` |

Image resolution is fixed at 224x224 (-> 196 patch tokens + 1 CLS = 197),
`hidden_dim=768`, `K=50` max concepts.

## Practical substitutions vs. the original spec

- **RadGraph / UMLS / MetaMap** are licensed/credentialed and unavailable
  offline. Concept extraction (`src/data/concept_utils.py`) instead uses
  scispacy `en_core_sci_md` NER + scispacy's UMLS `EntityLinker` for CUI
  grounding (filtered to TUIs `T047, T048, T190, T020, T023`, confidence
  > 0.7), and concept embeddings come from mean-pooled PubMedBERT
  (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`, 768-d)
  rather than a UMLS embedding table. The output schema
  (`text, cui, canonical_name, semantic_type, confidence, sentence_idx`) and
  the 3-edge-type graph (semantic / anatomical proximity / co-occurrence)
  match the spec.
- Everything else (token counts, hidden dims, layer counts, losses,
  optimizer schedule, beam search penalties, evaluation metrics) follows the
  spec as written.

## Directory layout

```
config/            # config.yaml (training/eval), model_config.yaml (architecture + ablation),
                    # dataset_config.yaml (paths, splits, concept settings)
src/data/          # XML parsing, preprocessing, tokenizer, concept extraction, dataset/collate, augmentation
src/models/        # ViT encoder, concept extractor, graph builder/transformer, fusion, memory, decoder, MCCAG
src/training/      # losses, optimizer/scheduler, callbacks, metrics, Trainer
src/evaluation/    # BLEU/ROUGE/METEOR/CIDEr wrappers, bootstrap-CI ensemble, Evaluator
src/inference/     # beam search, postprocessing, Predictor
src/utils/         # seeds, device, logger, checkpointer
scripts/           # prepare_data.py, check_shapes.py, train.py, evaluate.py, predict.py
data/processed/    # metadata.csv, concept_cache/, concept_embeddings.pt, cooccurrence.json, tokenizer/
data/splits/       # train.txt / val.txt / test.txt (study-level uids, 80/10/10)
checkpoints/       # latest.pt, best.pt, epoch_N.pt
logs/              # TensorBoard event files (logs/training/runs/<run_name>)
results/           # test_predictions.csv, evaluation_report.md
```

## Setup

Requires Python 3.10, an NVIDIA GPU (CUDA 12.1) is strongly recommended.

```powershell
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

The scispacy `UMLS` `EntityLinker` downloads its knowledge base (~1GB) on
first use (during `prepare_data.py` concept extraction, or `predict.py
--report-context`).

## 1. Prepare data

Parses `dataset/NLMCXR_reports/ecgen-radiology/*.xml`, builds
`data/processed/metadata.csv`, study-level train/val/test splits, trains the
BPE tokenizer, and runs concept extraction + graph statistics:

```powershell
python scripts/prepare_data.py
# Smoke test: skip the slow concept-extraction step
python scripts/prepare_data.py --skip-concepts
# Or limit concept extraction to the first N unique reports
python scripts/prepare_data.py --max-reports 50
```

## 2. Shape-check model components

```powershell
python scripts/check_shapes.py
```

## 3. Train

```powershell
python scripts/train.py
# Smoke test on a small subset
python scripts/train.py --max-train-samples 64 --max-val-samples 16 --num-epochs 2
# Resume
python scripts/train.py --resume checkpoints/latest.pt
```

TensorBoard logs are written to `logs/training/runs/<run_name>`; checkpoints
(`latest.pt`, `best.pt` by validation METEOR, periodic `epoch_N.pt`) go to
`checkpoints/`.

## 4. Evaluate

Runs beam-search generation over the test split and the full metric suite
(BLEU-1..4, ROUGE-L, METEOR, CIDEr, each with bootstrap 95% CI):

```powershell
python scripts/evaluate.py --checkpoint checkpoints/best.pt
# Smoke test
python scripts/evaluate.py --checkpoint checkpoints/latest.pt --max-test-samples 16 --n-bootstrap 100
```

Writes `results/test_predictions.csv`
(`sample_id, prediction, reference, beam_score, confidence_mean`) and
`results/evaluation_report.md` (metrics table, length stats, concept
coverage, worst-10 examples by ROUGE-L).

## 5. Predict on a single image

```powershell
python scripts/predict.py --image dataset/NLMCXR_png/CXR1_1_IM-0001-3001.png --checkpoint checkpoints/best.pt
# With clinical context to populate the concept pathway
python scripts/predict.py --image path/to/scan.png --report-context "history of cough and fever" --beam-size 5
```

## Ablation flags

`config/model_config.yaml`'s `ablation` section selects model variants
purely via config (no separate model classes):

| Variant | Flags |
|---|---|
| MCCAG-Full | all defaults |
| MCCAG-NoGraph | `use_graph: false` |
| MCCAG-NoConcept | `use_concepts: false` |
| MCCAG-NoMemory | `use_memory: false` |
| MCCAG-SimpleFusion | `fusion_type: concat` |
| CNN-Baseline | `vision_backbone: resnet50, use_concepts: false, use_memory: false, fusion_type: concat` |
| Transformer-Baseline | `use_concepts: false, use_memory: false, decoder_type: standard, fusion_type: concat` |

## Reproducibility

Seed `42` everywhere (`src/utils/seeds.py`), `cudnn.deterministic=True`,
`cudnn.benchmark=False`. AMP (fp16) training, AdamW with linear warmup ->
cosine decay, gradient accumulation (effective batch size 32 by default).

## Status / known limitations

The data pipeline, all 8 architecture components, training, evaluation, and
inference are implemented and verified end-to-end (shape checks, a 2-epoch
training smoke test, and evaluate/predict smoke tests all run without
error). Full training (50 epochs), the 7-way ablation comparison, the unit
test suite, notebooks, and extended docs are tracked as follow-up work and
have not been run.
