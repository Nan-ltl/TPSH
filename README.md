# TPSH

Clean open-source release of the **TPSH** codebase, containing only the
files required by `train.py` and `eval.py`.

The implementation combines a ViT-B/16 image backbone, CLIP text embeddings,
text-guided patch selection, and a hash layer for fine-grained image retrieval.
The training entry point optimizes hashing and patch-selection losses and reports
mean average precision (mAP), while the evaluation entry point performs a
single-batch profiler pass for parameter count, FLOPs, peak GPU memory, and
inference latency.

## Scope

This release intentionally excludes visualization scripts, homework/demo code,
IDE settings, deployment configuration, and unrelated branches. Private absolute
paths and server-specific settings have been removed. All configured asset paths
are local and relative to the repository root.

## Repository layout

```text
.
├── configs.yaml                 # Model assets and patch-selection settings
├── train.py                     # Training entry point
├── eval.py                      # Evaluation/profiling entry point
├── requirements.txt
├── LICENSE
├── models/
│   ├── configs.py               # ViT configuration presets
│   └── modeling.py              # Model, patch selector, and losses
├── retrieval/
│   └── retrieval_tools.py       # mAP calculation
└── utils/
    ├── Aircraf_dataloader.py
    ├── autoaugment.py
    ├── data_utils.py
    ├── dataset.py
    ├── dist_util.py
    ├── scheduler.py
    └── vegfru_dataloader.py
```

## Environment

- Linux is recommended
- Python 3.10 or newer
- An NVIDIA GPU with a working CUDA runtime; the entry points use CUDA-specific APIs
- PyTorch and torchvision installed for your CUDA version
- At least one of the supported fine-grained datasets
- ViT-B/16 and CLIP ViT-L/14 pretrained weights

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch separately when you need a specific CUDA build. See the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Pretrained assets

Create a local weights directory and place the checkpoints as follows:

```text
weights/
├── ViT-B_16.npz
├── clip-vit-large-patch14/
└── all-MiniLM-L6-v2/
```

Example downloads:

```bash
mkdir -p weights

wget -O weights/ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir weights/clip-vit-large-patch14

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir weights/all-MiniLM-L6-v2
```

`ViT-B_16.npz` is required by both entry points. The TPSH model uses the CLIP text
encoder at runtime. `text_encoder_path` is retained in `configs.yaml` for
configuration compatibility.

If your assets are stored elsewhere, edit `configs.yaml` or override the paths:

```bash
python train.py ... \
  --pretrained_dir /path/to/ViT-B_16.npz
```

For CLIP, edit `clip_path` in `configs.yaml`. All paths must be local paths; this
release contains no remote server configuration.

## Dataset layout

Pass the parent directory with `--data_root`. The selected dataset name is
appended automatically, so the default `./datasets/` plus
`--dataset CUB_200_2011` resolves to `./datasets/CUB_200_2011`.

Common layouts:

```text
datasets/
├── CUB_200_2011/
│   ├── images/
│   ├── BMVC/                       # One .txt description per image
│   ├── images.txt
│   ├── image_class_labels.txt
│   └── train_test_split.txt
└── Stanford_Dogs/
    ├── Images/
    ├── Annotation/
    ├── BMVC/                       # One .txt description per image
    ├── train_list.mat
    └── test_list.mat
```

The code also provides loaders for Stanford Cars, NABirds, iNaturalist 2017,
FGVC-Aircraft, and VegFru. Their expected directory names are defined in
`utils/data_utils.py` and the corresponding loader modules. Dataset files are
not included in this repository.

## Configuration

`configs.yaml` contains the portable defaults:

```yaml
text_encoder_path: "./weights/all-MiniLM-L6-v2/"
clip_path: "./weights/clip-vit-large-patch14/"
clip_architecture: "ViT-L-14"
img_mask_dim: 1
text_mask_dim: 1
least_chosen_patches: 12
select_layer_number: 1
max_enhance_percent: 0.66
```

Command-line arguments take precedence where a value is explicitly supplied.
Run either entry point with `-h` for the complete argument list.

## Training

Single-GPU example:

```bash
python train.py \
  --name cub_tpsh \
  --dataset CUB_200_2011 \
  --epoch 100 \
  --warmup_epochs 20 \
  --train_batch_size 64 \
  --hash_bit_list 16,32,48,64 \
  --learning_rate 0.02 \
  --fp16
```

Multi-GPU example:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --name cub_tpsh_ddp \
  --dataset CUB_200_2011 \
  --epoch 100 \
  --warmup_epochs 20 \
  --train_batch_size 64 \
  --hash_bit_list 16,32,48,64 \
  --learning_rate 0.02 \
  --fp16 \
  --dp_mode ddp
```

Outputs are written under `./output/<name>/`:

- `logger.log` – training log
- `<name>_<bits>bit_checkpoint.pth` – best checkpoint for each hash length
- `Hash_bit: _<bits>_Best_map.txt` – best mAP and run arguments
- `project/` – snapshot of the runtime source files

## Evaluation

`eval.py` follows the TPSH branch behavior: it performs a single-batch complexity
measurement and returns after the first batch. It reports parameter count, FLOPs,
peak allocated GPU memory, and per-batch inference time.

```bash
python eval.py \
  --name cub_eval \
  --dataset CUB_200_2011 \
  --hash_bit 64 \
  --train_batch_size 1 \
  --pretrained_dir ./weights/ViT-B_16.npz \
  --output_dir ./output
```

For a custom training checkpoint, pass `--pretrained_model /path/to/checkpoint.pth`.
The option is retained for compatibility with the branch interface.

## Notes on reproducibility

- The code is taken from the committed TPSH branch state, not from the local
  uncommitted working tree.
- The open-source copy changes only portability and packaging details: private
  absolute paths were replaced by relative paths, server/deployment material was
  removed, an unused SciPy test-only import was dropped, and stale private-path
  examples were removed.
- The model architecture, training loop, and profiler branches are preserved from
  the committed TPSH revision. `eval.py` intentionally returns after profiling the
  first batch, and `train.py` retains the profiler instrumentation present in that
  revision. Review these branches before starting a long benchmark run.
- Results depend on GPU count, batch size, random seed, PyTorch/CUDA versions,
  and the exact pretrained weights.
- Dataset and checkpoint files are intentionally excluded.

## Citation

This code builds on TransFG:

```bibtex
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

## License

MIT License. See [LICENSE](LICENSE). The original TransFG copyright notice is
preserved.

## Acknowledgements

This work builds on TransFG and the PyTorch ViT implementation from
[ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch).
