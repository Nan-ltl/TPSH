# TPSH

Text-guided patch selection and hashing for fine-grained image retrieval.
TPSH combines a ViT-B/16 image backbone with CLIP text embeddings, a
text-guided patch selector, and a hash layer.

## Installation

Python 3.10+ and an NVIDIA GPU with CUDA are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version separately.

## Pretrained assets

Place the required files under `weights/`:

```text
weights/
├── ViT-B_16.npz
├── clip-vit-large-patch14/
└── all-MiniLM-L6-v2/
```

```bash
mkdir -p weights

wget -O weights/ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir weights/clip-vit-large-patch14

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir weights/all-MiniLM-L6-v2
```

Paths can be changed in `configs.yaml` or overridden with command-line
arguments.

## Dataset

Datasets are not included. Place them under `./datasets/<dataset_name>/`.
`--data_root` sets the parent directory; for example, `--dataset CUB_200_2011`
resolves to `./datasets/CUB_200_2011`.

```text
datasets/
└── CUB_200_2011/
    ├── images/
    ├── BMVC/
    ├── images.txt
    ├── image_class_labels.txt
    └── train_test_split.txt
```

## Training

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

For distributed training:

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

Outputs are saved under `./output/<name>/`.

## Evaluation

```bash
python eval.py \
  --name cub_eval \
  --dataset CUB_200_2011 \
  --hash_bit 64 \
  --train_batch_size 1 \
  --pretrained_dir ./weights/ViT-B_16.npz \
  --output_dir ./output
```

Use `--pretrained_model /path/to/checkpoint.pth` to evaluate a training
checkpoint.

## Citation

```bibtex
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

## License

MIT License. See [LICENSE](LICENSE).
