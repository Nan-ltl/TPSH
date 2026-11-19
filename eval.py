from __future__ import absolute_import, division, print_function

import argparse
import random
import time

import numpy as np
import yaml

from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS, ViT
from retrieval.retrieval_tools import calculate_map
from utils.data_utils import get_loader

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_args_with_yaml(yaml_config, args):
    # 遍历 YAML 中的配置，如果 argparse 没有提供该参数，则使用 YAML 配置
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:  # 如果命令行未提供该参数
            setattr(args, key, value)  # 设置默认值

    return args

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    config.batch_size = args.train_batch_size
    config.update(vars(args))

    if args.dataset == "CUB_200_2011":
        config.num_classes = 200
    elif args.dataset == "car":
        config.num_classes = 196
    elif args.dataset == "nabirds":
        config.num_classes = 555
    elif args.dataset == "Stanford_Dogs":
        config.num_classes = 120
    elif args.dataset == "INat2017":
        config.num_classes = 5089
    elif args.dataset == "Aircraft":
        config.num_classes = 100
    elif args.dataset == "vegfru":
        config.num_classes = 292


    # model = VisionTransformer(config, args.img_size, num_classes=config.num_classes)
    # if args.pretrained_model is not None:
    #     pretrained_model = torch.load(args.pretrained_model)['model']
    #     model.load_state_dict(pretrained_model, strict=False)

    model = ViT(config=config, hash_bit=config.hash_bit, ratio=0.25, num_classes=config.num_classes)
    return args, model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def eval(args, model):

    train_loader, test_loader, database_loader = get_loader(args, world_size=1)

    model.to(args.device)
    model.eval()
    with torch.no_grad():

        query_images_hash_bit = torch.empty(1, args.hash_bit).to(args.device)

        query_labels = torch.empty(1).to(args.device)
        query_index = torch.empty(1).to(args.device)
        test_iterator = tqdm(test_loader,
                             desc="Query data... ",
                             bar_format="{l_bar}{bar}{r_bar}",
                             dynamic_ncols=True,
                             leave=False,
                             position=0,
                             total=len(test_loader),
                             disable=torch.cuda.current_device() != 0)
        for batch in test_iterator:
            x, y, text = batch
            index = torch.tensor([1])
            images = x.to(args.device)
            labels = y.to(args.device)
            index = index.to(args.device)

            from fvcore.nn import FlopCountAnalysis

            def evaluate_model_complexity(model, images, labels, text):

                device = images.device
                model = model.to(device)
                model.eval()

                ############################################################
                # 1️⃣ 参数量
                ############################################################
                # model.transformer.encoder.patch_selector[0].i2t.requires_grad_(False)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print("=" * 50)
                print(f"Total Params: {total_params / 1e6:.3f} M")
                print(f"Trainable Params: {trainable_params / 1e6:.3f} M")

                ############################################################
                # 2️⃣ FLOPs（包装 forward 以适配多输入）
                ############################################################
                class Wrapper(nn.Module):
                    def __init__(self, model, labels, text):
                        super().__init__()
                        self.model = model
                        self.labels = labels
                        self.text = text

                    def forward(self, images):
                        return self.model(
                            images,
                            labels=self.labels,
                            sentence=self.text,
                            retrieval=True
                        )

                wrapped_model = Wrapper(model, labels.cuda(), text).to(device)
                # wrapped_model.cpu()
                # wrapped_model.model.transformer.cuda()
                # wrapped_model.model.hash_layer.cuda()
                # wrapped_model.model.transformer.encoder.patch_selector[0].i2t.cpu()
                wrapped_model.eval()

                with torch.no_grad():
                    flops = FlopCountAnalysis(wrapped_model, images)
                    total_flops = flops.total()

                print(f"FLOPs: {total_flops / 1e9:.3f} GFLOPs")

                ############################################################
                # 3️⃣ 推理显存占用
                ############################################################
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # warm-up
                for _ in range(10):
                    with torch.no_grad():
                        _ = model(
                            images,
                            labels=labels.cuda(),
                            sentence=text,
                            retrieval=True
                        )

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                with torch.no_grad():
                    _ = model(
                        images,
                        labels=labels.cuda(),
                        sentence=text,
                        retrieval=True
                    )

                torch.cuda.synchronize()

                peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                print(f"Peak GPU Memory: {peak_memory / 1024:.3f} GB")

                ############################################################
                # 4️⃣ 推理时间
                ############################################################
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

                repetitions = 1

                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                starter.record()

                for _ in range(repetitions):
                    with torch.no_grad():
                        _ = model(
                            images,
                            labels=labels.cuda(),
                            sentence=text,
                            retrieval=True
                        )

                ender.record()
                end = time.perf_counter_ns()
                torch.cuda.synchronize()

                inference_time = (end - start) / 1e6 / repetitions
                print(f"Inference Time per batch: {inference_time:.3f} ms")

                print("=" * 50)

            ############################################################
            # 🔥 直接调用
            ############################################################

            evaluate_model_complexity(model, images, labels, text)
            return


            query_batch = torch.sign(model(images, labels=labels.cuda(),sentence = text, retrieval = True)).to(args.device)
            query_images_hash_bit = torch.cat((query_images_hash_bit, query_batch), dim=0)
            query_labels = torch.cat((query_labels, labels), dim=0)
            query_index = torch.cat((query_index, index), dim=0)
        query_images_hash_bit = query_images_hash_bit[1:]
        query_labels = query_labels[1:]
        query_index = query_index[1:]
        np_query_images = query_images_hash_bit.to("cpu").numpy()
        np_query_labels = query_labels.to("cpu").numpy()
        np_query_index = query_index.to("cpu").numpy()

        train_iterator = tqdm(database_loader,
                              desc="Retrieval data... ",
                              bar_format="{l_bar}{bar}{r_bar}",
                              dynamic_ncols=True,
                              leave=False,
                              position=0,
                              total=len(database_loader),
                              disable=torch.cuda.current_device() != 0)
        retrieval_images_hash_bit = torch.empty(1, args.hash_bit).to(args.device)
        retrieval_labels = torch.empty(1).to(args.device)
        retrieval_index = torch.empty(1).to(args.device)

        for batch in train_iterator:
            x, y, text = batch
            index = torch.tensor([1])
            images = x.to(args.device)
            labels = y.to(args.device)
            index = index.to(args.device)

            u_hash = model(images, labels=labels.cuda(),sentence = text, retrieval = True)
            u_hash = torch.sign(u_hash)
            retrieval_images_hash_bit = torch.cat((retrieval_images_hash_bit, u_hash), dim=0)
            retrieval_labels = torch.cat((retrieval_labels, labels), dim=0)
            retrieval_index = torch.cat((retrieval_index, index), dim=0)

        retrieval_images_hash_bit = retrieval_images_hash_bit[1:]  # [train_number, bit]
        retrieval_labels = retrieval_labels[1:]  # [train_number]
        retrieval_index = retrieval_index[1:]
        np_retrieval_images = retrieval_images_hash_bit.to("cpu").numpy()
        np_retrieval_labels = retrieval_labels.to("cpu").numpy()
        np_retrieval_index = retrieval_index.to("cpu").numpy()
        # retrieval_labels = torch.Tensor(dhd["retrieval_labels"]).cuda()
        # retrieval_images_hash_bit = torch.Tensor(dhd["retrieval_hash_codes"]).cuda()
        # query_labels = torch.Tensor(dhd["query_labels"]).cuda()
        # query_images_hash_bit = torch.Tensor(dhd["query_hash_codes"]).cuda()
        export_path = Path(args.output_dir) / args.name / f"TPSH_1K_{args.hash_bit}_{args.dataset}.npz"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_path,
                 query_labels=np_query_labels, query_hash_codes=np_query_images, query_index=0,
                 retrieval_labels=np_retrieval_labels, retrieval_hash_codes=np_retrieval_images, retrieval_index=0)
        return
        # def compute_topk_accuracy(query_hashes, query_labels, retrieval_hashes, retrieval_labels,
        #                           topk=(1, 2, 5, 10, 20, 50)):
        #     query_hashes = query_hashes.sign()
        #     retrieval_hashes = retrieval_hashes.sign()
        #
        #     sim = torch.matmul(query_hashes, retrieval_hashes.T)  # [m, n]
        #     max_k = max(topk)
        #     topk_indices = sim.topk(max_k, dim=1, largest=True).indices  # [m, max_k]
        #
        #     accuracies = []
        #     for k in topk:
        #         correct = 0
        #         for i in range(query_labels.size(0)):
        #             retrieved_labels = retrieval_labels[topk_indices[i, :k]]
        #             if (retrieved_labels == query_labels[i]).any():
        #                 correct += 1
        #         acc = correct / query_labels.size(0)
        #         accuracies.append(acc)
        #     return topk, accuracies
        #
        # def plot_topk_curve(topk_list, accuracy_list):
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(topk_list, accuracy_list, marker='o', linestyle='-', color='blue', label='Top-k Accuracy')
        #     plt.xticks(topk_list)
        #     plt.xlabel('Top-K')
        #     plt.ylabel('Accuracy')
        #     plt.title('Top-K Accuracy Curve')
        #     plt.grid(True)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #
        # # 👇 替换成你自己的变量
        # # query_hashes = ...
        # # query_labels = ...
        # # retrieval_hashes = ...
        # # retrieval_labels = ...
        #
        # # 示例调用
        # topk_list, acc_list = compute_topk_accuracy(query_images_hash_bit, query_labels, retrieval_images_hash_bit, retrieval_labels)
        # # np.savez(
        # #     topk_list = topk_list, acc_list = acc_list)
        #
        # plot_topk_curve(topk_list, acc_list)
        hamming = (retrieval_images_hash_bit.shape[1] - torch.mm(query_images_hash_bit,
                                                                 retrieval_images_hash_bit.t())) / 2  # [query_number, retrieval_number]
        cat_hamming = torch.cat(
            (hamming.unsqueeze(dim=-1), retrieval_labels.unsqueeze(0).unsqueeze(-1).repeat(hamming.shape[0], 1, 1)),
            dim=-1)
        ap_list = torch.empty(1).to(args.device)
        for i in tqdm(range(cat_hamming.shape[0]),
                      dynamic_ncols=True,
                      leave=False,
                      disable=torch.cuda.current_device() != 0):
            single_hamming = cat_hamming[i, :]
            indicate = torch.argsort(single_hamming[:, 0])
            rank_labels = single_hamming[:, 1][indicate]
            rank_labels = rank_labels[:-1]
            true_number = (rank_labels == query_labels[i]).sum()
            best = torch.linspace(1, true_number.item(), true_number.item()).to(args.device)
            result = (rank_labels == query_labels[i].expand(rank_labels.shape[0]))
            tindex = (torch.nonzero(result == True, as_tuple=False).squeeze(dim=-1) + 1.0).float()
            if best.shape[0] == tindex.shape[0]:
                ap = (best / tindex).mean().unsqueeze(0)
                ap_list = torch.cat((ap_list, ap), dim=0)
            else:
                raise ValueError("Doesn't match!")
        ap_list = ap_list[1:]
        map = ap_list.mean().item()
        print("map = ", map)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    # About information
    parser.add_argument("--name", default="Original_VIT_B_16",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "Stanford_Dogs",
                                              "nabirds", "INat2017","Aircraft","vegfru"], default="Stanford_Dogs",
                        help="Which dataset.")
    parser.add_argument("--model_type", choices=["ViT-S_16","ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    # About path
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument("--pretrained_dir", type=str, default="./weights/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Optional checkpoint to load")
    parser.add_argument("--output_dir", default="./output/", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--config', type=str, default='./configs.yaml', help='Path to config file')

    # About default parameters
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument('--resize', type=int, default=256,  # 600
                        help="resize of images")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument('--seed', type=int, default=11,
                        help="random seed for initialization")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--hash_bit', type=int, default=16)

    args = parser.parse_args()

    # Some other args
    args.eval_batch_size = args.train_batch_size
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Merge with config.yaml
    yaml_config = load_config(args.config)
    args = merge_args_with_yaml(yaml_config, args)
    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args.device = device
    yaml_config = load_config(args.config)
    args = merge_args_with_yaml(yaml_config, args)
    # Setup logging

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    eval(args, model)

if __name__ == "__main__":
    main()
