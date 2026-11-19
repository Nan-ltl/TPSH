# coding=utf-8
from __future__ import absolute_import, division, print_function

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import argparse
import os
import random
import numpy as np
import time
import yaml
import shutil
import json
from datetime import datetime
import gc

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from utils.dist_util import set_model_dp_mode

from models.modeling import VisionTransformer, CONFIGS, ViT
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

from retrieval.retrieval_tools import calculate_map
from pathlib import Path


logger = logging.getLogger(__name__)



# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CUB_200_2011 --split non-overlap --epoch 100 --warmup_epochs 20  --name test --train_batch_size 64 --hash_bit_list 64 --learning_rate 0.02
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset Stanford_Dogs --epoch 100 --warmup_epochs 10  --name Dogs_Vit_all_bit_base_2 --train_batch_size 128 --hash_bit_list 16,32,48,64 --learning_rate 0.02

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RankFilter(logging.Filter):
    def filter(self, record):
        return torch.cuda.current_device() == 0

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

def save_model(args, model, optimizer, scheduler, scaler=None):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_checkpoint = Path(args.output_dir)/args.name/(f"{args.name}_{args.hash_bit}bit_checkpoint.pth")
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved best model checkpoint to [DIR: %s]", args.output_dir)


def save_project(args):
    (Path(args.output_dir) / args.name).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / args.name / "project").mkdir(parents=True, exist_ok=True)

    shutil.copy(Path(__file__), Path(args.output_dir) / args.name / "project"/ (Path(__file__).name))
    shutil.copy(Path(__file__).parent / "eval.py", Path(args.output_dir) / args.name / "project"/ ((Path(__file__).parent / "eval.py").name))
    shutil.copy(Path(__file__).parent / "configs.yaml", Path(args.output_dir) / args.name / "project" / ((Path(__file__).parent / "configs.yaml").name))
    shutil.copytree(Path(__file__).parent / "models", Path(args.output_dir) / args.name / "project" / "models", dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parent / "retrieval", Path(args.output_dir) / args.name / "project" / "retrieval", dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parent / "utils", Path(args.output_dir) / args.name / "project" / "utils", dirs_exist_ok=True)

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


    model = VisionTransformer(config, args.img_size, num_classes=config.num_classes)

    model.load_from(np.load(args.pretrained_dir))
    # model = ViT(config=config, hash_bit=config.hash_bit, ratio=0.25, num_classes=config.num_classes)
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def record_best_map_and_args(args, epoch, best_map):
    with open(Path(args.output_dir) / args.name / (f"Hash_bit:_{args.hash_bit}_Best_map.txt"),
              "w") as f:
        content = f"{datetime.now()}  Training epoch : {epoch + 1},bit: {args.hash_bit}, best_map : {best_map}\n"
        f.write(content)
        # 将 Namespace 转换为字典
        condition_dict = vars(args)
        condition_dict["device"] = str(condition_dict["device"])
        # 将字典转换为字符串（JSON 格式）
        condition_str = json.dumps(condition_dict, indent=4)
        f.write(condition_str)

def train(rank, world_size, args, model):
    """ Train the model """
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total epoch = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size  * (
                    torch.distributed.get_world_size() if torch.cuda.current_device() else 1))
    logger.info("Model hash_bit = %d ", args.hash_bit)

    #prepare for training
    if torch.cuda.current_device() == 0:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    # writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # Prepare dataset and set dp model
    if world_size>1:
        train_loader, train_sampler, test_loader, database_loader = get_loader(args, rank, world_size)
        model = set_model_dp_mode(args, model, rank, world_size)
    else:
        train_loader, test_loader, database_loader = get_loader(args, rank, world_size)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(params=model.parameters(),lr=2.0e-5, weight_decay=1e-5)
    epoch = args.epoch
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_epochs*len(train_loader), t_total=len(train_loader)*epoch)
        # scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_epochs*len(train_loader), t_total=len(train_loader)*epoch)

    if args.fp16:
        scaler = GradScaler()
    else:
        scaler = None

    set_seed(args)
    # scheduler.step()

    best_map = 0
    start_time = time.time()

    for epoch in range(args.epoch):
        # save_model(args, model, optimizer, scheduler)
        if world_size>1:
            train_sampler.set_epoch(epoch)
        model.train()

        epoch_iterator = tqdm(train_loader,
                              desc="Training (loss=X.X)",
                              dynamic_ncols=True,
                              bar_format="{l_bar}{bar}{r_bar}",
                              position=0,
                              disable=torch.cuda.current_device() != 0,
                              ncols=100)
        train_loss = {}
        for batch in epoch_iterator:
            optimizer.zero_grad()

            x, y, z = batch
            x = x.to(args.device)

            # y = torch.nn.functional.one_hot(y,200)
            y = y.to(args.device)
            # z = z.to(args.device)


            if args.fp16:
                with autocast():
                    loss = model(x, y, z)
                    train_loss += loss.item()
                scaler.scale(loss).backward()
                # 使用scaler来处理混合精度训练中的梯度裁剪
                scaler.unscale_(optimizer)  # 取消梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                from thop import profile
                from thop import clever_format

                # =========================
                # 1️⃣ 统计参数量
                # =========================
                def count_parameters(model):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    print(f"Total params: {total_params:,}")
                    print(f"Trainable params: {trainable_params:,}")
                    return total_params, trainable_params

                def compute_flops(model, inputs):
                    model.eval()
                    with torch.no_grad():
                        flops, params = profile(model, inputs=inputs, verbose=False)

                    flops, params = clever_format([flops, params], "%.3f")

                    print(f"FLOPs: {flops}")
                    print(f"Params (from FLOPs tool): {params}")

                    return flops

                # =========================
                # 3️⃣ 统计显存
                # =========================
                def measure_gpu_memory(model, inputs, device="cuda"):
                    model.to(device)
                    model.train()

                    # 重置显存峰值
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()

                    # 前向
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    start = time.perf_counter_ns()
                    iteration = 20
                    for i in range(iteration):
                        outputs = model(*inputs)[0]

                        # 构造一个简单loss（假设输出是tensor）
                        if isinstance(outputs, (list, tuple)):
                            loss = sum(o.mean() for o in outputs)
                        else:
                            loss = outputs.mean()

                        # 反向
                        loss.backward()
                        optimizer.step()
                    ender.record()
                    end = time.perf_counter_ns()
                    # 当前显存
                    current_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
                    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2

                    print(f"Current GPU memory: {current_memory / 1024:.2f} GB")
                    print(f"Peak GPU memory: {peak_memory / 1024:.2f} GB")
                    inference_time = (end-start)/ 1e6 / iteration
                    print(f"Inference Time per batch: {inference_time:.3f} ms")

                    return current_memory, peak_memory


                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = (x, y, z)

                print("========== 参数量 ==========")
                count_parameters(model)

                print("\n========== FLOPs ==========")
                compute_flops(model.to(device), inputs)

                if device == "cuda":
                    print("\n========== 显存占用（前向+反向） ==========")
                    measure_gpu_memory(model, inputs, device)
                return
                loss, loss_dict, mean_num_patch = model(x, y, z)  # [bs, 13, 768]
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
                # 每个 batch 中更新累加值
                for k, v in loss_dict.items():
                    if k not in train_loss:
                        train_loss[k] = v  # 第一次赋值
                    else:
                        train_loss[k] += v # 后续累加
                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_iterator.set_description(
                f"Training (loss_total={loss:.4f}, "
                # f"loss_cross_entropy={loss_dict['loss_cross_entropy']:.4f}, "
                # f"loss_qua_reg={loss_dict['loss_qua_reg']:.4f}, "
                f"loss_select={loss_dict['loss_select']:.4f}), "
                # f"mean_num_patch={mean_num_patch:.2f}"
            )
        for k, v in train_loss.items():
             train_loss[k] /= len(train_loader)
        logger.info(
            f"{datetime.now()}  Training epoch : {epoch + 1}/{args.epoch}, bit: {args.hash_bit},  lr: {optimizer.param_groups[0]['lr']:.4f},"
            f"loss_total:{sum(loss for loss in train_loss.values()):.4f},loss_cross_entropy:{train_loss['loss_cross_entropy']:.4f}," 
            f"loss_qua_reg:{train_loss['loss_qua_reg']:.4f},"
            f"loss_select:{train_loss['loss_select']:.4f}"
        )
        if world_size<=1:
            if epoch+1 == args.epoch or (epoch % args.eval_every == 0 and epoch>=25):
                map = calculate_map(args=args, test_loader=test_loader, database_loader=database_loader, model=model)
                if map > best_map:
                    best_map = map
                    save_model(args, model, optimizer, scheduler, scaler= scaler if scaler else None)
                    record_best_map_and_args(args, epoch, best_map)
                logger.info(f"{datetime.now()}  Training epoch : {epoch+1}/{args.epoch},bit: {args.hash_bit}, map:{map}, best_map : {best_map}")
        else:
            if epoch + 1 == args.epoch or epoch % args.eval_every == 0:
                dist.barrier()

                map = calculate_map(args=args, test_loader=test_loader, database_loader=database_loader,
                                    model=model)
                if map > best_map:
                    best_map = map
                    if rank == 0:
                        save_model(args, model, optimizer, scheduler, scaler=scaler if scaler else None)
                        record_best_map_and_args(args, epoch, best_map)
                dist.barrier()
                logger.info(
                    f"{datetime.now()}  Training epoch : {epoch + 1}/{args.epoch},bit: {args.hash_bit}, map:{map}, best_map : {best_map}")

    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    # About information
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "Stanford_Dogs",
                                              "nabirds", "INat2017","Aircraft","vegfru"], default="CUB_200_2011",
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
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output/", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--config', type=str, default='./configs.yaml', help='Path to config file')


    # About default parameters
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument('--resize', type=int, default=256,#600
                        help="resize of images")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    # About train parameters
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=0.02, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--epoch", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=40, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=152,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--eval_every", default=10, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument('--hash_bit_list',type=str,default='16,32,48,64')
    if torch.cuda.device_count() > 1:
        parser.add_argument('--dp_mode', type=str, choices=["fsdp","ddp"], default='ddp', required=True, help="Choose which mode to train")

    args = parser.parse_args()

    # Some other args
    args.eval_batch_size = args.train_batch_size
    args.hash_bit_list = list(map(int, args.hash_bit_list.split(',')))
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    args.n_gpu = torch.cuda.device_count()


    # Merge with config.yaml
    yaml_config = load_config(args.config)
    args = merge_args_with_yaml(yaml_config, args)

    #set logger
    (Path(args.output_dir) / args.name).mkdir(parents=True, exist_ok=True)
    logger_path = Path(args.output_dir) / args.name / "logger.log"



    rank = 0

    # Setup CUDA, GPU & distributed training
    if args.n_gpu > 1:
        rank = int(os.environ["LOCAL_RANK"])
        logger.warning("Initiating distributed training. Rank = %s" % rank)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=args.n_gpu)
        torch.cuda.set_device(rank)

    #set logger
    logger.setLevel(logging.INFO)
    # 文件处理器
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(logging.WARNING)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 添加过滤器
    logger.addFilter(RankFilter())

    #
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if torch.cuda.current_device() else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (os.environ["LOCAL_RANK"] if args.n_gpu > 1 else 0, os.environ["CUDA_VISIBLE_DEVICES"], args.n_gpu, bool(args.n_gpu > 1), args.fp16))

    # Set seed
    set_seed(args)

    # Save the project first (including "models", "retrival", "utils", "eval.py", train.py")
    save_project(args)

    # Training all bits
    for bit in args.hash_bit_list:

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.hash_bit = bit
        # Model & Tokenizer Setup
        args, model = setup(args)
        model.to(args.device)


        # Training
        train(rank, args.n_gpu, args, model)
        del model
        gc.collect()
        torch.cuda.empty_cache()


    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()

# export PYTHONPATH=''
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset Stanford_Dogs --split overlap --epoch 200 --warmup_epochs 100 --name DOGS_VIT_B_16_all_bit_Trans --train_batch_size 64 --img_size 224
# python setup.py install
#./download_model.sh

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset Stanford_Dogs  --split overlap --epoch 100 --warmup_epochs 10  --name Dogs_Vit_Overlap_all_bit_base_2 --train_batch_size 64 --hash_bit_list 16,32,48,64 --learning_rate 0.02
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CUB_200_2011 --epoch 100 --warmup_epochs 20  --name CUB_test --train_batch_size 64 --hash_bit_list 16,32,48,64 --learning_rate 0.02
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset Stanford_Dogs --split overlap --epoch 100 --warmup_epochs 10  --name Dogs_TPSH_text_sim --train_batch_size 64 --hash_bit_list 16,32,48,64 --learning_rate 0.02


#CUDA_VISIBLE_DEVICES=0 python train.py --dataset nabirds --epoch 100 --warmup_epochs 20  --name Nabirds_ViT-S_16 --train_batch_size 128 --hash_bit_list 16,32,48,64 --learning_rate 0.02
