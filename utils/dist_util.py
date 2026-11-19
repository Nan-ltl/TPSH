from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools



def set_model_dp_mode(args, model, rank, world_size):
    """

    Args:
        args:
        model:
        rank:
        world_size:

    Returns: a ddp_model or a size_based_auto_wrap_fsdp_model

    """

    if args.dp_mode == "ddp":
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    elif args.dp_mode == "fsdp":
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad) / world_size)
        )
        model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, use_orig_params=True, mixed_precision=args.fp16, device_id=rank)
    return model