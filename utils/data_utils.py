import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017
from .Aircraf_dataloader import load_Aircraft
from .vegfru_dataloader import load_vegfru
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args, rank=None, world_size=None):
    # if torch.cuda.current_device() not in [-1, 0]:
    #     torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        # Cub2011.init(root=args.data_root)
        train_transform=transforms.Compose([
                                    transforms.Resize((args.resize, args.resize), Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        test_transform=transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # trainset = Cub2011(root=args.data_root, mode="train", transform=train_transform)
        # testset = Cub2011(root=args.data_root, mode="query", transform=test_transform)
        # databaseset = Cub2011(root=args.data_root, mode="retrieval", transform=test_transform)
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
        databaseset = CUB(root=args.data_root, is_train=True, transform= test_transform)
    elif args.dataset == 'Stanford_Dogs':
        train_transform=transforms.Compose([
                                    transforms.Resize((args.resize, args.resize), Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        test_transform=transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
        databaseset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=test_transform,
                        download=False
                        )
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )

    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
        databaseset = NABirds(root=args.data_root,train=True, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'Aircraft':
        train_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        testset, trainset, databaseset = load_Aircraft(args.data_root,
                                                       train_transform=train_transform, test_transform=test_transform
                                                       )
    elif args.dataset == 'vegfru':
        train_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize), Image.BILINEAR),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        testset, trainset, databaseset = load_vegfru(os.path.join(args.data_root, "VegFru"),
                                                     train_transform=train_transform, test_transform=test_transform)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()
    if world_size>1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset) # if args.local_rank == -1 else DistributedSampler(testset)
    database_sampler = SequentialSampler(databaseset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=16,
                              # shuffle=True,
                              # drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=16,
                             # shuffle=True,
                             pin_memory=True) if testset is not None else None
    database_loader = DataLoader(databaseset,
                              sampler=database_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=16,
                              # shuffle=True,
                              # drop_last=True,
                              pin_memory=True)
    if world_size > 1:
        return train_loader, train_sampler, test_loader, database_loader
    else:
        return train_loader, test_loader, database_loader
