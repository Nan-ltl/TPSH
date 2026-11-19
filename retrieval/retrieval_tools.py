import torch

from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np

def calculate_map(args, test_loader, database_loader, model, topk = -1):
    model.to(args.device)
    model.eval()
    with torch.no_grad():

        query_images_hash_bit = torch.empty(1, args.hash_bit).to(args.device)

        query_labels = torch.empty(1).to(args.device)
        test_iterator = tqdm(test_loader,
                              desc = "Query data... ",
                              bar_format = "{l_bar}{bar}{r_bar}",
                              dynamic_ncols = True,
                              leave = False,
                              position = 0,
                              total = len(test_loader),
                              disable=torch.cuda.current_device() != 0)
        for batch in test_iterator:
            x, y, z = batch
            images = x.to(args.device)
            labels = y.to(args.device)

            query_batch = torch.sign(model(images, retrieval=True)).to(args.device)
            query_images_hash_bit = torch.cat((query_images_hash_bit,query_batch), dim = 0)
            query_labels = torch.cat((query_labels, labels),dim = 0)

        query_images_hash_bit = query_images_hash_bit[1:]
        query_labels = query_labels[1:]

        train_iterator = tqdm(database_loader,
                              desc="Retrieval data... ",
                              bar_format="{l_bar}{bar}{r_bar}",
                              dynamic_ncols=True,
                              leave=False,
                              position = 0,
                              total = len(database_loader),
                              disable=torch.cuda.current_device() != 0)
        retrieval_images_hash_bit = torch.empty(1, args.hash_bit).to(args.device)
        retrieval_labels = torch.empty(1).to(args.device)

        for batch in train_iterator:
            x, y, z = batch
            images = x.to(args.device)
            labels = y.to(args.device)


            u_hash =model(images, retrieval=True)
            u_hash = torch.sign(u_hash)
            retrieval_images_hash_bit = torch.cat((retrieval_images_hash_bit,u_hash), dim = 0)
            retrieval_labels = torch.cat((retrieval_labels, labels),dim = 0)

        retrieval_images_hash_bit = retrieval_images_hash_bit[1:] #[train_number, bit]
        retrieval_labels = retrieval_labels[1:] #[train_number]

        hamming = (retrieval_images_hash_bit.shape[1]- torch.mm(query_images_hash_bit,retrieval_images_hash_bit.t())) / 2 #[query_number, retrieval_number]
        cat_hamming = torch.cat((hamming.unsqueeze(dim=-1),retrieval_labels.unsqueeze(0).unsqueeze(-1).repeat(hamming.shape[0],1,1)),dim=-1)
        ap_list = torch.empty(1).to(args.device)
        for i in tqdm(range(cat_hamming.shape[0]),
                      dynamic_ncols=True,
                      leave=False,
                      disable=torch.cuda.current_device() != 0):
            single_hamming = cat_hamming[i, :]
            indicate = torch.argsort(single_hamming[:, 0])
            rank_labels = single_hamming[:, 1][indicate]
            rank_labels = rank_labels[:topk]
            true_number = (rank_labels == query_labels[i]).sum()
            best = torch.linspace(1, true_number.item(), true_number.item()).to(args.device)
            result = (rank_labels == query_labels[i].expand(rank_labels.shape[0]))
            tindex = (torch.nonzero(result == True, as_tuple=False).squeeze(dim = -1) + 1.0).float()
            if best.shape[0] == tindex.shape[0]:
                ap = (best / tindex).mean().unsqueeze(0)
                ap_list = torch.cat((ap_list,ap), dim = 0)
            else:
                raise ValueError("Doesn't match!")
        ap_list = ap_list[1:]
        map = ap_list.mean().item()
        return map


if __name__=="__main__":
    x = torch.empty(1)
    print(x)
    x = torch.ones((4,13,768))
    print(x[:,0,:].shape)
    query_out = torch.tensor([1,-1,-1,1,1,1,1,1,-1,-1,-1])
    u_hash =    torch.tensor([[1,-1,-1,1,1,1,1,1,-1,-1,-1],

                              [0, 0, 0,0,0,0,0,0, 0, 0, 0],
                              [1,-1,-1,1,1,1,1,1,-1,-1,-1]])
    y = torch.tensor([2,2,4])
    query_out = query_out.unsqueeze(dim=0)
    hamming_dist = (query_out.shape[1] - torch.mm(u_hash, query_out.t())) / 2
    hamming_dist = hamming_dist.to(torch.device('cpu'))
    hamming_dist = torch.cat((hamming_dist, y.unsqueeze(dim=-1)), dim=-1)  # [batch, 2]
    sorted_indices = torch.from_numpy(np.argsort(hamming_dist[:,0].numpy(), kind='stable'))
    x=0













