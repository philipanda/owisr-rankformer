from parse import args
import torch
from torch import nn
import torch.nn.functional as F
import rec
from rec import GCN, Rankformer
import numpy as np


def InfoNCE(x, y, tau=0.15, b_cos=True):
    if b_cos:
        x, y = F.normalize(x), F.normalize(y)
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()


def test(pred, test, recall_n):
    pred = torch.isin(pred[recall_n > 0], test)
    recall_n = recall_n[recall_n > 0]
    pre, recall, ndcg, rr, ap = [], [], [], [], []
    for k in args.topks:
        right_pred = pred[:, :k].sum(1)
        recall_k = recall_n.clamp(max=k)
        # precision
        pre.append((right_pred/k).sum())
        # recall
        recall.append((right_pred/recall_k).sum())
        # ndcg
        dcg = (pred[:, :k]/torch.arange(2, k+2).to(args.device).unsqueeze(0).log2()).sum(1)
        d_val = (1/torch.arange(2, k+2).to(args.device).log2()).cumsum(0)
        idcg = d_val[recall_k-1]
        ndcg.append((dcg / idcg).sum())

        # RR@K
        rr_scores = torch.zeros_like(right_pred, dtype=torch.float, device=args.device) # init reciprocal rank for every prediction with 0 
        for i in range(pred.shape[0]): # for every user
            hits = pred[i, :k].nonzero() # indices of true predictions, indices of non-zero elements, shape[numel, dimensions]
            if hits.numel() > 0: # number of true predictions
                rr_scores[i] = 1.0 / (hits[0].item() + 1) # hits[0] is the index of first prediction that appears in test - the highest ranked
                # in MRR only the first correct prediction is considered
                # rank[i] = 1/index of items that were predicted and existed in test set
        rr.append(rr_scores.sum())

        # AP@K
        ap_scores = torch.zeros_like(right_pred, dtype=torch.float, device=args.device)
        for i in range(pred.shape[0]): # for every user
            hits = pred[i, :k].nonzero() # indices of true predictions
            if hits.numel() == 0:
                continue
            precisions = torch.tensor(
                [(pred[i, :hit_idx + 1].sum().item()) / (hit_idx + 1) for hit_idx in hits], # how many of the predictions up to k are right, for k in [1,K]
                device=args.device
            )
            ap_scores[i] = precisions.mean() # mean over the predictions which were right, up to some k in [1,K], not all predictions
        ap.append(ap_scores.sum()) # sum of averages for users, should be divided by number of users to get MAP
    return recall_n.shape[0], torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg), torch.tensor(rr), torch.tensor(ap)


def multi_negative_sampling(u, i, m, k):
    edge_id = u*m+i
    j = torch.randint(0, m, (i.shape[0], k)).to(u.device)
    mask = torch.isin(u.unsqueeze(1)*m+j, edge_id)
    while mask.sum() > 0:
        j[mask] = torch.randint_like(j[mask], 0, m)
        mask = torch.isin(u.unsqueeze(1)*m+j, edge_id)
    return j


def negative_sampling(u, i, m):
    edge_id = u*m+i
    j = torch.randint_like(i, 0, m)
    mask = torch.isin(u*m+j, edge_id)
    while mask.sum() > 0:
        j[mask] = torch.randint_like(j[mask], 0, m)
        mask = torch.isin(u*m+j, edge_id)
    return j


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = args.hidden_dim
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]
        self.layers = []
        self.GCN = GCN(dataset, args.gcn_left, args.gcn_right)
        self.Rankformer = Rankformer(dataset, args.rankformer_alpha)
        self._users, self._items, self._users_cl, self._items_cl = None, None, None, None
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)
        if args.softmax:
            self.loss_func = self.loss_softmax
        else:
            self.loss_func = self.loss_bpr

    def computer(self):
        u, i = self.dataset.train_user, self.dataset.train_item
        users_emb, items_emb = self.embedding_user.weight, self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        emb_cl = all_emb
        if args.use_gcn:
            embs = [all_emb]
            for _ in range(args.gcn_layers):
                all_emb = self.GCN(all_emb, u, i)
                if args.use_cl:
                    random_noise = torch.rand_like(all_emb)
                    all_emb += torch.sign(all_emb)*F.normalize(random_noise, dim=-1)*args.cl_eps
                if _ == args.cl_layer-1:
                    emb_cl = all_emb
                embs.append(all_emb)
            if args.gcn_mean:
                all_emb = torch.stack(embs, dim=-1).mean(-1)
        if args.use_rankformer:
            for _ in range(args.rankformer_layers):
                rec_emb = self.Rankformer(all_emb, u, i)
                all_emb = (1-args.rankformer_tau)*all_emb+args.rankformer_tau*rec_emb
        self._users, self._items = torch.split(all_emb, [self.dataset.num_users, self.dataset.num_items])
        self._users_cl, self._items_cl = torch.split(emb_cl, [self.dataset.num_users, self.dataset.num_items])

    def evaluate(self, test_batch, test_degree):
        self.eval()
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items
        max_K = max(args.topks)
        all_pre = torch.zeros(len(args.topks))
        all_recall = torch.zeros(len(args.topks))
        all_ndcg = torch.zeros(len(args.topks))
        all_rr = torch.zeros(len(args.topks))
        all_ap = torch.zeros(len(args.topks))
        all_cnt = 0
        with torch.no_grad():
            for batch_users, batch_train, ground_true in zip(self.dataset.batch_users, self.dataset.train_batch, test_batch):
                user_e = user_emb[batch_users]
                rating = torch.mm(user_e, item_emb.t())
                rating[batch_train[:, 0]-batch_users[0], batch_train[:, 1]] = -(1 << 10)
                _, pred_items = torch.topk(rating, k=max_K)
                cnt, pre, recall, ndcg, rr, ap = test(
                    batch_users.unsqueeze(1)*self.dataset.num_items+pred_items,
                    ground_true[:, 0]*self.dataset.num_items+ground_true[:, 1],
                    test_degree[batch_users])
                all_pre += pre
                all_recall += recall
                all_ndcg += ndcg
                all_cnt += cnt
                all_rr += rr
                all_ap += ap
            all_pre /= all_cnt
            all_recall /= all_cnt
            all_ndcg /= all_cnt
            all_rr /= all_cnt
            all_ap /= all_cnt
        return all_pre, all_recall, all_ndcg, all_rr, all_ap

    def valid_func(self):
        return self.evaluate(self.dataset.valid_batch, self.dataset.valid_degree)

    def test_func(self):
        return self.evaluate(self.dataset.test_batch, self.dataset.test_degree)

    def train_func(self):
        self.train()
        if args.loss_batch_size == 0:
            return self.train_func_one_batch(self.dataset.train_user, self.dataset.train_item)
        train_losses = []
        shuffled_indices = torch.randperm(self.dataset.train_user.shape[0], device=args.device)
        train_user = self.dataset.train_user[shuffled_indices]
        train_item = self.dataset.train_item[shuffled_indices]
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            train_losses.append(self.train_func_one_batch(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size]))
        return torch.stack(train_losses).mean()

    def train_func_one_batch(self, u, i):
        self.computer()
        train_loss = self.loss_func(u, i)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        # memory_allocated = torch.cuda.max_memory_allocated(args.device)
        # print(f"Max memory allocated after backward pass: {memory_allocated} bytes = {memory_allocated/1024/1024:.4f} MB = {memory_allocated/1024/1024/1024:.4f} GB.")
        return train_loss

    def loss_softmax(self, u, i):
        j = negative_sampling(u, i, self.dataset.num_items)
        u_emb0, u_emb = self.embedding_user(u), self._users[u]
        i_emb0, i_emb = self.embedding_item(i), self._items[i]
        j_emb0, j_emb = self.embedding_item(j), self._items[j]

        pos_scores = torch.sum(u_emb * i_emb, dim=-1, keepdim=True)  # [B, 1]
        neg_scores = torch.sum(u_emb * j_emb, dim=-1, keepdim=True)  # [B, 1]
        logits = torch.cat([pos_scores, neg_scores], dim=1)          # [B, 2]
        labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)  # Positive = class 0

        loss = F.cross_entropy(logits, labels)

        reg_loss = 0.5 * (u_emb0.norm(2).pow(2) + i_emb0.norm(2).pow(2) + j_emb0.norm(2).pow(2)) / float(u.shape[0])

        if args.use_cl:
            all_user, all_item = self._user_embeddings, self._item_embeddings
            cl_user, cl_item = self._users_cl_embeddings, self._items_cl_embeddings
            u_idx, i_idx = torch.unique(u), torch.unique(i)
            cl_loss = InfoNCE(all_user[u_idx], cl_user[u_idx], args.cl_tau) + \
                    InfoNCE(all_item[i_idx], cl_item[i_idx], args.cl_tau)
            return loss + args.reg_lambda * reg_loss + args.cl_lambda * cl_loss

        return loss + args.reg_lambda * reg_loss

    def loss_bpr(self, u, i):
        j = negative_sampling(u, i, self.dataset.num_items)
        u_emb0, u_emb = self.embedding_user(u), self._users[u]
        i_emb0, i_emb = self.embedding_item(i), self._items[i]
        j_emb0, j_emb = self.embedding_item(j), self._items[j]
        scores_ui = torch.sum(torch.mul(u_emb, i_emb), dim=-1)
        scores_uj = torch.sum(torch.mul(u_emb, j_emb), dim=-1)
        loss = torch.mean(F.softplus(scores_uj-scores_ui))
        reg_loss = (1/2)*(u_emb0.norm(2).pow(2)+i_emb0.norm(2).pow(2)+j_emb0.norm(2).pow(2))/float(u.shape[0])
        if args.use_cl:
            all_user, all_item = self._users, self._items
            cl_user, cl_item = self._users_cl, self._items_cl
            u_idx, i_idx = torch.unique(u), torch.unique(i)
            cl_loss = InfoNCE(all_user[u_idx], cl_user[u_idx], args.cl_tau)+InfoNCE(all_item[i_idx], cl_item[i_idx], args.cl_tau)
            return loss+args.reg_lambda*reg_loss+args.cl_lambda*cl_loss
        return loss+args.reg_lambda*reg_loss

    def train_func_batch(self):
        train_losses = []
        train_user = self.dataset.train_user
        train_item = self.dataset.train_item
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            self.computer()
            train_loss = self.loss_bpr(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size])
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_losses.append(train_loss)
        return torch.stack(train_losses).mean()

    def save_emb(self):
        torch.save(self.embedding_user, f'../saved/{args.data:s}_user.pt')
        torch.save(self.embedding_item, f'../saved/{args.data:s}_item.pt')

    def load_emb(self):
        self.embedding_user = torch.load(f'../saved/{args.data:s}_user.pt').to(args.device)
        self.embedding_item = torch.load(f'../saved/{args.data:s}_item.pt').to(args.device)