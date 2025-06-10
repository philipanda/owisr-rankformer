from parse import args
import torch
from torch import nn
import torch.nn.functional as F
import rec
from rec import GCN, Rankformer
import numpy as np


def InfoNCE(x, y, tau=0.15, b_cos=True):
    if b_cos:
        x, y = F.normalize(x), F.normalize(y) # normalizes to unit length
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()


def test(is_pred_right, true_interactions, n_true_interactions_per_user):
    """
    pred - predicted interactions, K items for every user as x*col_size+y values, where x is the user and y the item
    test - real interactions, in x*col_size+y vector where x is the user and y is the item
    recall_n - amount of true interactions for every user in test
    """
    # assert sum(recall_n).item() == torch.numel(test) # recall_n - amount of true interactions for every user in test

    is_pred_right = torch.isin(is_pred_right[n_true_interactions_per_user > 0], true_interactions) # array of 0/1 - whether the prediction at given index 
        # is in the true items
        # recall_n > 0 means that only the preds where the user has more than 0 interactions in the test
        # set are considered
        
    n_true_interactions_per_user = n_true_interactions_per_user[n_true_interactions_per_user > 0]
    pre, recall, ndcg, rr, ap = [], [], [], [], []
    for k in args.topks: # can be tested on multiple top-K values
        right_pred_count = is_pred_right[:, :k].sum(1) # up to k right predictions for every user
        recall_k = n_true_interactions_per_user.clamp(max=k) # clamp the amount of interactions of every user to K

        # precision
        pre.append((right_pred_count/k).sum()) # number of right predictions over predictions count

        # recall
        recall.append((right_pred_count/recall_k).sum()) # number of right predictions over how many interactions
            # of the user were in the test set (clamped to K)

        # ndcg
        dcg = (is_pred_right[:, :k]/torch.arange(2, k+2, device=args.device).unsqueeze(0).log2()).sum(1) # relevance of every prediction,
            # falling with the index
        d_val = (1/torch.arange(2, k+2, device=args.device).log2()).cumsum(0) # max possible relevance??
        idcg = d_val[recall_k-1] # ideal dcg
        ndcg.append((dcg / idcg).sum())

        # RR@K
        rr_scores = torch.zeros_like(right_pred_count, dtype=torch.float, device=args.device) # init reciprocal rank for every prediction with 0 
        for i in range(is_pred_right.shape[0]): # for every user
            hits = is_pred_right[i, :k].nonzero() # indices of true predictions, indices of non-zero elements, shape[numel, dimensions]
            if hits.numel() > 0: # number of true predictions
                rr_scores[i] = 1.0 / (hits[0].item() + 1) # hits[0] is the index of first prediction that appears in test - the highest ranked
                # in MRR only the first correct prediction is considered
                # rank[i] = 1/index of items that were predicted and existed in test set
        rr.append(rr_scores.sum())

        # AP@K
        ap_scores = torch.zeros_like(right_pred_count, dtype=torch.float, device=args.device)
        for i in range(is_pred_right.shape[0]): # for every user
            hits = is_pred_right[i, :k].nonzero() # indices of true predictions
            if hits.numel() == 0:
                continue
            precisions = torch.tensor(
                [(is_pred_right[i, :hit_idx + 1].sum().item()) / (hit_idx + 1) for hit_idx in hits], # how many of the predictions up to k are right, for k in [1,K]
                device=args.device
            )
            ap_scores[i] = precisions.mean() # mean over the predictions which were right, up to some k in [1,K], not all predictions
        ap.append(ap_scores.sum()) # sum of averages for users, should be divided by number of users to get MAP


    return n_true_interactions_per_user.shape[0], \
        torch.tensor(pre, device=args.device), \
        torch.tensor(recall, device=args.device), \
        torch.tensor(ndcg, device=args.device), \
        torch.tensor(rr, device=args.device), \
        torch.tensor(ap, device=args.device)

def negative_sampling(u, i, m):
    """
    u - list of users
    i - list of items
        - (u[x], i[x]) is one user-item interaction
    m - number of items in the whole dataset

    creates a new items vector so that no user-item pair is repeated
    from the original u and i
    u and i hold interactions: u[idx] interacts with i[idx]

    the result: j is negative items, so that
    (u[idx], j[idx]) != (u[idx], i[idx]) for all idx
    """
    edge_id = u*m+i
    assert u.shape == i.shape
    j = torch.randint_like(u, 0, m) # random numbers from 0 to m in shape of i
    # j - random choice of items
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

        # creates embeddings of size self.hidden_dim for users and items
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim, device=args.device)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim, device=args.device)
        # initializes the embedding layers with a normal distribution
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]

        self.layers = []

        # creates the GCN and Rankformer no matter if they are used or not
        self.GCN = GCN(dataset, args.gcn_left, args.gcn_right)
        self.Rankformer = Rankformer(dataset, args.rankformer_alpha)

        self._user_embeddings, self._item_embeddings = None, None
        self._users_cl_embeddings, self._items_cl_embeddings = None, None # contrast learning embeddings

        # adam optimizer, why is it a part of the model??
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)
        
        # loss function defined in the class, again, why inside the model, weird
        self.loss_func = self.loss_bpr

    def computer(self):
        """
        something like a model step? Updates the embeddings using the whole train dataset
        """
        interaction_users, interaction_items = self.dataset.train_user, self.dataset.train_item
        user_embeddings, item_embeddings = self.embedding_user.weight, self.embedding_item.weight
        embeddings = torch.cat([user_embeddings, item_embeddings]) # join the two embeddings, [1,2]+[3,4] = [1,2,3,4]
        cl_embeddings = embeddings # cl - contrast learning

        # applies the GCN and Rankformer layers
        if args.use_gcn:
            updated_embeddings = [embeddings] # history of embeddings changing when traversing the GCN
            for _ in range(args.gcn_layers):
                embeddings = self.GCN(embeddings, interaction_users, interaction_items) # updates the embeddings
                if args.use_cl: 
                    random_noise = torch.rand_like(embeddings)
                    embeddings += torch.sign(embeddings)*F.normalize(random_noise, dim=-1)*args.cl_eps # adds random vector to the embeddings with length cl_eps
                if _ == args.cl_layer-1:
                    cl_embeddings = embeddings
                updated_embeddings.append(embeddings)
            if args.gcn_mean:
                embeddings = torch.stack(updated_embeddings, dim=-1).mean(-1)

        if args.use_rankformer:
            for _ in range(args.rankformer_layers):
                rec_emb = self.Rankformer(embeddings, interaction_users, interaction_items)
                embeddings = (1-args.rankformer_tau)*embeddings+args.rankformer_tau*rec_emb

        # set the temporary embeddings: split-back the new embeddings into user and item embeddings
        self._user_embeddings, self._item_embeddings = torch.split(embeddings, [self.dataset.num_users, self.dataset.num_items])
        self._users_cl_embeddings, self._items_cl_embeddings = torch.split(cl_embeddings, [self.dataset.num_users, self.dataset.num_items])

    def evaluate(self, true_interactions_batches, n_interactions_per_user):
        """
        - `true_interactions_batches` - list of batches interactions, args.batch_size unique users in each (last one might have less)
            number of actual interactions might vary by a lot, thats why n_interactions_per_user is needed.
            true_interactions_batches contains pairs (user,item) which are the True interactions, used to evaluate.
            It is a paremeter and not global to allow giving either validation or testing set

        - `n_interactions_per_user` - amount of interactions for every user.
            Knowing that batch x contains users A and B the amount of interactions in the batch
            is n_interactions_per_user[A] + n_interactions_per_user[B]
        """
        
        self.eval()
        if self._user_embeddings is None: # create the initial embeddings if they are None?
            self.computer()

        # metrics for every amount of '@K' to be tested
        max_K = max(args.topks)
        sum_pre = torch.zeros(len(args.topks), device=args.device)
        sum_recall = torch.zeros(len(args.topks), device=args.device)
        sum_ndcg = torch.zeros(len(args.topks), device=args.device)
        sum_rr = torch.zeros(len(args.topks), device=args.device)
        sum_ap = torch.zeros(len(args.topks), device=args.device)
        sum_interactions = 0
        with torch.no_grad():

            for batch_users, batch_train, ground_true in zip(self.dataset.batch_users, self.dataset.train_batches, true_interactions_batches):
                user_e = self._user_embeddings[batch_users]
                rating = torch.mm(user_e, self._item_embeddings.t())
                for i, user in enumerate(batch_users):
                    train_items = batch_train[batch_train[:, 0] == user][:, 1]
                    rating[i, train_items] = float('-inf')
                _, pred_items = torch.topk(rating, k=max_K) # (values, indices), the rating does not matter, only if its the largest

                # pred_items is the K top rated items for every user in batch
                # the users, items arrays are reshaped into a single dimension vector
                # so that cell[x,y] = vector[x*col_size + y], the vector contains the index numbers in the result vector
                # print(batch_users.unsqueeze(1))
                # print(pred_items)
                # print(batch_users.unsqueeze(1)*self.dataset.num_items+pred_items)

                interactions_count, precision, recall, ndcg, rr, ap = test(
                    batch_users.unsqueeze(1)*self.dataset.num_items+pred_items, # for every user K predictions of items as x*col_size + y
                    ground_true[:, 0]*self.dataset.num_items+ground_true[:, 1], # true ???
                    n_interactions_per_user[batch_users])
                
                sum_pre += precision
                sum_recall += recall
                sum_ndcg += ndcg
                sum_rr += rr
                sum_ap += ap
                sum_interactions += interactions_count
            mean_pre = sum_pre /  sum_interactions
            mean_recall = sum_recall / sum_interactions
            mean_ndcg = sum_ndcg / sum_interactions
            mean_rr = sum_rr / sum_interactions
            mean_ap = sum_ap / sum_interactions
        return mean_pre, mean_recall, mean_ndcg, mean_rr, mean_ap

    # evaluates the model on the validation set
    def valid_func(self):
        return self.evaluate(self.dataset.valid_batches, self.dataset.valid_n_interactions_per_user)

    # evaluates the model on the test set
    def test_func(self):
        return self.evaluate(self.dataset.test_batches, self.dataset.test_n_interactions_per_user)


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

    # u true user
    # i true item
    # j negative item
    def loss_bpr(self, u, i):
        j = negative_sampling(u, i, self.dataset.num_items)
        u_emb0, u_emb = self.embedding_user(u), self._user_embeddings[u]
        i_emb0, i_emb = self.embedding_item(i), self._item_embeddings[i]
        j_emb0, j_emb = self.embedding_item(j), self._item_embeddings[j]
        scores_ui = torch.sum(torch.mul(u_emb, i_emb), dim=-1)
        scores_uj = torch.sum(torch.mul(u_emb, j_emb), dim=-1)
        loss = torch.mean(F.softplus(scores_uj-scores_ui)) # SoftPlus is a smooth approximation to the ReLU function, RelU but smooth and differentiable
        reg_loss = (1/2)*(u_emb0.norm(2).pow(2)+i_emb0.norm(2).pow(2)+j_emb0.norm(2).pow(2))/float(u.shape[0])
        if args.use_cl:
            all_user, all_item = self._user_embeddings, self._item_embeddings
            cl_user, cl_item = self._users_cl_embeddings, self._items_cl_embeddings
            u_idx, i_idx = torch.unique(u), torch.unique(i)
            cl_loss = InfoNCE(all_user[u_idx], cl_user[u_idx], args.cl_tau)+InfoNCE(all_item[i_idx], cl_item[i_idx], args.cl_tau)
            return loss+args.reg_lambda*reg_loss+args.cl_lambda*cl_loss
        return loss+args.reg_lambda*reg_loss
    
    def loss_softmax(self, u, i):
        j = negative_sampling(u, i, self.dataset.num_items)
        u_emb0, u_emb = self.embedding_user(u), self._user_embeddings[u]
        i_emb0, i_emb = self.embedding_item(i), self._item_embeddings[i]
        j_emb0, j_emb = self.embedding_item(j), self._item_embeddings[j]

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

    def train_func_batch(self):
        train_losses = []
        train_user = self.dataset.train_user
        train_item = self.dataset.train_item
        for _ in range(0, train_user.shape[0], args.loss_batch_size):
            self.computer()
            train_loss = self.loss_func(train_user[_:_+args.loss_batch_size], train_item[_:_+args.loss_batch_size])
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
