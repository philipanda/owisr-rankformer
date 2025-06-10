import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import args


def sparse_sum(values, indices0, indices1, n):
    """
    Sums up the `values` at indices in `indices1`.
    If `indices1` repeat then the values at those indices are sumed up.

    if `indices0` is not None, then it is used to collect a subset of indices of `values`
    to be sumed up using `indices1`. Values at `indices0` can repeat too, causing the values
    to be repeated after applying `indices0` to them.

    The output vector is of size `[n, ...]`, where `n` has to be at least `max(indices1)+1`.
    The output vector is initially full of zeros, but then the `values`
    or `values[indices0]` (if indices0 is not None) are added to the zeroes.
    pseudocode:

    if indices0 is not None:
        selected_values = values[indices0]
    else:
        selected_values = values

    result = zeros(n)
    for idx, i in enumerate(indices1):
        result[i] += selected_values[idx]
    """
    if indices0 is None:
        # indices1 is one-dimensional and as long as values
        assert (len(indices1.shape) == 1 and values.shape[0] == indices1.shape[0])
    else:
        # both indices0 and indices1 are one-dimensional and the same length
        assert (len(indices0.shape) == 1 and len(indices1.shape) == 1)
        assert (indices0.shape[0] == indices1.shape[0])
    # assert (len(values.shape) <= 2)

    if indices0 is not None:
        values = values[indices0]
    # initialize with zeros
    result = torch.zeros([n]+list(values.shape)[1:], device=values.device, dtype=values.dtype)
    result = result.index_add(0, indices1, values if indices0 is None else values[indices0])
    return result
            

def rest_sum(values, indices0, indices1, n):
    return values.sum(0).unsqueeze(0)-sparse_sum(values, indices0, indices1, n)


# Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, dataset, alpha=1.0, beta=0.0):
        super(GCN, self).__init__()
        self.dataset = dataset
        self.alpha, self.beta = alpha, beta

    def forward(self, x, u, i):
        n, m = self.dataset.num_users, self.dataset.num_items
        du = sparse_sum(torch.ones_like(i), None, u, n).clamp(1)
        di = sparse_sum(torch.ones_like(i), None, i, m).clamp(1)
        w1 = (torch.ones_like(u)/du[u].pow(self.alpha)/di[i].pow(self.beta)).unsqueeze(-1)
        w2 = (torch.ones_like(u)/du[u].pow(self.beta)/di[i].pow(self.alpha)).unsqueeze(-1)
        xu, xi = torch.split(x, [n, m])
        zu = sparse_sum(xi[i]*w1, None, u, n)
        zi = sparse_sum(xu[u]*w2, None, i, m)
        return torch.concat([zu, zi], 0)


class Rankformer(nn.Module):
    def __init__(self, dataset, alpha):
        super(Rankformer, self).__init__()
        self.dataset = dataset
        self.my_parameters = []
        self.alpha = alpha

    def forward(self, embeddings, interactions_u, interactions_i):
        """
        embeddings - user and item embeddings, joined together into one vector
            shape [n + m, latent_space_vector]
        interactions_u - user ids, for every interaction
        interactions_i - item ids, for every interaction
            u[x], i[x] is the x'th interaction
        
        returns: updated embeddings (x) using the provided interactions
        """

        
        # count of items the user interacted with, and had not interacted with
        interactions_per_user_count = sparse_sum(torch.ones_like(interactions_u), None, interactions_u, self.dataset.num_users)
        items_user_not_interacted_with_count = self.dataset.num_items - interactions_per_user_count

        # the values are clamped to be at least 1, (prevent division by 0?)
        interactions_per_user_count = interactions_per_user_count.clamp(1).unsqueeze(1)
        items_user_not_interacted_with_count = items_user_not_interacted_with_count.clamp(1).unsqueeze(1)

        # `embeddings` is user and item embeddings joined into one vector. Split into user and item embeddings
        # accordint to the count of users and items in the dataset
        normalized_user_embeddings, normalized_item_embeddings = torch.split(F.normalize(embeddings), [self.dataset.num_users, self.dataset.num_items])
        raw_user_embeddings, raw_item_embeddings = torch.split(embeddings, [self.dataset.num_users, self.dataset.num_items])

        # dot product of user and item embeddings for every interaction
        # represents "interaction strength", or the similarity between the user and the item embeddings
        # for every interaction, how similar the user and the item were
        interactions_strength = (normalized_user_embeddings[interactions_u]*normalized_item_embeddings[interactions_i]).sum(1).unsqueeze(1)

        # For each user, sum the embeddings of all items they interacted with
        # Sums item embeddings `normalized_item_embeddings[i]` by user
        sum_interacted_item_norm_embeds_per_user = \
            sparse_sum(normalized_item_embeddings, interactions_i, interactions_u, self.dataset.num_users)
        sum_not_interacted_item_norm_embeds_per_user = \
            normalized_item_embeddings.sum(0)-sum_interacted_item_norm_embeds_per_user

        # Same as above but using the not normalized embeddings
        sum_interacted_item_raw_embeds_per_user = \
            sparse_sum(raw_item_embeddings, interactions_i, interactions_u, self.dataset.num_users)
        sum_not_interacted_item_raw_embeds_per_user = \
            raw_item_embeddings.sum(0) - sum_interacted_item_raw_embeds_per_user

        if args.del_benchmark:
            average_pos_interaction_strength, average_neg_interaction_strength = 0, 0
        else:
            # Average interaction strength per user, used for benchmarking
            average_pos_interaction_strength = \
                (normalized_user_embeddings * sum_interacted_item_norm_embeds_per_user).sum(1).unsqueeze(1) \
                    / interactions_per_user_count
            # for negative interactions = not interacted with
            average_neg_interaction_strength = \
                (normalized_user_embeddings * sum_not_interacted_item_norm_embeds_per_user).sum(1).unsqueeze(1) \
                    / items_user_not_interacted_with_count
            
        # outer products of embeddings
        # embeddings.shape = [m, d]
        # unsqueeze(1) makes the shape [m, 1, d]
        # unsqueeze(2) makes the shape [m, d, 1]
        # xxi = normalized_item_embeddings.unsqueeze(1)*normalized_item_embeddings.unsqueeze(2) # unused?
        item_raw_norm_embedding_outer_product = normalized_item_embeddings.unsqueeze(1)*raw_item_embeddings.unsqueeze(2)
        
        # update signals for user embeddings - encourages similar embeddings for positive interactions
        # dot product of user embeddings and sum of positive interaction item embeddings
        # divided by the count of interactions per user - normalization
        # so its the average similarity of user and item embeddings for items they interacted with
        # - average_neg_interaction_strength removes user bias by showing how much more the positive
        # interactions align compared to negative ones
        # + alpha hyperparameter increases the impact of positive interactions
        user_embeddings_push_towards_positives = \
            (normalized_user_embeddings * sum_interacted_item_norm_embeds_per_user).sum(1).unsqueeze(1) \
                / interactions_per_user_count - average_neg_interaction_strength + self.alpha
        # the same, but for negative interactions - encourages different embeddings for negative interactions
        # `-` at the beginning to result in negative values - similarity for negative interactions shall give negative feedback
        # items and users to have their embeddings distant from each other
        # + alpha hyperparameter reduces the impact of negative interactions
        user_embeddings_push_away_from_negatives = \
            -(normalized_user_embeddings * sum_not_interacted_item_norm_embeds_per_user).sum(1).unsqueeze(1) \
                / items_user_not_interacted_with_count + average_pos_interaction_strength + self.alpha
        
        
        # Aggregate user influence weighted by normalized embeddings per item (positive)
        positive_user_influence = sparse_sum(normalized_user_embeddings / interactions_per_user_count, interactions_u, interactions_i, self.dataset.num_items)
        positive_bias = sparse_sum((-average_neg_interaction_strength + self.alpha) / interactions_per_user_count, interactions_u, interactions_i, self.dataset.num_items)
        item_embeddings_push_towards_positives = (normalized_item_embeddings * positive_user_influence).sum(1).unsqueeze(1) + positive_bias

        # Aggregate user influence for non-interacted (negative) sets
        neg_user_influence = rest_sum(normalized_user_embeddings / items_user_not_interacted_with_count, interactions_u, interactions_i, self.dataset.num_items)
        neg_bias = rest_sum((average_pos_interaction_strength + self.alpha) / items_user_not_interacted_with_count, interactions_u, interactions_i, self.dataset.num_items)
        item_embeddings_push_away_from_negatives = -(normalized_item_embeddings * neg_user_influence).sum(1).unsqueeze(1) + neg_bias
        

        # for each user, sum of (interaction strength × raw item embedding) over their interacted items.
        # Normalized by dividing by the total count of interactions of this user
        # 
        user_vector_towards_positive_items = \
            sparse_sum(interactions_strength * raw_item_embeddings[interactions_i], None, interactions_u, self.dataset.num_users) / interactions_per_user_count

        # adjusts user embeddings toward weighted sums of their interacted item embeddings (weighted by interaction strength),
        # normalized by interaction counts,
        # with a correction term removing negative bias weighted by the sum of raw item embeddings.
        zu1 = (
            user_vector_towards_positive_items
                - (sum_interacted_item_raw_embeds_per_user
                    * (average_neg_interaction_strength - self.alpha)
                    / interactions_per_user_count)
        )

        zu2 = (
            (
            torch.mm(normalized_user_embeddings, (item_raw_norm_embedding_outer_product).sum(0))
            - user_vector_towards_positive_items
            )
                / items_user_not_interacted_with_count
                - sum_not_interacted_item_raw_embeds_per_user
                    * (average_pos_interaction_strength + self.alpha)
                    / items_user_not_interacted_with_count
        )


        zi1 = (
            sparse_sum(
                interactions_strength
                    * raw_user_embeddings[interactions_u]
                    / interactions_per_user_count[interactions_u]
                , None, interactions_i, self.dataset.num_items
                )
            - sparse_sum(
                raw_user_embeddings
                    * (average_neg_interaction_strength - self.alpha)
                    / interactions_per_user_count
                , interactions_u, interactions_i, self.dataset.num_items
                )
        )

        zi2 = (
            torch.mm(normalized_item_embeddings,
                     (
                         ( normalized_user_embeddings / items_user_not_interacted_with_count)
                            .unsqueeze(2)
                            * raw_user_embeddings.unsqueeze(1)
                     ).sum(0)
            )
            - sparse_sum(interactions_strength
                            * (raw_user_embeddings / items_user_not_interacted_with_count)[interactions_u]
                        , None, interactions_i, self.dataset.num_items)
            - rest_sum(raw_user_embeddings
                            * (average_pos_interaction_strength + self.alpha)
                                / items_user_not_interacted_with_count
                        , interactions_u, interactions_i, self.dataset.num_items)
        )
        
        
        # concatenated user and item 
        z1 = torch.concat([zu1, zi1], 0)
        z2 = torch.concat([zu2, zi2], 0)

        d1 = torch.concat([user_embeddings_push_towards_positives, item_embeddings_push_towards_positives], 0).clamp(args.rankformer_clamp_value)
        d2 = torch.concat([user_embeddings_push_away_from_negatives, item_embeddings_push_away_from_negatives], 0).clamp(args.rankformer_clamp_value)
        if args.del_neg:
            z2, d2 = 0, 0
        z, d = z1+z2, d1+d2
        if args.del_omega_norm:
            return z
        
        #print(torch.sum((z/d - self.old_forward(embeddings, interactions_u, interactions_i))**2, 0).sum(0) == 0)
        return(z/d)
        #return self.old_forward(embeddings, interactions_u, interactions_i)


    def old_forward(self, x, u, i):
        n, m = self.dataset.num_users, self.dataset.num_items
        dui = sparse_sum(torch.ones_like(u), None, u, n)
        duj = m-dui
        dui, duj = dui.clamp(1).unsqueeze(1), duj.clamp(1).unsqueeze(1)
        xu, xi = torch.split(F.normalize(x), [n, m])
        vu, vi = torch.split(x, [n, m])
        xui = (xu[u]*xi[i]).sum(1).unsqueeze(1)
        sxi = sparse_sum(xi, i, u, n)
        sxj = xi.sum(0)-sxi
        svi = sparse_sum(vi, i, u, n)
        svj = vi.sum(0)-svi
        b_pos = (xu*sxi).sum(1).unsqueeze(1)/dui
        b_neg = (xu*sxj).sum(1).unsqueeze(1)/duj
        if args.del_benchmark:
            b_pos, b_neg = 0, 0
        xxi = xi.unsqueeze(1)*xi.unsqueeze(2)
        xvi = xi.unsqueeze(1)*vi.unsqueeze(2)
        du1 = (xu*sxi).sum(1).unsqueeze(1)/dui-b_neg+self.alpha
        du2 = -(xu*sxj).sum(1).unsqueeze(1)/duj+b_pos+self.alpha
        di1 = (xi*sparse_sum(xu/dui, u, i, m)).sum(1).unsqueeze(1)+sparse_sum((-b_neg+self.alpha)/dui, u, i, m)
        di2 = -(xi*rest_sum(xu/duj, u, i, m)).sum(1).unsqueeze(1)+rest_sum((b_pos+self.alpha)/duj, u, i, m)
        A = sparse_sum(xui*vi[i], None, u, n)
        zu1 = A/dui-svi*(b_neg-self.alpha)/dui
        zu2 = (torch.mm(xu, (xvi).sum(0))-A)/duj-svj*(b_pos+self.alpha)/duj
        zi1 = sparse_sum(xui*vu[u]/dui[u], None, i, m)-sparse_sum(vu*(b_neg-self.alpha)/dui, u, i, m)
        zi2 = torch.mm(xi, ((xu/duj).unsqueeze(2)*vu.unsqueeze(1)).sum(0))-sparse_sum(xui*(vu/duj)[u], None, i, m) \
            - rest_sum(vu*(b_pos+self.alpha)/duj, u, i, m)
        z1 = torch.concat([zu1, zi1], 0)
        z2 = torch.concat([zu2, zi2], 0)
        d1 = torch.concat([du1, di1], 0).clamp(args.rankformer_clamp_value)
        d2 = torch.concat([du2, di2], 0).clamp(args.rankformer_clamp_value)
        if args.del_neg:
            z2, d2 = 0, 0
        z, d = z1+z2, d1+d2
        if args.del_omega_norm:
            return z
        return z/d