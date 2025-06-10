import torch
from torch.utils.data import Dataset
import pandas as pd
from parse import args
import numpy as np
import torch.nn.functional as F
import os

class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device


        # train dataset
        # read users and items into two arrays: self.valid_data and self.valid_user where valid_data[i], valid_user[i] = interaction
        # naturally the values in both arrays can repeat as one user can interact with multiple items and vice versa
        print(os.path.abspath(train_file))
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_data = train_data[torch.argsort(train_data[:, 0]), :]
        self.train_data.sort(dim=0)
        self.train_user, self.train_item = self.train_data[:, 0], self.train_data[:, 1]

        # valid dataset, the same as train
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_data = valid_data[torch.argsort(valid_data[:, 0]), :]
        self.valid_user, self.valid_item = self.valid_data[:, 0], self.valid_data[:, 1]

        # test dataset, the same as train and valid
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_data = test_data[torch.argsort(test_data[:, 0]), :]
        self.test_user, self.test_item = self.test_data[:, 0], self.test_data[:, 1]

        # statistics, user/item IDs start from 0, so the amount is ID_max + 1
        self.num_users = max(self.train_user.max(), self.valid_user.max(), self.test_user.max()).cpu()+1
        self.num_items = max(self.train_item.max(), self.valid_item.max(), self.test_item.max()).cpu()+1
        self.num_nodes = self.num_users+self.num_items # number of interactions

        print(f'{self.num_users:d} users, {self.num_items:d} items.')
        print(f'train: {self.train_user.shape[0]:d}, valid: {self.valid_user.shape[0]:d}, test: {self.test_user.shape[0]:d}.')
        self.build_batch()

    def build_batch(self):

        # how many interactions for each user
        # self.train_n_interactions_per_user = torch.zeros(self.num_users).long().to(args.device) # init zeros for every user
        # # for every user counts how many times it appears in interactions
        # self.train_n_interactions_per_user = self.train_n_interactions_per_user.index_add(0, self.train_user, torch.ones_like(self.train_user))
        # replace with much simpler way of doing exactly the same - counting how many interaction each user has
        # 
        # the same operation on valid and test sets
        # self.test_n_interactions_per_user = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.test_user, torch.ones_like(self.test_user))
        # self.valid_n_interactions_per_user = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.valid_user, torch.ones_like(self.valid_user))
        
        def count_interactions(users, pad_to_size):
            counts = users.bincount().to(self.device)
            paded_with_zeros = torch.nn.functional.pad(counts, (0, pad_to_size - counts.size(0)), value=0).to(self.device)
            return paded_with_zeros
        
        self.train_n_interactions_per_user = count_interactions(self.train_user, self.num_users)
        self.valid_n_interactions_per_user = count_interactions(self.valid_user, self.num_users)
        self.test_n_interactions_per_user = count_interactions(self.test_user, self.num_users)

        #self.batch_users = [torch.arange(i, min(i+args.test_batch_size, self.num_users)).to(args.device) for i in range(0, self.num_users, args.test_batch_size)]

        # split all unique users in the world into batches, with no respect to train/valid/test split
        self.batch_users = torch.split(torch.arange(0, self.num_users, device=self.device), args.test_batch_size)
        # split into batches so that every batch of every split (train/valid/test) contains the same users
        
        # split into pieces, so that for every batch of users from self.batch_users:
        #  for every user in batch:
        #    user 0: get train-data[0 : len(user0 interactions)]
        #    user 1: get train-data[len(user0 interactions) : len(user0 interactions) + len(user1 interactions)]\
        #    etc
        # this assumes that the train_data contains all the interactions sorted by user
        # so that the result is the interactions performed by the users from a user batch for every batch
        def split_into_user_batches(data, interactions_per_user, user_batches):
            sizes = [interactions_per_user[user_batch].sum() for user_batch in user_batches]
            return list(data.split(sizes))

        # list of batches of pairs (user,item)
        # [ [ (1,2), (4,5), (0,1), ... ], [...], ... ]
        self.train_batches = split_into_user_batches(self.train_data, self.train_n_interactions_per_user, self.batch_users)
        self.test_batches = split_into_user_batches(self.test_data, self.test_n_interactions_per_user, self.batch_users)
        self.valid_batches = split_into_user_batches(self.valid_data, self.valid_n_interactions_per_user, self.batch_users)


dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
