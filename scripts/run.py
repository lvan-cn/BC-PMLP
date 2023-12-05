import pickle
import numpy as np
import sys
import os
import torch
from tqdm.auto import tqdm

from time import time
from Dataset import LoadDataset
from BC_PMLP import REG, CLA
from random import randint
from math import log
from torch.utils.data import DataLoader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BC(object):

    def __init__(self, num_user, num_item, user_records, user_emb, item_emb):
        # for Netflix
        self.lr = 0.001
        self.maxIter = 160
        self.regulation = 0.0001

        self.P = user_emb
        self.Q = item_emb
        self.PositiveSet = user_records
        self.user_num = num_user
        self.item_num = num_item

    def train_model(self):

        for epoch in range(self.maxIter):
            t = time()
            self.loss = 0
            with tqdm(total=len(self.PositiveSet)) as progress:
                for u in self.PositiveSet:
                    ure = self.PositiveSet[u]
                    # pair_num = max(int(self.item_num / (0.4 * (len(ure) ** 2))), 1)
                    for i in ure:
                        # for pair in range(pair_num):
                        j = randint(0, self.item_num - 1)
                        while j in ure:
                            j = randint(0, self.item_num - 1)
                        v = randint(0, self.user_num - 1)

                        s = sigmoid(np.sum(self.P[u] * self.Q[i]) - np.sum(self.P[u] * self.Q[j]))
                        r = sigmoid(np.sum(self.P[u] * self.Q[i]) - np.sum(self.P[v] * self.Q[i]))
                        self.P[u] += self.lr * ((1 - s) * (self.Q[i] - self.Q[j]) + (1 - r) * self.Q[i])
                        self.Q[i] += self.lr * ((1 - s) * self.P[u] + (1 - r) * (self.P[u] - self.P[v]))
                        self.Q[j] -= self.lr * (1 - s) * self.P[u]
                        self.P[v] -= self.lr * (1 - r) * self.Q[i]
                        self.P[u] += self.lr * self.regulation * self.P[u]
                        self.Q[i] += self.lr * self.regulation * self.Q[i]
                        self.Q[j] += self.lr * self.regulation * self.Q[j]
                        self.P[v] += self.lr * self.regulation * self.P[v]
                        self.loss += -log(s) - log(r)
                    progress.update(1)
                self.loss += 0.5 * self.regulation * (self.P * self.P).sum() + 0.5 * self.regulation * (
                        self.Q * self.Q).sum()

            print("Finish %d epoch in time: %.2f, loss: %.4f" % (epoch, time() - t, self.loss))

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        user = self.P[u]
        return [np.dot(user, item) for item in self.Q]


def evaluate(ranklist, ground):
    hit = 0
    dcg, idcg, hrt = 0.0, 0.0, 0.0

    for itemid in ranklist:
        if itemid in ground:
            hit += 1
            ind = ranklist.index(itemid)
            dcg += 1 / (log(ind + 2) / log(2))
    for ind in range(min(len(ranklist), len(ground))):
        idcg += 1 / (log(ind + 2) / log(2))
    hrt = 1.0 if hit > 0 else 0.0

    return (hit, dcg / idcg, hrt)


if __name__ == '__main__':
	data = "Netflix"
    top_k = 10

    mlp_layers = "[64,32,16,8]" # for ml-1m and Netflix
    # mlp_layers = "[50,32,16,8]" # for FilmTrust and Yelp

    mlp_regulation_rate = 0.0001
    mlp_epoch = 10  # the training epoch of PMLP, default is 1
    alpha = 0.9  # the balance factor

    with open("./SVD++_emb_file_" + data, 'rb') as f:
        emb_file = pickle.load(f)
        user_emb, item_emb, prediction_dic = emb_file["user_emb"], emb_file["item_emb"], emb_file["prediction_dic"]

    t = time()
    full_dataset = LoadDataset(
        "../data/" + data, prediction_dic, pair_num=3)
    train, train_ure, test_ure = full_dataset.trainMatrix, full_dataset.train_ure, full_dataset.test_ure
    user_num, item_num = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d" % (time() - t, user_num, item_num))

    # The following code is used for BC training in advance, and saves two feature matrices.
    # Dimension reduction of SVD++ vectors
    flag = True if os.path.exists('../data/temp/QUAVector_' + data) else False
    if not flag:
        for key in item_emb.keys():
            emb = item_emb[key]
            new = []
            for idx in range(0, len(emb) - 1, 2):
                emb[idx] += emb[idx + 1]
                new.append(emb[idx])
            item_emb[key] = new
        user_emb = np.array([user_emb[user] for user in range(user_num)])
        item_emb = np.array([item_emb[item] for item in range(item_num)])

        BC_model = BC(user_num, item_num, train_ure, user_emb, item_emb)
        BC_model.train_model()
        user_dic, item_dic = {}, {}
        for u in range(user_num):
            user_dic[u] = BC_model.P[u]
        for i in range(item_num):
            item_dic[i] = BC_model.Q[i]
        with open('../data/temp/QUAVector_' + data, 'wb') as f:
            pickle.dump({"user_emb": user_dic, "item_emb": item_dic}, f)

    # Initialize REG
    if flag:
        for key in item_emb.keys():
            emb = item_emb[key]
            new = []
            for idx in range(0, len(emb) - 1, 2):
                emb[idx] += emb[idx + 1]
                new.append(emb[idx])
            item_emb[key] = new
    user_emb = np.array([user_emb[user] for user in range(user_num)])
    item_emb = np.array([item_emb[item] for item in range(item_num)])
    reg = REG(user_emb, item_emb, layers=eval(mlp_layers))
    reg_optimizer = torch.optim.Adam(reg.parameters(), weight_decay=0.0001)
    # Initialize CLA
    with open("../data/temp/QUAVector_" + data, 'rb') as f:
        emb_file = pickle.load(f)
        user_emb, item_emb = emb_file['user_emb'], emb_file['item_emb']
        user_emb = np.array([user_emb[user] for user in range(user_num)])
        item_emb = np.array([item_emb[item] for item in range(item_num)])
    cla = CLA(user_emb, item_emb)

    training_data_generator = DataLoader(
        full_dataset, batch_size=512, num_workers=0)

    for epoch in range(mlp_epoch):

        reg_loss_list = []
        t = time()
        for feed_dic in training_data_generator:

            for key in feed_dic:
                if type(feed_dic[key]) != type(None):
                    feed_dic[key] = feed_dic[key].to(dtype=torch.long)

            reg.train()
            prediction = reg(feed_dic)
            true_rating = feed_dic["true_rating"].to(dtype=torch.float)
            true_rating = true_rating.float().view(prediction.size())

            reg_loss = torch.nn.MSELoss()(prediction, true_rating)

            reg_optimizer.zero_grad()
            reg_loss.backward()
            reg_optimizer.step()
            reg_loss_list.append(reg_loss.item())

        print("We get mean_regression_loss = %.4f in time: %.2f." % (np.mean(reg_loss_list), time() - t))

    total_hit, recall_len, ndcgs, hrts = [], [], [], []

    ranks, grounds = {}, {}
    with tqdm(total=len(test_ure)) as progress:

        reg.eval()
        for user in test_ure.keys():
            feed_dic = {
                "user_id": np.array([user for _ in range(item_num)]),
                "item_id": np.array([item for item in range(item_num)]),
            }
            reg_prediction = torch.sigmoid(reg.predict(feed_dic)).numpy()
            cla_prediction = cla.predict(feed_dic).reshape(item_num, 1).numpy()

            probability = (1 - alpha) * reg_prediction + alpha * cla_prediction

            try:
                ranklist = list(np.argsort([i[0] for i in probability])[::-1])[:100 + len(train_ure[user])]
                ranklist = [item for item in ranklist if item not in train_ure[user]]
            except:
                continue
            ranks[user] = ranklist[:100]
            grounds[user] = test_ure[user]

            hit, ndcg, hrt = evaluate(ranklist[:top_k], test_ure[user])
            total_hit.append(hit)
            recall_len.append(len(test_ure[user]))
            ndcgs.append(ndcg)
            hrts.append(hrt)
            progress.update(1)

    with open("../data/temp/user_ranklist_" + data, 'wb') as f:
        pickle.dump(ranks, f)
    with open("../data/temp/ground_truth_" + data, 'wb') as f:
        pickle.dump(grounds, f)

    precision = round(float(sum(total_hit)) / float(len(total_hit) * top_k) * 100, 5)
    precision10 = round(float(sum(total_hit10)) / float(len(total_hit10) * top_k10) * 100, 5)
    precision20 = round(float(sum(total_hit20)) / float(len(total_hit20) * top_k20) * 100, 5)


    # recall = round(float(sum(total_hit)) / float(sum(recall_len)) * 100, 5)
    NDCG = round(np.mean(ndcgs) * 100, 5)
    HR = round(np.mean(hrts) * 100, 5)

    print("precision = %.4f, recall = %.4f, NDCG = %.4f, HR = %.4f " % (precision, NDCG, HR))


