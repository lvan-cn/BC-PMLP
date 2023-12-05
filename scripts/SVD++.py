import numpy as np
import pickle
import random
import os

class SVDPP:
    def __init__(self, mat, K, user_num, item_num):
        self.mat = np.array(mat)
        self.K, self.user_num, self.item_num = K, user_num, item_num
        self.bi, self.bu, self.qi, self.pu = {}, {}, {}, {}
        self.avg = np.mean(self.mat[:, 2])
        self.y, self.u_dict = {}, {}
        for i in range(np.shape(self.mat)[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.u_dict.setdefault(uid, [])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.y.setdefault(iid, np.zeros((self.K, 1)) + .1)

        for iid in range(self.item_num):
            self.bi.setdefault(iid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.y.setdefault(iid, np.zeros((self.K, 1)) + .1)

    def predict(self, uid, iid):
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        self.y.setdefault(uid, np.zeros((self.K, 1)))
        self.u_dict.setdefault(uid, [])
        u_impl_prf, sqrt_Nu = self.getY(uid, iid)
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * (self.pu[uid] + u_impl_prf))

        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def getY(self, uid, iid):
        Nu = self.u_dict[uid]
        I_Nu = len(Nu)
        sqrt_Nu = np.sqrt(I_Nu)
        y_u = np.zeros((self.K, 1))
        if I_Nu == 0:
            u_impl_prf = y_u
        else:
            for i in Nu:
                y_u += self.y[i]
            u_impl_prf = y_u / sqrt_Nu

        return u_impl_prf, sqrt_Nu

    def train(self, steps=100, gamma=0.04, Lambda=0.15):
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                predict = self.predict(uid, iid)
                u_impl_prf, sqrt_Nu = self.getY(uid, iid)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                self.pu[uid] += gamma * (eui * self.qi[iid] - Lambda * self.pu[uid])
                self.qi[iid] += gamma * (eui * (self.pu[uid] + u_impl_prf) - Lambda * self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Nu - Lambda * self.y[j])

            gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))


def getData(train_file, test_file):

    train_data, test_data = [], []
    with open(train_file, 'r') as f:
        line = f.readline()
        while line != '' and line != "":
            line = line.split(" ")
            train_data.append([int(float(i)) for i in line[:3]])
            line = f.readline()
    with open(test_file, 'r') as f:
        line = f.readline()
        while line != None and line != "":
            line = line.split(" ")
            test_data.append([int(float(i)) for i in line[:3]])
            line = f.readline()

    print('load data finished')
    print('total data: ', len(train_data) + len(test_data))
    return train_data, test_data


if __name__ == '__main__':

    user_num, item_num = 9627, 5000
    file = "Netflix"
    train_data, test_data = getData("../data/"+ file +"_train.dat", "../data/"+ file +"_test.dat")
    # The second parameter controls the dimension of item vectors. 32 -- for ml-1m and Netflix ; 25 -- for FilmTrust and Yelp
    a = SVDPP(train_data, 25, user_num, item_num) # The second parameter controls the dimension of item vectors.
    a.train(steps=100)
    a.test(test_data)
    prediction_dic, item_emb = {}, {}
    for u in range(user_num):
        prediction_dic[u] = []
        ratings = prediction_dic[u]
        for i in range(item_num):
            ratings.append(a.predict(u, i))
    for item in range(item_num):
        item_emb[item] = [float('%.8f' % dim) for dim in np.concatenate((a.qi[item], a.y[item]))]
    for user in range(user_num):
        prediction_dic[user] = [float('%.8f' % dim) for dim in prediction_dic[user]]
        a.pu[user] = [float('%.8f' % dim) for dim in a.pu[user]]
    with open("SVD++_emb_file_" + file, 'wb') as f:
        pickle.dump({"prediction_dic": prediction_dic, "user_emb": a.pu, "item_emb": item_emb}, f)
    print("Finish!")



