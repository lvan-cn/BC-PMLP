import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset

np.random.seed(7)


class LoadDataset(Dataset):
    def __init__(self, file_name, prediction_dic, pair_num=1):
        self.trainMatrix, self.train_ure = self.load_rating_file_as_matrix(file_name + "_train.dat")
        self.user_input, self.item_input, self.probability, self.true_ratings = \
            self.get_train_instances(self.trainMatrix, prediction_dic, pair_num)
        self.test_ure = self.load_rating_file_as_list(file_name + "_test.dat")

    def __len__(self):
        'Denotes the total number of rating in test set'
        return len(self.user_input)

    def __getitem__(self, index):

        user_id = self.user_input[index]
        item_id = self.item_input[index]
        probability = self.probability[index]
        true_rating = self.true_ratings[index]

        return {'user_id': user_id,
                'item_id': item_id,
                'probability': probability,
                'true_rating': true_rating}

    def get_train_instances(self, train, prediction_dic, num_negatives):
        user_input, item_input, probabilities, predict_ratings, true_ratings = [], [], [], [], []
        num_users, num_items = train.shape
        for (u, i) in train.keys():
            prediction_list = prediction_dic[u]
            # positive instance
            user_input.append(u)
            item_input.append(i)
            probabilities.append(1)
            true_ratings.append(train[u, i])
            # negative instances
            for _ in range(num_negatives):
                j = np.random.randint(0, num_items - 1)
                while (u, j) in train:
                    j = np.random.randint(0, num_items - 1)
                user_input.append(u)
                item_input.append(j)
                probabilities.append(0)
                true_ratings.append(prediction_list[j])
        return user_input, item_input, probabilities, true_ratings

    def load_rating_file_as_list(self, filename):
        test_ure = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                try:
                    test_ure[user].append(item)
                except:
                    test_ure[user] = [item]
                line = f.readline()
        return test_ure

    def load_rating_file_as_matrix(self, filename):

        train_ure = {}
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                try:
                    train_ure[user].append(item)
                except:
                    train_ure[user] = [item]
                if (rating > 0):
                    mat[user, item] = rating
                line = f.readline()
        return mat, train_ure
