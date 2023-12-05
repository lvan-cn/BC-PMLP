import torch
from torch import nn
import torch.nn.functional as F


class REG(nn.Module):

    def __init__(self, user_emb, item_emb, layers=[16, 8]):

        super(REG, self).__init__()
        assert (layers[0] / 2 == user_emb.shape[1] and layers[0] / 2 == item_emb.shape[1]), \
            "The size of the first MLP layer is illegal."
        self.__alias__ = "MLP {}".format(layers)

        self.user_embedding = torch.nn.Parameter(torch.from_numpy(user_emb).to(dtype=torch.float))
        self.item_embedding = torch.nn.Parameter(torch.from_numpy(item_emb).to(dtype=torch.float))

        self.fc_layers = torch.nn.ModuleList()

        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, feed_dict):

        users = feed_dict['user_id']
        items = feed_dict['item_id']

        user_embedding = self.user_embedding.index_select(0, users)
        item_embedding = self.item_embedding.index_select(0, items)
        x = torch.cat([user_embedding, item_embedding], 1)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        prediction = self.output_layer(x)

        return prediction

    def predict(self, feed_dict):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(
                    feed_dict[key]).to(dtype=torch.long)
        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach()

    def get_alias(self):
        return self.__alias__

# CLA is only used to directly obtain the interaction probability and does not participate in training.
# Function __init__ is directly initialized with the trained vectors.
class CLA(nn.Module):

    def __init__(self, user_emb, item_emb):
        super(CLA, self).__init__()

        self.user_embedding = torch.from_numpy(user_emb).to(dtype=torch.float)
        self.item_embedding = torch.from_numpy(item_emb).to(dtype=torch.float)

    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']

        user_embedding = self.user_embedding.index_select(0, users)
        item_embedding = self.item_embedding.index_select(0, items)

        return torch.sigmoid(torch.sum(user_embedding * item_embedding, dim=1))

    def predict(self, feed_dict):

        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach()
