import tez
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing

import torch
import torch.nn as nn

class Model(tez.Model):
    def __init__(self, num_users, num_items):
        super ().__init__()
        self.user_embed = nn.Embedding (num_users, 32)
        self.item_embed = nn.Embedding(num_items, 32)
        self.out = nn.Linear(64, 1)
    
    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.fetch_optimizer, step_size = 3, gamma=0.7)
        return sch

    def monitor_metrics(self, output, rating):
        output = output.detach() .cpu(). numpy()
        rating = rating. detach() .cpu(). numpy()
        return {"rmse": np.sqrt(metrics.mean_squared_error(rating, output))}

    def forward(self, users, items, ratings=None):
        user_embeds = self.user_embed (users)
        item_embeds = self.item_embed (items)
        output = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.out (output)
        loss = nn.MSELoss()(output, ratings.view(-1, 1))
        calc_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
        return output, loss, calc_metrics

class ItemDataset:
    def __init__ (self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def len (self) :
        return len(self.users)

    def getitem (self, index):
        user = self.users[index]
        item = self.items[index]
        rating = self.ratings[index]

        return {"user": torch.tensor (user, dtype=torch.long),
        "item": torch.tensor (item, dtype=torch.long),
        "rating": torch.tensor(rating, dtype=torch.flaot)}

def train():
    data = pd.read_csv("customer_interaction.csv")
    # ID, user, movie, rating
    label_user = preprocessing.LabelEncoder()
    label_item = preprocessing.LabelEncoder()
    data.user = label_user.fit_transform(data.user.values)
    data.item = label_item.fit_transform(data.item.values)
    print("dfghjk")
    df_train, df_validation = model_selection.train_test_split(data, test_size=0.1, random_state=42, stratify = data.rating.values)
    train_dataset = ItemDataset(users=df_train.user.values, items=df_train.item.values, ratings=df_train.rating.values)
    valid_dataset = ItemDataset(users=df_validation.user.values, items=df_validation.item.values, ratings=df_validation.rating.values)

    model = Model(num_users=len(label_user.classes_), num_items=len(label_item.classes_))
    model.fit(train_dataset, valid_dataset, train_bs=1024, valid_bs=1024, fp16=True)

train()