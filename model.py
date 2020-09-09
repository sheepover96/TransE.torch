import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TransEModel(nn.Module):

    def __init__(self, nentity, nrelation, vector_dim, *args, **kwargs):
        super(TransEModel, self).__init__(*args, **kwargs)
        self.nentity = nentity #
        self.nrelation = nrelation #
        self.vector_dim = vector_dim # k
        self.Ee = nn.Embedding(self.nentity, self.vector_dim)
        self.Ee.weight.data.uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5))
        self.El = nn.Embedding(self.nrelation, self.vector_dim)
        self.El.weight.data.uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5))

    def forward(self, e, l, t):
        # print(torch.max(l), self.nrelation)
        e_emb = self.Ee(e)
        l_emb = self.El(l)
        t_emb = self.Ee(t)
        d = F.normalize((e_emb + l_emb - t_emb))
        return d
        # neg_d1 = (e_emb + l_emb - neg_tail_emb)
        # neg_d2 = (neg_head_emb + l_emb - t_emb)
        # score1 = torch.max(F.normalize(self.margin + pos_d - neg_d1), torch.tensor([0.]))
        # score2 = torch.max(F.normalize(self.margin + pos_d - neg_d2), torch.tensor([0.]))


class TransE():
    def __init__(self, nentity, nrelation, vector_dim, margin):
        super(TransE, self).__init__()
        self.nentity = nentity #
        self.nrelation = nrelation #
        self.vector_dim = vector_dim # k
        self.margin = margin

    def _negative_sample(self, batch):
        neg_head = torch.randint(high=(self.nentity-1), size=(batch.shape[0],1))
        neg_link = torch.randint(high=(self.nentity-1), size=(batch.shape[0],1))
        return neg_head, neg_link

    def fit(self, X, batch_size=200, nepoch=100, lr=0.01):
        self.X = X
        self.hs = self.X[:,0]; self.rs = self.X[:,1]; self.ts = self.X[:,2]
        self.model = TransEModel(nentity=self.nentity, nrelation=self.nrelation, vector_dim=self.vector_dim)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.model.apply(init_params)
        # self.model = self.model.to(self.device)
        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size)
        self.model.train()
        for epoch in range(nepoch):
            batch_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                neg_head, neg_tail = self._negative_sample(batch)
                hs_batch = batch[:,0]; rs_batch = batch[:,1]; ts_batch = batch[:,2]
                pos_d = self.model(hs_batch, rs_batch, ts_batch)
                neg_d1 = self.model(neg_head, rs_batch, ts_batch)
                neg_d2 = self.model(hs_batch, rs_batch, neg_tail)
                loss1 = torch.max((self.margin + pos_d - neg_d1), torch.tensor([0.]))
                loss2 = torch.max((self.margin + pos_d - neg_d2), torch.tensor([0.]))
                loss = torch.sum(loss1) + torch.sum(loss2)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            print('epoch', epoch+1, batch_loss/len(train_loader))

    def test(self, Xtest):
        self.model.eval()
        Xtest = Xtest.to(self.device)
        #hs = Xtest[:,0]; rs = Xtest[:,1]; ts = Xtest[:,2]
        candidate_e = torch.tensor(list(range(self.nentity))).to(self.device)
        #candidate_eb = torch.stack([candidate_e for _ in range(hs.shape[0])])
        candidate_r = torch.tensor(list(range(self.nrelation))).to(self.device)
        #candidate_rb = torch.stack([candidate_r for _ in range(hs.shape[0])])

        hitk_t = 0; mean_rank_t = 0
        hitk_h = 0; mean_rank_h = 0
        test_loader = torch.utils.data.DataLoader(Xtest, batch_size=128)
        for batch_idx, batch in enumerate(test_loader):
            hs = batch[:,0]; rs = batch[:,1]; ts = batch[:,2]
            candidate_eb = torch.stack([candidate_e for _ in range(batch.shape[0])]).to(self.device)
            tail_ranking_score = self.model(hs, rs, candidate_eb, 0)
            rank_idxs = torch.argsort(tail_ranking_score, dim=1, descending=True)
            for idx, rank_idx in enumerate(rank_idxs):
                tail_prediction = candidate_e[rank_idx]
                t = ts[idx]
                rank = ((tail_prediction == t).nonzero()).squeeze().item()
                mean_rank_t += rank
                if t in tail_prediction[:10]:
                    hitk_t += 1

        for batch_idx, batch in enumerate(test_loader):
            hs = batch[:,0]; rs = batch[:,1]; ts = batch[:,2]
            candidate_eb = torch.stack([candidate_e for _ in range(batch.shape[0])]).to(self.device)
            head_ranking_score = self.model(ts, rs, candidate_eb, 2)
            rank_idxs = torch.argsort(head_ranking_score, dim=1, descending=True)
            for idx, rank_idx in enumerate(rank_idxs):
                head_prediction = candidate_e[rank_idx]
                h = hs[idx]
                rank = ((head_prediction == h).nonzero()).squeeze().item()
                mean_rank_h += rank
                if h in head_prediction[:10]:
                    hitk_h += 1

        hitk_t = (hitk_t / Xtest.shape[0]) * 100
        hitk_h = (hitk_h / Xtest.shape[0]) * 100
        mean_rank_t /= Xtest.shape[0]
        mean_rank_h /= Xtest.shape[0]

        return hitk_t, mean_rank_t, hitk_h, mean_rank_h