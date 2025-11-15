
# -*- coding: utf-8 -*-

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


class EntityClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=3072,
                 trans1_layers=3,
                 trans2_layers=2,
                 nhead=8,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()

        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder1 = nn.TransformerEncoder(encoder_layer_1, num_layers=trans1_layers)

        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder2 = nn.TransformerEncoder(encoder_layer_2, num_layers=trans2_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_entity_token_seq, entity_token_mask, entity_mask):

        B, N, L, D = input_entity_token_seq.shape

        x1 = input_entity_token_seq.view(B * N, L, D)
        mask1 = entity_token_mask.view(B * N, L)
        out1 = self.encoder1(x1, src_key_padding_mask=mask1)

        entity_vecs = out1[:, -1, :]
        entity_vecs = entity_vecs.view(B, N, D)

        out2 = self.encoder2(entity_vecs, src_key_padding_mask=entity_mask)
        logits = self.classifier(out2)

        return logits


class EntityTokenDataset(Dataset):
    def __init__(self, pt_file,
                 asp,
                 max_entity_tokens=32,
                 max_entities=100,
                 embedding_dim=3072):

        data = torch.load(pt_file)
        self.embedding_dict = data["data"][asp]
        self.label_dict = data["label"][asp]
        self.sample_ids = list(self.embedding_dict.keys())

        self.max_entity_tokens = max_entity_tokens
        self.max_entities = max_entities
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        entity_emb_list = [a for a in self.embedding_dict[sample_id].values()]
        entity_kw = [a for a in self.embedding_dict[sample_id].keys()]
        label_list = [a for a in self.label_dict[sample_id].values()]
        label_kw = [a for a in self.label_dict[sample_id].keys()]
        assert entity_kw == label_kw

        num_entities = min(len(entity_emb_list), self.max_entities)
        entity_tensor = torch.zeros(self.max_entities, self.max_entity_tokens, self.embedding_dim)
        token_mask = torch.ones(self.max_entities, self.max_entity_tokens).bool()
        entity_mask = torch.ones(self.max_entities).bool()
        labels = torch.full((self.max_entities,), -100).long()

        for i in range(num_entities):
            ent_emb = entity_emb_list[i]
            ent_emb = torch.stack(ent_emb)
            L = ent_emb.size(0)
            L_pad = self.max_entity_tokens
            if L >= L_pad:
                padded = ent_emb[-L_pad:]
                token_mask[i] = False
            else:
                pad_len = L_pad - L
                padded = torch.cat([torch.zeros(pad_len, self.embedding_dim), ent_emb.cpu()], dim=0)
                token_mask[i, pad_len:] = False

            entity_tensor[i] = padded
            labels[i] = label_list[i]
            entity_mask[i] = False

        return {
            "entity_tokens": entity_tensor,
            "token_mask": token_mask,
            "entity_mask": entity_mask,
            "labels": labels,
            "label_kw": label_kw
        }


class ModelBuilder:
    def __init__(self):
        self.aspects = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
        self.aim_asp = None
        print(self.aim_asp)
        self.batch_size = 3
        self.num_epochs = 20
        self.lr = 5e-5
        self.weight_decay = 0.01
        if self.aim_asp == "general":
            self.pt_file = "./exp1_file/all_general_train_emb_label"
        else:
            self.pt_file = "./exp1_file/all_asp_train_emb_label"

    def _training_process(self):
        dataset = EntityTokenDataset(self.pt_file, self.aim_asp)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EntityClassifier()
        model = model.to(device)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                # 迁移到设备
                entity_tokens = batch["entity_tokens"].to(device)
                token_mask = batch["token_mask"].to(device)
                entity_mask = batch["entity_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(entity_tokens, token_mask, entity_mask)

                loss = criterion(logits.view(-1, 2), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
            if epoch in [10, 15]:
                torch.save(model.state_dict(),
                           r"./model_checkpoints/space_entity_{}_epoch{}.pt".format(self.aim_asp, epoch))

        torch.save(model.state_dict(), r"./model_checkpoints/space_entity_{}.pt".format(self.aim_asp))
        print("over")

    def _evaluating_process(self):
        if self.aim_asp == "general":
            self.pt_file = "./exp1_file/all_asp_valid_emb_label"
        else:
            self.pt_file = "./exp1_file/all_general_valid_emb_label"

        dataset = EntityTokenDataset(self.pt_file, self.aim_asp)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EntityClassifier()
        model.load_state_dict(torch.load("./model_checkpoints/space_entity_{}.pt".format(self.aim_asp)))
        model = model.to(device)
        model.eval()

        all_choosed_entity = {}
        sample_index = 0
        for batch in tqdm(dataloader):
            entity_tokens = batch["entity_tokens"].to(device)
            token_mask = batch["token_mask"].to(device)
            entity_mask = batch["entity_mask"].to(device)
            labels = batch["labels"].to(device)
            label_kw = batch["label_kw"]
            with torch.no_grad():
                logits = model(entity_tokens, token_mask, entity_mask)
                pred_label = torch.argmax(logits, dim=-1)
                pred_label = pred_label.squeeze(0).cpu()
                choosed_entity = [a[0] for a, b in zip(label_kw, pred_label[:len(label_kw)]) if b == 1]
                all_choosed_entity[str(sample_index)] = choosed_entity
            sample_index += 1

        with open("./exp2_file/choosed_entity_{}.json".format(self.aim_asp), "w", encoding="utf-8") as f:
            json.dump(all_choosed_entity, f)

    def training(self):
        for self.aim_asp in self.aspects:
            self._training_process()

    def evaluating(self):
        for self.aim_asp in self.aspects:
            self._evaluating_process()


class PairClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=3072,
                 trans0_layers=1,
                 trans1_layers=3,
                 trans2_layers=2,
                 nhead=8,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()

        encoder_layer_0 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder0 = nn.TransformerEncoder(encoder_layer_0, num_layers=trans0_layers)

        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder1 = nn.TransformerEncoder(encoder_layer_1, num_layers=trans1_layers)

        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder2 = nn.TransformerEncoder(encoder_layer_2, num_layers=trans2_layers)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_pair_token_seq, entity_token_mask, entity_mask):

        B, N, L, two, D = input_pair_token_seq.shape
        assert two == 2

        x0 = input_pair_token_seq.view(B * N * L, 2, D)
        out0 = self.encoder0(x0, src_key_padding_mask=None)

        pair_vecs = out0[:, -1, :]
        pair_vecs = pair_vecs.view(B, N, L, D)

        x1 = pair_vecs.view(B * N, L, D)
        mask1 = entity_token_mask.view(B * N, L)
        out1 = self.encoder1(x1, src_key_padding_mask=mask1)

        pair_vecs = out1[:, -1, :]
        pair_vecs = pair_vecs.view(B, N, D)

        out2 = self.encoder2(pair_vecs, src_key_padding_mask=entity_mask)
        logits = self.classifier(out2)

        return logits


class PairDataset(Dataset):
    def __init__(
            self,
            pair_emb_path: str,
            gold_pairs_path: str,
            sim_model_path: str,
            aspect: str = "general",
            n_cap: int = 32,
            l_cap: int = 32,
            embedding_dim: int = 3072,
            device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.aspect = aspect
        self.n_cap = n_cap
        self.l_cap = l_cap
        self.embedding_dim = embedding_dim

        self.pair_emb = torch.load(pair_emb_path, map_location="cpu")

        self.sim_model = SentenceTransformer(sim_model_path)

        with open(gold_pairs_path, "r", encoding="utf-8") as f:
            self.gold_pairs = json.load(f)

        self._build_sample()
        self.index = self._build_index()

    def _build_sample(self):

        gold_entites = {}
        for i in range(25):
            aim_list = [a[0] for a in self.gold_pairs[str(i)][self.aspect]]
            aim_list = list(set(aim_list))
            gold_entites[str(i)] = aim_list

        self.all_sample_emb_dict = {}
        for i in range(25):
            gold_ent_list = gold_entites[str(i)]
            sample_pair_emb = self.pair_emb[str(i)]
            sample_emb_dict = {}
            for ent_key in gold_ent_list:
                sample_emb_dict[ent_key] = {k: v for k, v in sample_pair_emb.items() if k.split("--")[0] == ent_key}
                if len(sample_emb_dict[ent_key].keys()) == 0:
                    sample_emb_dict.pop(ent_key)

            self.all_sample_emb_dict[str(i)] = sample_emb_dict

    def _build_index(self):

        index = []

        for sid, sample_emb_dict in self.all_sample_emb_dict.items():
            sid = str(sid)
            for gold_ent, ent_pair_emb in sample_emb_dict.items():
                ent_pairs = [a for a in ent_pair_emb.keys()]
                gold_pair_label = [a for a in self.gold_pairs[sid][self.aspect] if a[0] == gold_ent]
                index.append({"sid": sid, "gold_ent": gold_ent, "ent_pairs": ent_pairs, "gold_pair_label": gold_pair_label})

        return index


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        def pad_trunc(x: torch.Tensor, L: int) -> [torch.Tensor]:
            pad_mask = torch.ones(self.l_cap).bool()
            n, D = x.shape
            if n >= L:
                pad_mask[:] = False
                return x[:L], pad_mask
            out = torch.zeros((L, D), dtype=x.dtype, device=x.device)
            out[:n] = x
            pad_mask[:n] = False
            return out, pad_mask

        def max_sim_scores_by_disc(gold_pairs, cand_pairs, ts=0.7):

            gold_discs = [d for _, d in gold_pairs]
            cand_discs = [d for _, d in cand_pairs]

            gold_emb = self.sim_model.encode(gold_discs, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
            cand_emb = self.sim_model.encode(cand_discs, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

            sims = cand_emb @ gold_emb.T

            max_scores = sims.max(axis=1)

            results = [(cand_pairs[i][0], cand_pairs[i][1], float(max_scores[i])) for i in range(len(cand_pairs))]
            results.sort(key=lambda x: x[2], reverse=True)
            ret_pairs = [f"{a[0]}--{a[1]}" for a in results]
            label = torch.tensor([1 if a[2] > ts else 0 for a in results])

            return ret_pairs, label


        sample_keys = self.index[idx]

        sid, gold_ent, gold_pair, ent_pairs = (sample_keys["sid"], sample_keys["gold_ent"],
                                               sample_keys["gold_pair_label"], sample_keys["ent_pairs"])
        ent_pairs_emb = self.all_sample_emb_dict[sid][gold_ent]
        cand_pairs = [[a.split("--")[0], a.split("--")[1]] for a in ent_pairs]

        ent_pairs, label = max_sim_scores_by_disc(gold_pair, cand_pairs)

        all_mention_mask = []
        all_x = []
        for ent_pair in ent_pairs:
            ent_emb_list, disc_emb_list = ent_pairs_emb[ent_pair]

            ent = torch.stack([torch.as_tensor(t) for t in ent_emb_list], dim=0)
            disc = torch.stack([torch.as_tensor(t) for t in disc_emb_list], dim=0)

            ent_fixed, ent_mask = pad_trunc(ent, self.l_cap)
            disc_fixed, disc_mask = pad_trunc(disc, self.l_cap)

            x = torch.stack([ent_fixed, disc_fixed], dim=1)
            all_x.append(x)
            all_mention_mask.append(ent_mask)

        all_x = torch.stack(all_x)
        all_mention_mask = torch.stack(all_mention_mask)

        ret_tensor = torch.zeros(self.n_cap, self.l_cap, 2, self.embedding_dim)

        pair_num = len(ent_pairs)
        ret_pair_mask = torch.ones(self.n_cap).bool()
        ret_mention_mask = torch.ones(self.n_cap, self.l_cap).bool()
        ret_label = torch.zeros(self.n_cap, dtype=torch.long)
        if pair_num < self.n_cap:
            ret_tensor[:pair_num] = all_x
            ret_pair_mask[:pair_num] = False
            ret_mention_mask[:pair_num] = all_mention_mask
            ret_label[:pair_num] = label
            ret_ent_pairs = ent_pairs
        else:
            ret_tensor = all_x[:self.n_cap]
            ret_pair_mask[:] = False
            ret_mention_mask = all_mention_mask[:self.n_cap]
            ret_label = label[:self.n_cap]
            ret_ent_pairs = ent_pairs[:self.n_cap]

        ret_dict = {
            "ret_tensor": ret_tensor,
            "ret_pair_mask": ret_pair_mask,
            "ret_mention_mask": ret_mention_mask,
            "ret_label": ret_label,
            "ret_ent_pairs": ret_ent_pairs
        }

        return ret_dict


class PairModelBuilder:
    def __init__(self, sim_model_path):

        self.sim_model_path = sim_model_path
        self.aspects = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
        self.aim_asp = None
        print(self.aim_asp)
        self.batch_size = 1
        self.num_epochs = 20
        self.lr = 5e-5
        self.weight_decay = 0.01
        self.train_pair_emb_path = r".\exp0_file\pair_emb_train.pt"
        self.train_gold_pairs_path = r".\exp0_file\gold_summary_train_pair.json"
        self.valid_pair_emb_path = r".\exp0_file\pair_emb_valid.pt"

    def _training_process(self):
        dataset = PairDataset(self.train_pair_emb_path, self.train_gold_pairs_path, self.sim_model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PairClassifier()
        model = model.to(device)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                ret_tensor = batch["ret_tensor"].to(device)
                ret_pair_mask = batch["ret_pair_mask"].to(device)
                ret_mention_mask = batch["ret_mention_mask"].to(device)
                ret_label = batch["ret_label"].to(device)
                ret_ent_pairs = batch["ret_ent_pairs"]
                ret_sid = batch["ret_sid"]
                logits = model(ret_tensor, ret_mention_mask, ret_pair_mask)
                loss = criterion(logits.view(-1, 2), ret_label.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
            if epoch in [10, 15]:
                torch.save(model.state_dict(),
                           r"./model_checkpoints/space_pair_{}_epoch{}.pt".format(self.aim_asp, epoch))

        torch.save(model.state_dict(), r"./model_checkpoints/space_pair_{}.pt".format(self.aim_asp))
        print("over")

    def _evaluating_process(self):
        self.valid_gold_pairs_path = "./exp2_file/choosed_entity_{}.json".format(self.aim_asp)
        dataset = PairDataset(self.valid_pair_emb_path, self.valid_gold_pairs_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PairClassifier()
        model.load_state_dict(torch.load("./model_checkpoints/space_pair_{}.pt".format(self.aim_asp)))
        model = model.to(device)
        model.eval()

        all_choosed_disc = {}
        sample_index = 0
        for batch in tqdm(dataloader):
            ret_tensor = batch["ret_tensor"].to(device)
            ret_pair_mask = batch["ret_pair_mask"].to(device)
            ret_mention_mask = batch["ret_mention_mask"].to(device)
            ret_label = batch["ret_label"].to(device)
            ret_ent_pairs = batch["ret_ent_pairs"]
            ret_sid = batch["ret_sid"]
            if ret_sid not in all_choosed_disc.keys():
                all_choosed_disc[ret_sid] = []

            with torch.no_grad():
                logits = model(ret_tensor, ret_mention_mask, ret_pair_mask)
                probs = F.softmax(logits, dim=-1)
                pos_probs = probs[..., 1]
                valid = (ret_label == -100).sum().item()
                best_idx = pos_probs[:, valid:].argmax(dim=-1)
                best_disc = ret_ent_pairs[best_idx].split("--")[1]
                all_choosed_disc[ret_sid].append(best_disc)
            sample_index += 1

        with open("./exp2_file/choosed_pair_{}.json".format(self.aim_asp), "w", encoding="utf-8") as f:
            json.dump(all_choosed_disc, f)

    def training(self):
        for self.aim_asp in self.aspects:
            self._training_process()

    def evaluating(self):
        for self.aim_asp in self.aspects:
            self._evaluating_process()


if __name__ == "__main__":
    sim_model_path = r"F:\python_code\pre_model\all-mpnet-base-v2",

    ele_class = ModelBuilder()
    ele_class.training()
    ele_class.evaluating()

    ele_class = PairModelBuilder(sim_model_path)
    ele_class.training()
    ele_class.evaluating()








