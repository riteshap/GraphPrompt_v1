
import json
import os.path

import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Compress:
    def __init__(self, lllm_path:str):
        self.t_v_mode = "train"
        self._load_data()
        self.aspects = ["general"]

        self.model_name = lllm_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _load_data(self):
        with open("./exp0_file/gold_entities_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.gold_entities = json.load(f)

    def compress_general(self):
        compressed_entity_general_path = "./exp1_file/compressed_entity_general_{}.json".format(self.t_v_mode)
        if os.path.exists(compressed_entity_general_path):
            return True

        best_R = 0
        best_top_n = 0

        with open("./exp0_file/entity_freq_train.json", "r", encoding="utf-8") as f:
            entity_freq = json.load(f)

        for top_n in range(200, 10, -10):
            tp = 0
            fp = 0
            fn = 0
            for i in range(len(self.gold_entities.keys())):
                sorted_kws = sorted(entity_freq[str(i)], key=entity_freq[str(i)].get, reverse=True)
                retained_entity = sorted_kws[:top_n]
                remove_entity = sorted_kws[top_n:]
                tp += len([a for a in retained_entity if a in self.gold_entities[str(i)]["general"]])
                fp += len([a for a in retained_entity if a not in self.gold_entities[str(i)]["general"]])
                fn += len([a for a in remove_entity if a in self.gold_entities[str(i)]["general"]])

            P = tp / (tp + fp)
            R = tp / (tp + fn)
            F1 = 2 * P * R / (P + R)
            if R > 0.8:
                best_R = R
                best_top_n = top_n

        print(best_R)
        print(best_top_n)

        with open("./exp0_file/entity_freq_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            entity_freq = json.load(f)

        all_compressed_entity = {}
        for i in range(len(self.gold_entities.keys())):
            sorted_kws = sorted(entity_freq[str(i)], key=entity_freq[str(i)].get, reverse=True)
            retained_entity = sorted_kws[:best_top_n]

            all_compressed_entity[str(i)] = retained_entity

        with open(compressed_entity_general_path, "w", encoding="utf-8") as f:
            json.dump(all_compressed_entity, f, indent=4, ensure_ascii=False)

    def prepare_general_train_emb(self):
        all_general_emb_path = "./exp1_file/all_general_{}_emb_label".format(self.t_v_mode)
        if os.path.exists(all_general_emb_path):
            return True

        with open("./exp1_file/compressed_entity_general_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            compressed_entity = json.load(f)
        with open("./exp0_file/entity_freq_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            all_entity_freq = json.load(f)
        all_limit_length_entity_emb = torch.load(
            "./exp0_file/limit_length_entity_emb_{}.pt".format(self.t_v_mode))

        all_train_emb = {}
        all_labels = {}

        for i in tqdm(range(len(self.gold_entities.keys()))):
            all_train_emb[str(i)] = {}
            all_labels[str(i)] = {}
            i_compressed_entity = ice = compressed_entity[str(i)]
            kw_exist_emb = all_limit_length_entity_emb[str(i)].keys()
            for kw in ice:
                if kw not in kw_exist_emb:
                    continue
                kw_emb_list = all_limit_length_entity_emb[str(i)][kw]
                kw_freq = all_entity_freq[str(i)][kw]
                inputs1 = self.tokenizer(str(kw_freq), return_tensors="pt").to(self.model.device)
                inputs2 = self.tokenizer(kw, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs1, output_hidden_states=True)
                    kw_freq_emb = outputs.hidden_states[-1][0][1:].cpu()
                    kw_freq_emb = [a for a in kw_freq_emb]
                    outputs = self.model(**inputs2, output_hidden_states=True)
                    kw_emb = outputs.hidden_states[-1][0][1:].cpu()
                    kw_emb = [a for a in kw_emb]

                kw_train_emb_list = kw_freq_emb + kw_emb_list + kw_emb
                all_train_emb[str(i)][kw] = kw_train_emb_list
                all_labels[str(i)][kw] = 1 if kw in self.gold_entities[str(i)]["general"] else 0
                if len(kw_train_emb_list) > 125:
                    print(len(kw_train_emb_list))
        ret_data = {"data": {"general": all_train_emb},
                    "label": {"general": all_labels}}
        torch.save(ret_data, all_general_emb_path)

    def class_main(self):
        self.compress_general()
        self.prepare_general_train_emb()
        self.t_v_mode = "valid"
        self._load_data()
        self.compress_general()
        self.prepare_general_train_emb()


if __name__ == "__main__":
    lllm_path = r"G:\python_code\pre_model\Llama-3.2-3B-Instruct"
    ele = Compress(lllm_path)
    ele.class_main()
