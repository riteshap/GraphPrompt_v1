
import json
import os
import spacy
from tqdm import tqdm
from openai import OpenAI
from sklearn.cluster import KMeans
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Aexp0:
    def __init__(self, train_data_root_path, valid_data_root_path, amasum_part_path):
        self.train_data_root_path = train_data_root_path
        self.valid_data_root_path = valid_data_root_path
        self.amasum_part_path = amasum_part_path
        self.t_v_mode = "train"
        self.nlp = spacy.load("en_core_web_sm")
        self._fold_prepare()
        self.collect_train_data()
        self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))

    def _fold_prepare(self):
        all_path = ["./dataset/seeds", "./exp0_file", "./exp1_file", "./exp2_file",
                    "./exp0_ori_pair", "./model_checkpoints", "./ablation_summary"]
        for a_path in all_path:
            if not os.path.exists(a_path):
                os.makedirs(a_path, exist_ok=True)


    def _read_json(self, json_path: str):
        ret_id_list = []
        all_line = open(json_path, "r", encoding="utf-8")
        for line in all_line.readlines():
            sample = json.loads(line)
            entity_id = sample["entity_id"]
            if entity_id not in ret_id_list:
                ret_id_list.append(entity_id)
        return ret_id_list


    def _extract_train_data(self, ret_id_list, data_root_path):
        ret_data_dict = {}
        for i, entity_id in enumerate(ret_id_list):
            sample_dict = {}
            file_path = os.path.join(data_root_path, entity_id + ".json")
            file_content = json.load(open(file_path, "r", encoding="utf-8"))
            summary = (file_content["verdict"][0] + ". ".join(file_content["pros"][0])
                       + ". ".join(file_content["cons"][0]))
            reviews = []
            for review in file_content["all_review"]:
                reviews.append({"sentences": [review["text"]]})

            sample_dict["summary"] = {"general": [summary]}
            sample_dict["reviews"] = reviews
            ret_data_dict[str(i)] = sample_dict

        return ret_data_dict


    def collect_train_data(self):

        train_data_path = "./dataset/amasum-train.json"
        valid_data_path = "./dataset/amasum-valid.json"
        if os.path.exists(train_data_path) and os.path.exists(valid_data_path):
            return True

        train_id_list = []
        valid_id_list = []
        for ele in os.walk(self.amasum_part_path):
            root_path = ele[0]
            for child_path in ele[2]:
                if not child_path.endswith(".jsonl"):
                    continue
                if "train" in child_path:
                    ret_id_list = self._read_json(os.path.join(root_path, child_path))
                    train_id_list.extend(ret_id_list)
                if "dev" in child_path:
                    ret_id_list = self._read_json(os.path.join(root_path, child_path))
                    valid_id_list.extend(ret_id_list)

        train_data = self._extract_train_data(train_id_list, self.train_data_root_path)
        valid_data = self._extract_train_data(valid_id_list, self.valid_data_root_path)

        json.dump(train_data, open(train_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(valid_data, open(valid_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        return True


    def extract_all_gold_entity(self):
        gold_entities_path = "./exp0_file/gold_entities_{}.json".format(self.t_v_mode)
        if os.path.exists(gold_entities_path):
            return True

        all_gold_entitys = {}
        for i in tqdm(range(len(self.train_data.keys()))):
            all_gold_entitys[str(i)] = {}
            for asp in ["general"]:
                entitys = []
                for j in range(1):
                    gold_summary = self.train_data[str(i)]["summary"][asp][j]
                    doc = self.nlp(gold_summary)
                    ori_list = list(
                        set([token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token.lemma_) > 2]))
                    entitys.extend([a for a in ori_list if self.nlp(a)[0].pos_ == "NOUN"])
                all_gold_entitys[str(i)][asp] = list(set(entitys))

        with open(gold_entities_path, "w", encoding="utf-8") as f:
            json.dump(all_gold_entitys, f, indent=4, ensure_ascii=False)
        return True


    def extract_entity_freq(self):
        entity_freq_path = "./exp0_file/entity_freq_{}.json".format(self.t_v_mode)
        if os.path.exists(entity_freq_path):
            return True

        all_entity_freq = {}
        for i in tqdm(range(len(self.train_data.keys()))):
            entity_freq = {}
            for reviews in self.train_data[str(i)]["reviews"]:
                for sent in reviews["sentences"]:
                    doc = self.nlp(sent)
                    sent_entity = [token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token.lemma_) > 2]
                    sent_entity = [a for a in sent_entity if self.nlp(a)[0].pos_ == "NOUN"]
                    for ent in sent_entity:
                        if ent not in entity_freq.keys():
                            entity_freq[ent] = 1
                        else:
                            entity_freq[ent] += 1

            entities = [a for a in entity_freq.keys()]
            for ent in entities:
                if entity_freq[ent] < 3:
                    entity_freq.pop(ent)

            all_entity_freq[str(i)] = entity_freq

        with open(entity_freq_path, "w", encoding="utf-8") as f:
            json.dump(all_entity_freq, f, indent=4, ensure_ascii=False)
        return True

    def class_main(self):
        self.extract_all_gold_entity()
        self.extract_entity_freq()
        self.t_v_mode = "valid"
        self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))
        self.extract_all_gold_entity()
        self.extract_entity_freq()


class ExtractEmb:
    def __init__(self, encode_model_path):
        self.t_v_mode = "train"
        self.aspects = ["general"]
        self._load_data()
        self.model_name = encode_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)  # 自定义token时删除use_fast
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _load_data(self):
        self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))
        with open("./exp0_file/entity_freq_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.all_entity_freq = json.load(f)

    def _generate_emb(self, input_str):
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return hidden_states, tokens


    def extract_enity_emb(self):
        entity_emb_path = "./exp0_file/entity_emb_{}.pt".format(self.t_v_mode)
        if os.path.exists(entity_emb_path):
            return True
        all_kw_embeddings = {}
        for i in range(len(self.train_data.keys())):
            kw_set = [a for a in self.all_entity_freq[str(i)].keys()]
            all_reviews = self.train_data[str(i)]["reviews"]
            kw_embeddings = {}
            for ele in all_reviews:
                for sent in ele["sentences"]:

                    hidden_states, tokens = self._generate_emb(sent)

                    for idx, token in enumerate(tokens):
                        clean_token = token.lstrip("Ġ").lower()
                        if clean_token in kw_set:
                            if clean_token not in kw_embeddings.keys():
                                kw_embeddings[clean_token] = [hidden_states[idx].cpu()]
                            else:
                                kw_embeddings[clean_token].append(hidden_states[idx].cpu())
            all_kw_embeddings[str(i)] = kw_embeddings
            print(len(kw_embeddings))
        torch.save(all_kw_embeddings, entity_emb_path)
        return True


    def compress_multi_entity_emb(self, n_clusters=20):
        limit_length_entity_emb_path = "./exp0_file/limit_length_entity_emb_{}.pt".format(self.t_v_mode)
        if os.path.exists(limit_length_entity_emb_path):
            return True

        all_entity_emb = torch.load("./exp0_file/entity_emb_{}.pt".format(self.t_v_mode))

        all_compress_entity_emb = {}
        for i in tqdm(range(len(self.train_data.keys()))):
            all_compress_entity_emb[str(i)] = {}
            entity_emb = all_entity_emb[str(i)]
            for kw in entity_emb.keys():
                emb_list = entity_emb[kw]
                if len(emb_list) <= n_clusters:
                    all_compress_entity_emb[str(i)][kw] = emb_list
                    continue
                X = torch.stack(emb_list).cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                kmeans.fit(X)

                labels = kmeans.labels_

                representative_indices = []
                for j in range(n_clusters):
                    cluster_indices = np.where(labels == j)[0]
                    if len(cluster_indices) == 0:
                        continue
                    cluster_vectors = X[cluster_indices]
                    center = kmeans.cluster_centers_[j]
                    distances = np.linalg.norm(cluster_vectors - center, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    representative_indices.append(closest_idx)

                compressed_tensor_list = [emb_list[j] for j in representative_indices]
                all_compress_entity_emb[str(i)][kw] = compressed_tensor_list

        torch.save(all_compress_entity_emb, limit_length_entity_emb_path)

    def class_main(self):
        self.extract_enity_emb()
        self.compress_multi_entity_emb()
        self.t_v_mode = "valid"
        self._load_data()
        self.extract_enity_emb()
        self.compress_multi_entity_emb()


class ExtractPair:
    def __init__(self, api_key):

        self.t_v_mode = "train"
        self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))
        self.examples = open("./Aexp0pair_sample", "r", encoding="utf-8").readlines()
        self.examples = " ".join(self.examples)

        self.instruction = (
                "The following is a customer review. Please extract all (noun, adjective) pairs that represent an entity and its descriptive attribute."
                "Each pair must consist of exactly one single-word noun and one single-word adjective. "
                "\nExample: {}"
                "\nNow extract pairs from the following review:\nReviews:{}")

        self.api_key = api_key

        self.nlp = spacy.load("en_core_web_sm")


    def ds_get(self, que):
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an helpful assistant that extracts (entity, attribute) pairs from customer reviews. "
                                              "Each pair should consist of a noun phrase (entity) and an attribute that describes it."},
                {"role": "user", "content": "{}".format(que)},
            ],
            stream=False
        )
        ss = response.choices[0].message
        return ss.content

    def get_gold_train_pair(self):
        aspects = ["general"]
        for sample_index in range(len(self.train_data.keys())):
            gold_pair_path = "./exp0_ori_pair/gold_pair_{}_{}.json".format(self.t_v_mode, sample_index)
            if os.path.exists(gold_pair_path):
                continue
            ori_pair = {}
            for asp in aspects:
                print("{}-{}".format(asp, sample_index))
                ori_pair[asp] = []
                for i in range(1):
                    gold_summary = self.train_data[str(sample_index)]["summary"][asp][i]
                    ret_pair = self.ds_get(self.instruction.format(self.examples, gold_summary))
                    ori_pair[asp].append(ret_pair)

            with open(gold_pair_path, "w", encoding="utf-8") as f:
                json.dump(ori_pair, f, indent=4, ensure_ascii=False)
        return True

    def gold_ori2gold_pair(self):
        print("gold_ori2gold_pair")
        gold_summary_pair_path = "./exp0_file/gold_summary_pair_{}.json".format(self.t_v_mode)
        if os.path.exists(gold_summary_pair_path):
            return True
        aspects = ["general"]
        all_gold_pair = {}
        for i in tqdm(range(len(self.train_data.keys()))):
            all_gold_pair[str(i)] = {}
            with open("./exp0_ori_pair/gold_pair_{}_{}.json".format(self.t_v_mode, i), "r", encoding="utf-8") as f:
                ori_file = json.load(f)
            pattern = r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
            for asp in aspects:
                aim_pair = []
                for text in ori_file[asp]:
                    matches = re.findall(pattern, text)
                    matches = [[a[0].lower(), a[1].lower()] for a in matches]

                    lemmatized_data = [[self.nlp(noun.strip("'"))[0].lemma_, self.nlp(adj.strip("'"))[0].lemma_] for noun, adj in matches]
                    lemmatized_data = [a for a in lemmatized_data if a != ["noun", "adjective"]]
                    aim_pair.extend(lemmatized_data)
                all_gold_pair[str(i)][asp] = aim_pair

        with open(gold_summary_pair_path, "w", encoding="utf-8") as f:
            json.dump(all_gold_pair, f, indent=4, ensure_ascii=False)

        return True

    def get_pair(self):
        for sample_index in range(len(self.train_data.keys())):
            ori_pair_path = "./exp0_ori_pair/ori_pair_{}_{}.json".format(self.t_v_mode, sample_index)
            if os.path.exists(ori_pair_path):
                continue
            ori_pair = []
            for index, sample in enumerate(self.train_data[str(sample_index)]["reviews"]):
                print("{}-{}".format(sample_index, index))
                a_reviews = " ".join(sample["sentences"])
                ret_pair = self.ds_get(self.instruction.format(self.examples, a_reviews))
                ori_pair.append(ret_pair)
            with open(ori_pair_path, "w", encoding="utf-8") as f:
                json.dump(ori_pair, indent=4, ensure_ascii=False)
        return True

    def ori_pair2pair(self):
        print("ori_pair2pair")
        all_pair = {}
        all_pair_path = "./exp0_file/all_pair_{}_no_freq.json".format(self.t_v_mode)
        if os.path.exists(all_pair_path):
            return True
        for i in tqdm(range(len(self.train_data.keys()))):
            aim_pair = []
            with open("./exp0_ori_pair/ori_pair_{}_{}.json".format(self.t_v_mode, i), "r", encoding="utf-8") as f:
                ori_file = json.load(f)
            pattern = r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
            for text in ori_file:
                matches = re.findall(pattern, text)
                matches = [[a[0].lower(), a[1].lower()] for a in matches]
                lemmatized_data = [[self.nlp(noun.strip("'"))[0].lemma_, self.nlp(adj.strip("'"))[0].lemma_] for noun, adj in matches]
                lemmatized_data = [a for a in lemmatized_data if a != ["noun", "adjective"]]
                aim_pair.extend(lemmatized_data)
            all_pair[str(i)] = aim_pair

        with open(all_pair_path, "w", encoding="utf-8") as f:
            json.dump(all_pair, f, indent=4, ensure_ascii=False)
        return True

    def pair2freq_pair(self):
        print("pair2freq_pair")
        pair_freq_path = "./exp0_file/pair_freq_{}.json".format(self.t_v_mode)
        if os.path.exists(pair_freq_path):
            return True
        with open("./exp0_file/all_pair_{}_no_freq.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            pair_file = json.load(f)
        pair_freq = {}
        for i in range(len(self.train_data.keys())):
            pair_freq[str(i)] = {}
            for a in pair_file[str(i)]:
                if a[0] not in pair_freq[str(i)].keys():
                    pair_freq[str(i)][a[0]] = {a[1]: 1}
                elif a[1] not in pair_freq[str(i)][a[0]].keys():
                    pair_freq[str(i)][a[0]][a[1]] = 1
                else:
                    pair_freq[str(i)][a[0]][a[1]] += 1

        with open(pair_freq_path, "w", encoding="utf-8") as f:
            json.dump(pair_freq, f, indent=4, ensure_ascii=False)
        return True

    def class_main(self):
        for i in range(2):
            self.get_gold_train_pair()
            self.gold_ori2gold_pair()
            self.get_pair()
            self.ori_pair2pair()
            self.pair2freq_pair()
            self.t_v_mode = "valid"
            self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))


class ExtractPairEmb:
    def __init__(self, encode_model_path):
        self.t_v_mode = "train"
        self._load_data()
        self.model_name = encode_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _load_data(self):
        self.train_data = json.load(open("./dataset/amasum-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))

        with open("./exp0_file/gold_summary_{}_pair.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.gold_summary_pair = json.load(f)

        with open("./exp0_file/all_pair_{}_no_freq.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.all_pair = json.load(f)


    def _generate_emb(self, input_str):
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]  # shape: [seq_len, hidden_size]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return hidden_states, tokens

    def extract_pair_emb(self):

        pair_emb_path = "./exp0_file/pair_emb_{}.pt".format(self.t_v_mode)
        if os.path.exists(pair_emb_path):
            return True
        all_kw_embeddings = {}
        for i in range(len(self.train_data.keys())):
            pair_embeddings = {}
            sample_id = str(i)
            sample_pairs = self.all_pair[sample_id]
            all_reviews = self.train_data[sample_id]["reviews"]

            for j, review in enumerate(all_reviews):
                print(f"sample:{i}-review:{j}")
                for sent in review["sentences"]:
                    hidden_states, tokens = self._generate_emb(sent)

                    norm_tokens = [tok.lstrip("Ġ").lower() for tok in tokens]

                    token_pos = {tok: [] for tok in set(norm_tokens)}
                    for idx, tok in enumerate(norm_tokens):
                        token_pos[tok].append(idx)

                    sent_pair_emb = {}

                    for a_pair in sample_pairs:
                        a0, a1 = a_pair[0].lower(), a_pair[1].lower()
                        if a0 in token_pos and a1 in token_pos:
                            cur_key = f"{a0}--{a1}"

                            idx_0 = token_pos[a0][0]
                            idx_1 = token_pos[a1][0]

                            emb_0 = hidden_states[idx_0].detach().cpu()
                            emb_1 = hidden_states[idx_1].detach().cpu()

                            if cur_key not in sent_pair_emb:
                                sent_pair_emb[cur_key] = [[emb_0], [emb_1]]
                            else:
                                sent_pair_emb[cur_key][0].append(emb_0)
                                sent_pair_emb[cur_key][1].append(emb_1)

                    for k, (sideA_list, sideB_list) in sent_pair_emb.items():
                        if k not in pair_embeddings:
                            pair_embeddings[k] = [[], []]
                        pair_embeddings[k][0].extend(sideA_list)
                        pair_embeddings[k][1].extend(sideB_list)

            pair_embeddings = {
                k: v for k, v in pair_embeddings.items()
                if len(v[0]) == len(v[1]) and len(v[0]) > 0
            }

            all_kw_embeddings[sample_id] = pair_embeddings

        torch.save(all_kw_embeddings, pair_emb_path)
        return True

    def class_main(self):
        self.extract_pair_emb()
        self.t_v_mode = "valid"
        self._load_data()
        self.extract_pair_emb()


if __name__ == "__main__":
    train_data_root_path = r".\amasum\train"
    valid_data_root_path = r".\amasum\valid"
    amasum_part_path = r".\amasum-filtered"

    ele_class = Aexp0(train_data_root_path, valid_data_root_path, amasum_part_path)
    ele_class.class_main()

    encode_model_path = r"Llama-3.2-3B-Instruct"
    ele_class = ExtractEmb(encode_model_path)
    ele_class.class_main()

    api_key = "deepseek api-key"
    ele_class = ExtractPair(api_key)
    ele_class.class_main()
    ele_class = ExtractPairEmb(encode_model_path)
    ele_class.class_main()


