import os.path
import json
import spacy
import shutil
from openai import OpenAI
import numpy as np
import re
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Cexp0:
    def __init__(self, data_root_path, project_source, sim_model_path):
        self.data_root_path = data_root_path
        self.project_source = project_source
        self.sim_model = SentenceTransformer(sim_model_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
        self.space_data_copy()
        self.t_v_mode = "train"
        self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))

        self._fold_prepare()

    def _fold_prepare(self):
        all_path = ["./dataset/seeds", "./exp0_file", "./exp1_file", "./exp2_file",
                    "./exp0_ori_pair", "./model_checkpoints", "./ablation_summary"]
        for a_path in all_path:
            if not os.path.exists(a_path):
                os.makedirs(a_path, exist_ok=True)

    def space_data_copy(self):

        seed_save_path = os.path.join(self.project_source, "data/seeds")
        for seed_name in os.listdir(seed_save_path):
            seed_file_path = os.path.join(seed_save_path, seed_name)
            shutil.copy2(seed_file_path, "./dataset/seeds")

        train_data_path = "./dataset/space-train.json"
        valid_data_path = "./dataset/space-valid.json"
        if os.path.exists(train_data_path) and os.path.exists(valid_data_path):
            return True
        split_file_path = os.path.join(self.data_root_path, "space_summ_splits.txt")
        split_file = open(split_file_path, "r", encoding="utf-8").readlines()
        train_name_list = [split_file[i].split("dev")[0].strip() for i in range(len(split_file)) if
                           "dev" in split_file[i]]
        test_name_list = [split_file[i].split("test")[0].strip() for i in range(len(split_file)) if
                          "test" in split_file[i]]

        file_name = os.path.join(self.data_root_path, "space_summ.json")
        data_file = json.load(open(file_name, "r", encoding="utf-8"))
        train_data = {}
        valid_data = {}
        train_index = 0
        valid_index = 0
        for data in data_file:
            cur_data = {}
            cur_data["summary"] = data["summaries"]
            cur_data["reviews"] = data["reviews"]

            if data["entity_id"] in train_name_list:
                train_data[train_index] = cur_data
                train_index += 1
            if data["entity_id"] in test_name_list:
                valid_data[valid_index] = cur_data
                valid_index += 1

        json.dump(train_data, open(train_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(valid_data, open(valid_data_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        return True

    def extract_seed_entity_kw(self):
        seed_kw_path = "./exp0_file/seed_kw.json"
        if os.path.exists(seed_kw_path):
            return True

        all_asp_kw = {}
        for asp in self.aspects:
            file = open("./dataset/seeds/{}.txt".format(asp), "r", encoding="utf-8")
            kw_list = [a.split(" ")[1].replace("\n", "") for a in file.readlines()]
            kw_list = [self.nlp(a)[0].lemma_ for a in kw_list if self.nlp(a)[0].pos_ == "NOUN"]
            all_asp_kw[asp] = kw_list

        with open(seed_kw_path, "w", encoding="utf-8") as f:
            json.dump(all_asp_kw, f, indent=4, ensure_ascii=False)
        return True

    def extract_all_gold_entity(self):
        gold_entities_path = "./exp0_file/gold_entities_{}.json".format(self.t_v_mode)
        if os.path.exists(gold_entities_path):
            return True

        all_gold_entitys = {}
        for i in tqdm(range(25)):
            all_gold_entitys[str(i)] = {}
            for asp in ["general"] + self.aspects:
                entitys = []
                for j in range(3):
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
        for i in tqdm(range(25)):
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

    def aspect_weight_entity_freq(self, alpha=0.0):
        all_asp_entity_freq_path = "./exp0_file/all_asp_entity_freq_{}.json".format(self.t_v_mode)
        if os.path.exists(all_asp_entity_freq_path):
            return True

        with open("./exp0_file/seed_kw.json", "r", encoding="utf-8") as f:
            seed_kw = json.load(f)
        with open("./exp0_file/entity_freq_{}.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            all_entity_freq = json.load(f)

        all_asp_entity_freq = {}
        for asp in self.aspects:
            asp_entity_freq = {}
            for i in tqdm(range(25)):
                asp_entity_freq[str(i)] = {}
                entity_freq = all_entity_freq[str(i)]
                entities = [a for a in entity_freq.keys()]
                entities_emb = self.sim_model.encode(entities, convert_to_numpy=True)
                seed_kw_emb = self.sim_model.encode(seed_kw[asp], convert_to_numpy=True)
                score = np.max(cosine_similarity(entities_emb, seed_kw_emb), axis=1)
                for tok, sc in zip(entities, score):
                    asp_entity_freq[str(i)][tok] = round(float(entity_freq[tok] * alpha + (1 - alpha) * sc * 10), 2)

            all_asp_entity_freq[asp] = asp_entity_freq

        with open(all_asp_entity_freq_path, "w", encoding="utf-8") as f:
            json.dump(all_asp_entity_freq, f, indent=4, ensure_ascii=False)
        return True

    def class_main(self):
        self.space_data_copy()
        self.extract_seed_entity_kw()
        self.extract_all_gold_entity()
        self.extract_entity_freq()
        self.aspect_weight_entity_freq()
        self.t_v_mode = "valid"
        self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))
        self.extract_seed_entity_kw()
        self.extract_all_gold_entity()
        self.extract_entity_freq()
        self.aspect_weight_entity_freq()


class ExtractEmb:
    def __init__(self, encode_model_path):
        self.t_v_mode = "train"
        self.aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
        with open("./exp0_file/seed_kw.json", "r", encoding="utf-8") as f:
            self.seed_kw = json.load(f)
        self._load_data()
        self.model_name = encode_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)  # 自定义token时删除use_fast
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _load_data(self):
        self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))
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
        for i in range(25):
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

    def extract_seed_emb(self):
        seed_emb_path = "./exp0_file/seed_emb.pt"
        if os.path.exists(seed_emb_path):
            return True
        seed_emb = {}
        for asp in self.aspects:
            seed_emb[asp] = {}
            asp_kw = self.seed_kw[asp]
            for kw in asp_kw:

                hidden_states, tokens = self._generate_emb(kw)

                for idx, token in enumerate(tokens):
                    clean_token = token.lstrip("Ġ").lower()
                    if clean_token in asp_kw:
                        seed_emb[asp][clean_token] = hidden_states[idx].cpu()
        torch.save(seed_emb, "./exp0_file/seed_emb.pt")
        return True

    def compress_multi_entity_emb(self, n_clusters=20):
        limit_length_entity_emb_path = "./exp0_file/limit_length_entity_emb_{}.pt".format(self.t_v_mode)
        if os.path.exists(limit_length_entity_emb_path):
            return True

        all_entity_emb = torch.load("./exp0_file/entity_emb_{}.pt".format(self.t_v_mode))

        all_compress_entity_emb = {}
        for i in tqdm(range(25)):
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

        torch.save(all_compress_entity_emb, "./exp0_file/limit_length_entity_emb_{}.pt".format(self.t_v_mode))

    def class_main(self):
        self.extract_enity_emb()
        self.extract_seed_emb()
        self.compress_multi_entity_emb()
        self.t_v_mode = "valid"
        self._load_data()
        self.extract_enity_emb()
        self.extract_seed_emb()
        self.compress_multi_entity_emb()


class ExtractPair:
    def __init__(self, api_key):

        self.t_v_mode = "train"
        self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))

        self.instruction = (
                "The following is a customer review. Please extract all (noun, adjective) pairs that represent an entity and its descriptive attribute."
                "Each pair must consist of exactly one single-word noun and one single-word adjective. "
                "\nExample: The room was clean and the staff were helpful.\nExtracted pairs: ('room', 'clean'), ('staff', 'helpful')."
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
        aspects = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
        for sample_index in range(25):
            gold_pair_path = "./exp0_ori_pair/gold_pair_{}_{}.json".format(self.t_v_mode, sample_index)
            if os.path.exists(gold_pair_path):
                continue
            ori_pair = {}
            for asp in aspects:
                print("{}-{}".format(asp, sample_index))
                ori_pair[asp] = []
                for i in range(3):
                    gold_summary = self.train_data[str(sample_index)]["summary"][asp][i]
                    ret_pair = self.ds_get(self.instruction.format(gold_summary))
                    ori_pair[asp].append(ret_pair)

            with open(gold_pair_path, "w", encoding="utf-8") as f:
                json.dump(ori_pair, f, indent=4, ensure_ascii=False)
        return True

    def gold_ori2gold_pair(self):
        print("gold_ori2gold_pair")
        gold_summary_pair_path = "./exp0_file/gold_summary_pair_{}.json".format(self.t_v_mode)
        if os.path.exists(gold_summary_pair_path):
            return True
        aspects = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
        all_gold_pair = {}
        for i in tqdm(range(25)):
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
        for sample_index in range(25):
            ori_pair_path = "./exp0_ori_pair/ori_pair_{}_{}.json".format(self.t_v_mode, sample_index)
            if os.path.exists(ori_pair_path):
                continue
            ori_pair = []
            for index, sample in enumerate(self.train_data[str(sample_index)]["reviews"]):
                print("{}-{}".format(sample_index, index))
                a_reviews = " ".join(sample["sentences"])
                ret_pair = self.ds_get(self.instruction.format(a_reviews))
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
        for i in tqdm(range(25)):
            aim_pair = []
            with open("./exp0_ori_pair/ori_pair_{}_{}.json".format(self.t_v_mode, i), "r", encoding="utf-8") as f:
                ori_file = json.load(f)
            pattern = r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
            for text in ori_file:
                matches = re.findall(pattern, text)
                matches = [[a[0].lower(), a[1].lower()] for a in matches]  # 先转换小写
                # 再转换为词根
                lemmatized_data = [[self.nlp(noun.strip("'"))[0].lemma_, self.nlp(adj.strip("'"))[0].lemma_] for noun, adj in matches]
                lemmatized_data = [a for a in lemmatized_data if a != ["noun", "adjective"]]  # 去除LLM生成的默认"noun", "adjective"
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
        for i in range(25):
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
            self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))


class ExtractPairEmb:
    def __init__(self, encode_model_path):
        self.t_v_mode = "train"
        self._load_data()
        self.model_name = encode_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _load_data(self):
        self.train_data = json.load(open("./dataset/space-{}.json".format(self.t_v_mode), "r", encoding="utf-8"))

        with open("./exp0_file/gold_summary_{}_pair.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.gold_summary_pair = json.load(f)

        with open("./exp0_file/all_pair_{}_no_freq.json".format(self.t_v_mode), "r", encoding="utf-8") as f:
            self.all_pair = json.load(f)


    def _generate_emb(self, input_str):
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return hidden_states, tokens

    def extract_pair_emb(self):

        pair_emb_path = "./exp0_file/pair_emb_{}.pt".format(self.t_v_mode)
        if os.path.exists(pair_emb_path):
            return True
        all_kw_embeddings = {}
        for i in range(25):
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
    data_root_path = r"space dataset path"
    project_source = "qt-main(space) path"
    sim_model_path = r"all-mpnet-base-v2"
    encode_model_path = r"Llama-3.2-3B-Instruct"

    ele_class = Cexp0(data_root_path, project_source, sim_model_path)
    ele_class.class_main()
    ele_class = ExtractEmb(encode_model_path)
    ele_class.class_main()

    api_key = "deepseek api-key"
    ele_class = ExtractPair(api_key)
    ele_class.class_main()
    ele_class = ExtractPairEmb(encode_model_path)
    ele_class.class_main()