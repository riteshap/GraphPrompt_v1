
import json
import time
import spacy
from rouge_score import rouge_scorer
from openai import OpenAI
import multiprocessing

class RequireLLM:
    def __init__(self, asp, exp_mode, shot_mode, llm_type, api_keys:dict):
        self.aspect = ["general"]
        self.asp = asp
        self.exp_mode = exp_mode
        self.shot_mode = shot_mode
        self.llm_type = llm_type
        assert self.asp in self.aspect
        assert self.exp_mode in ["baseline", "entity_only", "pair_only", "one_shot_only", "few_shot_only",
                                 "entity_pair", "entity_one_shot", "entity_few_shot",
                                 "pair_one_shot", "pair_few_shot"]
        assert self.shot_mode in ["none", "one", "three"]
        assert self.llm_type in ["dsv3", "GPT5", "Qwen32B", "llama70B"]

        self.dsv3_api_key = api_keys["dsv3"]
        self.GPT5_api_key = api_keys["GPT5"]
        self.Qwen32B_api_key = api_keys["Qwen32B"]
        self.llama70B_api_key = api_keys["llama70B"]

        self.train_data = json.load(open("./dataset/amasum-train.json", "r", encoding="utf-8"))
        self.valid_data = json.load(open("./dataset/amasum-valid.json", "r", encoding="utf-8"))
        with open("./exp2_file/choosed_entity_{}.json".format(self.asp), "r", encoding="utf-8") as f:
            self.all_choosed_entity = json.load(f)
        with open("./exp2_file/choosed_pair_{}.json".format(self.asp), "r", encoding="utf-8") as f:
            self.all_choosed_pair = json.load(f)

    def llama70B_get(self, que):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.llama70B_api_key,
        )

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>",
            },
            extra_body={},
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
                {"role": "user", "content": "{}".format(que)},
            ]
        )

        return completion.choices[0].message.content

    def gpt_get(self, que):
        client = OpenAI()
        time.sleep(15)
        response = client.responses.create(
            model="gpt-5",
            input=que
        )

        return response.output_text

    def Qwen32B_get(self, que):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.Qwen32B_api_key,
        )

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>",
            },
            extra_body={},
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
                {"role": "user", "content": "{}".format(que)},
            ]
        )

        return completion.choices[0].message.content

    def ds_get(self, que):
        client = OpenAI(api_key=self.dsv3_api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes customer reviews."},
                {"role": "user", "content": "{}".format(que)},
            ],
            stream=False
        )
        ss = response.choices[0].message
        return ss.content

    def build_instruction(self, sample_index:int):
        ins_beg = "Summarize the reviews by "
        ins_asp = "focusing on the {aspect} aspect and "
        ins_entity = "incorporating the following key entities: {entity}. "
        ins_pair = "incorporate the following opinion–sentiment pairs: {pair}. "
        ins_entity_inf = "The entities represent important aspects of the product or service, "
        ins_pair_inf = "the pairs reflect key opinions or sentiments expressed about those aspects. "
        ins_ent_pair_inf = "You may adapt, omit, or extend the phrasing as needed to generate a coherent and accurate summary. "
        ins_length = "The summary should be between {min_num} to {max_num} words. "

        if self.asp == "general":
            if self.exp_mode == "general":
                instruction = (ins_beg + ins_entity + "Also " + ins_pair + ins_entity_inf + "and " +
                               ins_pair_inf + ins_ent_pair_inf + ins_length)
            elif self.exp_mode == "baseline":
                instruction = ins_beg[:-4] + ". " + ins_length
                self.shot_mode = "none"
            elif self.exp_mode == "entity_only":
                instruction = ins_beg + ins_entity + ins_entity_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "none"
            elif self.exp_mode == "pair_only":
                instruction = ins_beg + ins_pair + ins_pair_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "none"
            elif self.exp_mode == "one_shot_only":
                instruction = ins_beg[:-4] + ". " + ins_length
                self.shot_mode = "one"
            elif self.exp_mode == "few_shot_only":
                instruction = ins_beg[:-4] + ". " + ins_length
                self.shot_mode = "three"
            elif self.exp_mode == "entity_pair":
                instruction = (ins_beg + ins_entity + "Also " + ins_pair + ins_entity_inf + "and " +
                               ins_pair_inf + ins_ent_pair_inf + ins_length)
                self.shot_mode = "none"
            elif self.exp_mode == "entity_one_shot":
                instruction = ins_beg + ins_entity + ins_entity_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "one"
            elif self.exp_mode == "entity_few_shot":
                instruction = ins_beg + ins_entity + ins_entity_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "three"
            elif self.exp_mode == "pair_one_shot":
                instruction = ins_beg + ins_pair + ins_pair_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "one"
            elif self.exp_mode == "pair_few_shot":
                instruction = ins_beg + ins_pair + ins_pair_inf + ins_ent_pair_inf + ins_length
                self.shot_mode = "three"
            else:
                raise Exception("error")
        else:
            instruction = (ins_beg + ins_asp + ins_entity + "Also " + ins_pair + ins_entity_inf + "and " +
                           ins_pair_inf + ins_ent_pair_inf + ins_length)

        one_shot = "Use the following example to guide the tone and sentence structure of your summary: {}"
        three_shot = "Use the following three examples to guide the tone and sentence structure of your summary: {}"

        insert_asp = "" if self.asp == "general" else self.asp

        gold_summary = self.valid_data[str(sample_index)]["summary"][self.asp][0]
        len_min = len(gold_summary.split(" "))
        len_max = len_min + 10

        prefix_dict = {"aspect": insert_asp,
                       "entity": ", ".join(self.all_choosed_entity[str(sample_index)]),
                       "pair": ", ".join(self.all_choosed_pair[str(sample_index)]),
                       "min_num": len_min,
                       "max_num": len_max}

        instruction = instruction.format(**prefix_dict)

        if self.shot_mode == "one":
            one_shot = one_shot.format(self.train_data["0"]["summary"][self.asp][0])
            instruction += one_shot
        elif self.shot_mode == "three":
            a = self.train_data["0"]["summary"][self.asp][0]
            b = self.train_data["1"]["summary"][self.asp][0]
            c = self.train_data["2"]["summary"][self.asp][0]
            one_shot = three_shot.format("1." + a + " 2." + b + " 3." + c)
            instruction += one_shot

        a_reviews = {}
        for index, sample in enumerate(self.valid_data[str(sample_index)]["reviews"]):
            a_reviews[index] = " ".join(sample["sentences"])
        reviews = json.dumps(a_reviews, ensure_ascii=False)

        return instruction, reviews

    def get_llm_summary(self):
        if self.llm_type == "dsv3":
            llm_get = self.ds_get
        elif self.llm_type == "GPT5":
            llm_get = self.gpt_get
        elif self.llm_type == "Qwen32B":
            llm_get = self.Qwen32B_get
        elif self.llm_type == "llama70B":
            llm_get = self.llama70B_get
        else:
            raise Exception("no such llm type")

        asp_llm_summary = {}
        for i in range(len(self.valid_data.keys())):
            print(i)
            instruction, reviews = self.build_instruction(i)
            ret_summary = llm_get(reviews + instruction)
            asp_llm_summary[str(i)] = ret_summary

        save_path = ("./ablation_summary/amasum_{}_{}_{}.json"
                     .format(self.asp, self.llm_type, self.exp_mode))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asp_llm_summary, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    aspect = ["general"]
    exp_modes = ["baseline", "entity_only", "pair_only", "one_shot_only", "few_shot_only",
                 "entity_pair", "entity_one_shot", "entity_few_shot",
                 "pair_one_shot", "pair_few_shot"]
    shot_modes = ["none", "one", "three"]
    llm_types = ["dsv3", "GPT5", "Qwen32B", "llama70B"]
    api_keys = {
        "dsv3": "sk-600",
        "GPT5": "Follow openai guidance, put key in environment",
        "Qwen32B": "sk-or-v1-6",
        "llama70B": "sk-or-v1-6"
    }

    ele = RequireLLM(aspect[0], exp_modes[0], shot_modes[0], llm_types[2], api_keys)
    ele.get_llm_summary()
