# GraphPrompt_v1
Due to the large size of the amasum dataset, only the directory structure has been retained.<br>
The directory structure of this amasum project is as follows:<br>
```
|--amasum
|--|--step0_preprocess.py
|--|--step1_coarse_filter.py
|--|--step2_trans_model.py
|--|--step3_llm_request.py
|--|--amasum
|--|--|--train
|--|--|--valid
|--|--amasum-filtered
|--|--|--amasum-electronics-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--amasum-electronics-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--|--reviews.train.jsonl
|--|--|--|--|--reviews.dev.jsonl
|--|--|--amasum-home-kitchen-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--amasum-home-kitchen-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--|--reviews.train.jsonl
|--|--|--|--|--reviews.dev.jsonl
|--|--|--amasum-shoes-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--amasum-shoes-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--|--reviews.train.jsonl
|--|--|--|--|--reviews.dev.jsonl
|--|--|--amasum-sports-outdoors-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--amasum-sports-outdoors-filtered-25toks-0pronouns-charfilt-all
|--|--|--|--|--reviews.train.jsonl
|--|--|--|--|--reviews.dev.jsonl
```
The amasum folder contains the original amasum dataset<br>
amasum-filtered contains the theme-filtered dataset<br>
Original amasum dataset:<br>
https://github.com/abrazinskas/SelSum/tree/master/data<br>
Theme-filtered amasum dataset:<br>
https://github.com/tomhosking/hercules<br>
<br>
The project also uses the following models:<br>
<br>
all-mpnet-base-v2 from Sentence Transformers:<br>
https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models<br>
Llama-3.2-3B-Instruct:<br>
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct<br>
