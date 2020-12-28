import os
# 1. export config from config class
from configure_pretraining import PretrainingConfig
from util import training_utils

model_name = 'electra_small_wiki40b_ja_mecab_ipadic'
data_dir = '/app/outputs/models'
config = PretrainingConfig(model_name, data_dir)
bert_config = training_utils.get_bert_config(config)
config_file =f'{data_dir}/bert_config.json'
with open(config_file, "w") as f:
  f.write(bert_config.to_json_string())

# 2. export pytorch model
from transformers import BertConfig, BertModel

ckpt_index_file = f'{data_dir}/model.ckpt-xxxx.index'
config = BertConfig.from_json_file(config_file)
model = BertModel.from_pretrained(ckpt_index_file, from_tf=True, config=config)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
model.save_pretrained(f'{data_dir}/')
with open(f'{data_dir}/model.log', 'w') as f:
    f.write(str(model))
print('Model size: {}MB'.format(os.path.getsize(f'{data_dir}/pytorch_model.bin') / 1000000))

# from transformers import BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained(f'{data_dir}/vocab.txt', do_lower_case=True)
# tokenizer.save_pretrained(f'{data_dir}/')