import argparse

import torch
from transformers import (
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    load_tf_weights_in_electra,
)
from transformers.utils import logging

from configure_pretraining import PretrainingConfig

# from util import training_utils

logging.set_verbosity_info()


def get_bert_config(config):
    """Get model hyperparameters based on a pretraining/finetuning config"""
    if config.model_size == "large":
        args = {"hidden_size": 1024, "num_hidden_layers": 24}
    elif config.model_size == "base":
        args = {"hidden_size": 768, "num_hidden_layers": 12}
    elif config.model_size == "small":
        args = {"hidden_size": 256, "num_hidden_layers": 12}
    else:
        raise ValueError("Unknown model size", config.model_size)
    args["vocab_size"] = config.vocab_size
    args.update(**config.model_hparam_overrides)
    # by default the ff size and num attn heads are determined by the hidden size
    args["num_attention_heads"] = max(1, args["hidden_size"] // 64)
    args["intermediate_size"] = 4 * args["hidden_size"]
    args.update(**config.model_hparam_overrides)
    return ElectraConfig.from_dict(args)  # use transformers instead


def convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator
):
    """Convert ELECTRA checkpoint.
    - transformers/src/transformers/models/electra/convert_electra_original_tf_checkpoint_to_pytorch.py
    """
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError(
            "The discriminator_or_generator argument should be either 'discriminator' or 'generator'"
        )

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model,
        config,
        tf_checkpoint_path,
        discriminator_or_generator=discriminator_or_generator,
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_dir",
        default="/app/outputs/models/",
        type=str,
        help="Path to the models.",
    )
    parser.add_argument(
        "--model_name",
        default="electra_small_wiki40b_ja_mecab_ipadic",
        required=True,
        type=str,
        help="Name of the model.",
    )
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        required=True,
        help="Name of the TensorFlow checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
        "'generator'.",
    )
    args = parser.parse_args()

    # 1. export config from config class
    data_dir = args.model_dir + args.model_name
    config = PretrainingConfig(args.model_name, data_dir)
    bert_config = get_bert_config(config)
    config_file = f"{data_dir}/config.json"
    with open(config_file, "w") as f:
        f.write(bert_config.to_json_string())

    # from transformers import BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained(f'{data_dir}/vocab.txt', do_lower_case=True)
    # tokenizer.save_pretrained(f'{data_dir}/')

    # 2. export pytorch model
    tf_checkpoint_path = f"{data_dir}/{args.ckpt_name}"
    d_or_g = args.discriminator_or_generator
    pytorch_dump_path = (
        f"{data_dir}/model_{d_or_g}.pt"
        if args.pytorch_dump_path is None
        else args.pytorch_dump_path
    )
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path, config_file, pytorch_dump_path, d_or_g
    )
