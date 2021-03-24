python3 convert_pytorch.py \
--model_name=electra_small_wiki40b_ja_mecab_ipadic \
--ckpt_name=model.ckpt-1000000 \
--discriminator_or_generator=discriminator \
--hparams '{"model_size": "base", "max_seq_length": 256, "generator_hidden_size": 0.33333, "learning_rate": 2e-4, "train_batch_size": 128, "embedding_size": 768, "num_train_steps": 1000000, "vocab_size": 30000}'
