DATA_DIR=/app/outputs/

CORPUS_DIR=${DATA_DIR}/data/
TOKENIZER_FILE=${DATA_DIR}/tokenizer.json
OUTPUT_DIR=${DATA_DIR}/pretrain_tfrecords
MAX_SEQ_LENTGH=128
NUM_PROCESSES=4

mkdir -p $CORPUS_DIR
rm -fr $OUTPUT_DIR && mkdir -p $OUTPUT_DIR
cd /app
python3 build_pretraining_dataset_hf.py \
--corpus-dir=$CORPUS_DIR \
--tokenizer-file=$TOKENIZER_FILE \
--output-dir=$OUTPUT_DIR \
--max-seq-length=$MAX_SEQ_LENTGH \
--num-processes=$NUM_PROCESSES

MODEL_NAME=electra_base_wiki40b_ja_mecab_ipadic
python3 run_pretraining.py --data-dir $DATA_DIR --model-name $MODEL_NAME --hparams '{"model_size": "base", "max_seq_length": 256, "generator_hidden_size": 0.33333, "learning_rate": 2e-4, "train_batch_size": 128, "embedding_size": 768, "num_train_steps": 1000000, "vocab_size": 30000}'
