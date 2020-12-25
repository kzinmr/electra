DATA_DIR=/app/outputs/

CORPUS_DIR=${DATA_DIR}/data/
VOCAB_FILE=${DATA_DIR}/vocab.txt
OUTPUT_DIR=${DATA_DIR}/pretrain_tfrecords
MAX_SEQ_LENTGH=128
NUM_PROCESSES=4

mkdir -p $CORPUS_DIR
mkdir -p $OUTPUT_DIR
cd /app
python3 build_pretraining_dataset_tfrecords.py \
--corpus-dir=$CORPUS_DIR \
--vocab-file=$VOCAB_FILE \
--output-dir=$OUTPUT_DIR \
--max-seq-length=$MAX_SEQ_LENTGH \
--num-processes=$NUM_PROCESSES


MODEL_NAME=electra_small_wiki40b_ja_mecab_ipadic
python3 run_pretraining.py --data-dir $DATA_DIR --model-name $MODEL_NAME
# --hparams setting.json
