RAW_CORPUS_DIR=/app/outputs/raw
CORPUS_DIR=/app/outputs/data/
VOCAB_FILE=/app/outputs/vocab.txt
TOKENIZER_FILE=/app/outputs/tokenizer.json

# MAX_SEQ_LENTGH=128
NUM_PROCESSES=8
SPLIT_FACTOR=50  # NUM_PROCESSES * SPLIT_FACTOR lines per file

mkdir -p $RAW_CORPUS_DIR
mkdir -p $CORPUS_DIR

cd /app
python3 build_pretraining_dataset_preprocess.py \
--raw-corpus-dir=$RAW_CORPUS_DIR \
--corpus-dir=$CORPUS_DIR \
--vocab-file=$VOCAB_FILE \
--tokenizer-file=$TOKENIZER_FILE \
--num-processes=$NUM_PROCESSES \
--split-factor=$SPLIT_FACTOR \ 
--wiki40b