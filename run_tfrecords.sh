CORPUS_DIR=/app/outputs/data/
VOCAB_FILE=/app/outputs/vocab.txt
OUTPUT_DIR=/app/outputs/tfrecords
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
