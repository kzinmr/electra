# coding=utf-8
"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import os
import re

import tensorflow_datasets as tfds

import tokenization_hf


def wiki40b_preprocess(data) -> str:
    text = data.decode("utf-8")
    text = re.sub("\n+", "\n", text)
    text = text.replace("_START_ARTICLE_", "")
    # text = text.replace("_START_SECTION_", "\n")
    text = re.sub("\n+_START_SECTION_\n+", "\n", text)
    # text = text.replace("_START_PARAGRAPH_", "\n")
    text = re.sub("\n+_START_PARAGRAPH_\n+", "\n", text)
    text = text.replace("_NEWLINE_", "\n")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-corpus-dir",
        # required=True,
        default="/app/outputs/raw",
        help="Location of raw text files.",
    )
    parser.add_argument(
        "--corpus-dir",
        # required=True,
        default="/app/outputs/data",
        help="Location of pre-training text files.",
    )
    parser.add_argument(
        "--vocab-file",
        # required=True,
        default="/app/outputs/vocab.txt",
        help="Where to write out vocabulary file.",
    )
    parser.add_argument(
        "--tokenizer-file",
        # required=True,
        default="/app/outputs/tokenizer.json",
        help="Where to write out tokenizer setting file.",
    )
    parser.add_argument(
        "--num-processes",
        default=8,
        type=int,
        help="Parallelize across multiple processes.",
    )
    parser.add_argument(
        "--process-factor",
        default=50,
        type=int,
        help="The pretraining texts are splitted into (split-factor * num-processes) files",
    )
    parser.add_argument(
        "--blanks-separate-docs",
        default=True,
        type=bool,
        help="Whether blank lines indicate document boundaries.",
    )
    parser.add_argument("--wiki40b", action="store_true")

    args = parser.parse_args()

    denom = args.process_factor * args.num_processes
    if args.wiki40b:
        # for mode in ['train', 'validation', 'test']:
        mode = "train"

        ds = tfds.load(
            name="wiki40b/ja",
            shuffle_files=False,
            download=True,
            data_dir=args.raw_corpus_dir,
        )
        n_document_batch = 128
        train_ds = ds[mode].batch(n_document_batch).prefetch(10)

        n_docs = ds[mode].cardinality().numpy()
        n_docs_per_file = n_docs // denom
        docs_to_write = []
        n_current_docs = 0
        file_no = 0
        for example in tfds.as_numpy(train_ds):  # example.shape == (128,)
            for doc_text in map(wiki40b_preprocess, example["text"]):
                docs_to_write.append(
                    "\n".join(
                        [
                            line.strip()
                            for line in doc_text.split("\n")
                            if line.strip() and line.endswith("。")
                        ]
                    )
                )
                n_current_docs += 1
                if n_current_docs > n_docs_per_file and docs_to_write:
                    file_no += 1
                    filepath = os.path.join(
                        args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt"
                    )
                    with open(filepath, "w") as fp:
                        fp.write("\n\n".join(docs_to_write))
                    docs_to_write = []
                    n_current_docs = 0
        if docs_to_write:
            file_no += 1
            filepath = os.path.join(args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt")
            with open(filepath, "w") as fp:
                fp.write("\n\n".join(docs_to_write))

    else:
        doc_dir = args.raw_corpus_dir
        n_docs = sum(
            [
                sum([1 for fn in files if fn.endswith(".txt")])
                for _, _, files in os.walk(doc_dir)
            ]
        )
        n_docs_per_file = n_docs // denom
        docs_to_write = []
        n_current_docs = 0
        file_no = 0
        for cur, dirs, files in os.walk(doc_dir):
            doc_paths = [os.path.join(cur, fn) for fn in files if fn.endswith(".txt")]
            for doc_path in doc_paths:  # one doc per one file
                with open(doc_path) as fp:
                    doc_lines = [l.strip() for l in fp.read().split("\n") if l.strip()]
                    docs_to_write.append("\n".join(doc_lines))
                    n_current_docs += 1
                if n_current_docs > n_docs_per_file and docs_to_write:
                    file_no += 1
                    filepath = os.path.join(args.corpus_dir, f"corpus_{file_no}.txt")
                    with open(filepath, "w") as fp:
                        fp.write("\n\n".join(docs_to_write))
                    docs_to_write = []
                    n_current_docs = 0
        if docs_to_write:
            file_no += 1
            filepath = os.path.join(args.corpus_dir, f"corpus_{file_no}.txt")
            with open(filepath, "w") as fp:
                fp.write("\n\n".join(docs_to_write))

    print("Fit wordpiece tokenzier")
    fnames = [
        os.path.join(args.corpus_dir, fn) for fn in sorted(os.listdir(args.corpus_dir))
    ]

    settings = dict(
        vocab_size=30000,
        min_frequency=5,
        limit_alphabet=6000,
    )
    tokenization_hf.train_custom_tokenizer(fnames, args.tokenizer_file, **settings)

    print("processed")


if __name__ == "__main__":
    main()
