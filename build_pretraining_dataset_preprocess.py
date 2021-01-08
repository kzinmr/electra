# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import os

import tensorflow_datasets as tfds

import tokenization_hf


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
        "--split-factor",
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

    n_files = args.split_factor * args.num_processes
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

        n_dataset = ds.cardinality().numpy()
        n_lines_per_file = n_dataset // n_files
        docs_to_write = []
        n_current_lines = 0
        file_no = 0
        for example in tfds.as_numpy(train_ds):  # example.shape == (128,)
            for doc_text in example["text"]:
                doc_lines = [
                    line.strip()
                    for line in doc_text.decode("utf-8").split("\n")
                    if line.strip()  # and line.endswith("。")
                ]
                docs_to_write.append("\n".join(doc_lines))
                n_current_lines += len(doc_lines)
                if n_current_lines > n_lines_per_file and docs_to_write:
                    file_no += 1
                    filepath = os.path.join(
                        args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt"
                    )
                    with open(filepath, "w") as fp:
                        fp.write("\n\n".join(docs_to_write))
                    docs_to_write = []
                    n_current_lines = 0
        if docs_to_write:
            file_no += 1
            filepath = os.path.join(args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt")
            with open(filepath, "w") as fp:
                fp.write("\n\n".join(docs_to_write))

    else:
        doc_dir = args.raw_corpus_dir
        n_dataset = sum(
            [
                sum([1 for fn in files if fn.endswith(".txt")])
                for _, _, files in os.walk(doc_dir)
            ]
        )
        n_lines_per_file = n_dataset // n_files
        docs_to_write = []
        n_current_lines = 0
        file_no = 0
        for cur, dirs, files in os.walk(doc_dir):
            doc_paths = [os.path.join(cur, fn) for fn in files if fn.endswith(".txt")]
            for doc_path in doc_paths:  # one doc per one file
                with open(doc_path) as fp:
                    doc_lines = [l.strip() for l in fp.read().split("\n") if l.strip()]
                    docs_to_write.append("\n".join(doc_lines))
                    n_current_lines += len(doc_lines)
                if n_current_lines > n_lines_per_file and docs_to_write:
                    file_no += 1
                    filepath = os.path.join(args.corpus_dir, f"corpus_{file_no}.txt")
                    with open(filepath, "w") as fp:
                        fp.write("\n\n".join(docs_to_write))
                    docs_to_write = []
                    n_current_lines = 0
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
        min_frequency=2,
        limit_alphabet=1000,
    )
    tokenization_hf.train_custom_tokenizer(fnames, args.tokenizer_file, **settings)

    print("processed")


if __name__ == "__main__":
    main()
