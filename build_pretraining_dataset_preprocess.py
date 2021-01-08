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
import re
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
    # parser.add_argument(
    #     "--output-dir",
    #     # required=True,
    #     help="Where to write out the tfrecords."
    # )
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
    parser.add_argument(
        "--wiki40b",
        default=True,
        type=bool,
        help="Whether to use wiki40b data",
    )
    # parser.add_argument(
    #     "--max-seq-length", default=128, type=int, help="Number of tokens per example."
    # )

    args = parser.parse_args()
    # mecab_option = (
    #     f"-Owakati -d {args.mecab_dict_path}"
    #     if args.mecab_dict_path is not None
    #     else "-Owakati"
    # )
    # tagger = MeCab.Tagger(mecab_option)

    def preprocess(data: dict) -> str:
        text = data["text"].decode("utf8")
        text = re.sub("\n+", "\n", text)
        text = text.replace("_START_ARTICLE_", "")
        # text = text.replace("_START_SECTION_", "\n")
        text = re.sub("\n+_START_SECTION_\n+", "\n", text)
        # text = text.replace("_START_PARAGRAPH_", "\n")
        text = re.sub("\n+_START_PARAGRAPH_\n+", "\n", text)
        text = text.replace("_NEWLINE_", "")
        return text.strip()  # tagger.parse(text).strip()

    if args.wiki40b:
        n_files = args.split_factor * args.num_processes
        # for mode in ['train', 'validation', 'test']:
        mode = "train"
        ds = tfds.load("wiki40b/ja", data_dir=args.raw_corpus_dir, split=mode)
        n_dataset = ds.cardinality().numpy()
        n_lines_per_file = n_dataset // n_files
        print(f"#{mode}: {n_dataset}")
        print(f"{n_lines_per_file} lines per file")
        lines_to_write = []
        file_no_prev = -1
        for i, d in enumerate(map(preprocess, ds.as_numpy_iterator())):
            if i % n_lines_per_file == 0 and lines_to_write:
                file_no = i // n_lines_per_file
                with open(
                    os.path.join(args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt"),
                    "w",
                ) as fp:
                    fp.write("\n\n".join(lines_to_write))
                lines_to_write = []
                file_no_prev = file_no
            lines_to_write.append(d)
        if lines_to_write:
            file_no = file_no_prev + 1
            with open(
                os.path.join(args.corpus_dir, f"wiki40b_ja_{mode}_{file_no}.txt"), "w"
            ) as fp:
                fp.write("\n\n".join(lines_to_write))
    else:
        exit(1)

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
