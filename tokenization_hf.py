import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional

import MeCab
import textspan
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer


def train_custom_tokenizer(
    files: List[str], tokenizer_file: str, **kwargs
) -> BertWordPieceTokenizer:
    tokenizer = BertWordPieceTokenizer(
        handle_chinese_chars=False,  # for ja
        strip_accents=False,  # for ja
    )
    tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())

    tokenizer.train(files, **kwargs)

    # print(
    #     tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
    #         tokenizer._tokenizer.normalizer.normalize_str(open(files[0]).read())
    #     )
    # )

    # Save model as f"vocab-{filename}.txt"
    filename = "wordpiece"
    model_files = tokenizer._tokenizer.model.save(
        str(Path(tokenizer_file).parent), filename
    )
    assert len(model_files) == 1
    new_path = Path(tokenizer_file).parent / "vocab.txt"
    shutil.move(model_files[0], new_path)

    # Set place holder because custom PreTokenzier cannot be serialized.
    tokenizer._tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(tokenizer_file)

    return tokenizer


def load_custom_tokenizer(tokenizer_file: str) -> Tokenizer:
    """Load a Tokenizer from tokenizer.json and set a custome PreTokenizer."""
    # Load all settings.
    tok = Tokenizer.from_file(tokenizer_file)
    # Set custom PreTokenizer.
    tok.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
    return tok


class MecabPreTokenizer:
    def __init__(
        self,
        mecab_dict_path: Optional[str] = None,
        space_replacement: Optional[str] = None,
    ):
        """Constructs a MecabPreTokenizer for huggingface tokenizers.
        - space_replacement: Character which is replaced with spaces.
            You might want to use it because MeCab drop spaces by default.
            This can be used to preserve spaces by replacing them with spaces later.
            Special characters like '_' are used sometimes.
        """

        self.space_replacement = space_replacement

        mecab_option = (
            f"-Owakati -d {mecab_dict_path}"
            if mecab_dict_path is not None
            else "-Owakati"
        )
        self.mecab = MeCab.Tagger(mecab_option)

    def tokenize(self, sequence: str) -> List[str]:
        text = unicodedata.normalize("NFKC", sequence)
        if self.space_replacement:
            text = text.replace(" ", self.space_replacement)
            splits = self.mecab.parse(text).strip().split(" ")
            return [x.replace(self.space_replacement, " ") for x in splits]
        else:
            return self.mecab.parse(text).strip().split(" ")

    def custom_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for char_spans in tokens_spans
            for st, ed in char_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


if __name__ == "__main__":
    s = "今日はいい天気だ\n"
    with open("test.txt", "wt") as fp:
        fp.write("今日はいい天気だ\n")

    fnames = ["test.txt"]
    tokenizer_file = "tokenizer_test.json"
    settings = dict(
        vocab_size=30000,
        min_frequency=1,
        limit_alphabet=1000,
    )
    tokenizer = train_custom_tokenizer(fnames, tokenizer_file, **settings)

    print("load as Tokenizer")
    tok = Tokenizer.from_file(tokenizer_file)
    print(tok.encode(s).tokens)
    tok.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
    print(tok.encode(s).tokens)

    print(tok.normalizer)
    print(tok.pre_tokenizer)
    print(tok.model)
    print(tok.decoder)

    # print("load custom tokenizer")
    # tok = load_custom_tokenizer(tokenizer_file)
    # print(tok.encode(s).tokens)
    # print(tok.token_to_id("[SEP]"))
    # print(tok.token_to_id("[CLS]"))
    # print(tok.token_to_id("[UNK]"))

    # print(tok.normalizer)
    # print(tok.pre_tokenizer)
    # print(tok.model)
    # print(tok.decoder)

    # Save model as f"vocab-{filename}.txt"
    # filename = "wordpiece"
    # model_files = tokenizer._tokenizer.model.save(Path(tokenizer_file).parent, filename)
    # with open(model_files[0]) as fp:
    #     print(fp.read())

    # from tokenizers.trainers import WordPieceTrainer
    # from tokenizers.normalizers import BertNormalizer, NFKC, Sequence
    # from tokenizers.pre_tokenizers import PreTokenizer, BertPreTokenizer
    # from tokenizers.decoders import WordPiece as WordPieceDecoder  # BPEDecoder,

    # model = WordPiece()
    # tokenizer = Tokenizer(model)
    # tokenizer.normalizer = Sequence(
    #     [NFKC(), BertNormalizer(handle_chinese_chars = False, strip_accents = False,)]
    # )
    # tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
    # tokenizer.decoder = WordPieceDecoder()

    # print([tokenizer.pre_tokenizer.pre_tokenize_str(tokenizer.normalizer.normalize_str(open(fn).read())) for fn in fnames])
    # trainer = WordPieceTrainer(
    #     vocab_size=30000,
    #     min_frequency=1,
    #     limit_alphabet=1000,
    # )
    # tokenizer.train(trainer, fnames)
    # print(tokenizer.encode(s).tokens)
    # # save model
    # model_files = tokenizer.model.save('.', 'model_test')
    # print(model_files)
    # with open(model_files[0]) as fp:
    #     print(fp.read())
    # # save tokenizer
    # tokenizer.pre_tokenizer = BertPreTokenizer()
    # tokenizer.save(tok_file)

    # print('load model only')
    # vocab_map = WordPiece.read_file(model_files[0])
    # print(vocab_map)
    # # model = WordPiece(vocab_map, unk_token='[UNK]')
    # # tok = Tokenizer(model)
    # tok = BertWordPieceTokenizer(vocab_map)
    # print(tok.encode(s).tokens)
    # tok._tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())

    # print(tok._tokenizer.normalizer)
    # print(tok._tokenizer.pre_tokenizer)
    # print(tok._tokenizer.decoder)
    # print(tok.encode(s).tokens)

    # extract vocab.txt from tokenizer.json
    # import tempfile
    # with open(tokenizer_file) as fp:
    #     jd = json.loads(fp.read())
    #     vocab_map = jd["model"]["vocab"]
    #     with tempfile.TemporaryDirectory() as dname:
    #         vocab_file = os.path.join(dname, "vocab.txt")
    #         with open(vocab_file, "w") as fp:
    #             fp.write(
    #                 "\n".join(
    #                     [w for w, vid in sorted(vocab_map.items(), key=lambda x: x[1])]
    #                 )
    #             )
    #         # NOTE: WordPiece model can only be loaded from vocab.txt.
    #         model = WordPiece(WordPiece.read_file(vocab_file), unk_token="[UNK]")
    # tok.model = model
