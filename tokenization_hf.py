import json
import os
import tempfile
import unicodedata
from typing import Dict, List, Optional, Union
import MeCab
from tokenizers import (
    AddedToken,
    BertWordPieceTokenizer,
    Encoding,
    EncodeInput,
    InputSequence,
    Tokenizer,
)

def load_tokenizer(tokenizer_file: str) -> Tokenizer:
    """ Load BertWordPieceTokenizer from tokenizer.json. This is necessary due to the following reasons:
    - BertWordPieceTokenizer cannot load from tokenizer.json via .from_file() method
    - Tokenizer.from_file(tokenizer_file) cannot be used because MecabPretokenizer is not a valid native PreTokenizer.
    """
    with open(tokenizer_file) as fp:
        jd = json.loads(fp.read())
        settings = jd['normalizer']
        settings.pop('type')
        vocab_map = jd['model']['vocab']
    with tempfile.TemporaryDirectory() as dname:
        vocab_file = os.path.join(dname, "vocab.txt")
        with open(vocab_file, 'w') as fp:
            fp.write('\n'.join([w for w, vid in sorted(vocab_map.items(), key=lambda x: x[1])]))
        tokenizer = MecabBertWordPieceTokenizer(vocab_file, **settings)

    return tokenizer


class MecabPreTokenizer:
    """ PreTokenizerを継承することはできない """

    def __init__(
        self,
        mecab_dict_path: Optional[str] = None,
        do_lower_case: bool = False,
        space_replacement: Optional[str] = None,
    ):
        """Constructs a MecabPreTokenizer for huggingface tokenizers.
        - space_replacement: Character which is replaced with spaces.
            You might want to use it because MeCab drop spaces by default.
            This can be used to preserve spaces by replacing them with spaces later.
            Special characters like '_' are used sometimes.
        """

        self.do_lower_case = do_lower_case
        self.space_replacement = space_replacement

        mecab_option = (
            f"-Owakati -d {mecab_dict_path}"
            if mecab_dict_path is not None
            else "-Owakati"
        )
        self.mecab = MeCab.Tagger(mecab_option)

    def __call__(self, text: str):
        return self.pre_tokenize_str(text)

    def pre_tokenize_str(self, sequence: str) -> str:
        """
        Pre tokenize the given string
        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`
        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize
        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        text = unicodedata.normalize("NFKC", sequence)
        if self.do_lower_case:
            text = text.lower()
        if self.space_replacement:
            text = text.replace(" ", self.space_replacement)

        return self.mecab.parse(text).strip()


class MecabBertWordPieceTokenizer(BertWordPieceTokenizer):
    """fast tokenizer"""

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        clean_text: bool = True,
        handle_chinese_chars: bool = False,  # for ja
        strip_accents: bool = False,  # for ja
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
        mecab_dict_path: Optional[str] = None,
        space_replacement: Optional[str] = None,
    ):
        """vocab: vocab.txt for WordPiece"""
        super().__init__(
            vocab=vocab,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
            wordpieces_prefix=wordpieces_prefix,
        )
        # from tokenizers.pre_tokenizers import BertPreTokenizer, Sequence
        # self.pre_tokenizer = Sequence([
        #     MecabPreTokenizer(
        #         mecab_dict_path, lowercase, space_replacement
        #     ),
        #     BertPreTokenizer()
        # ])
        self.mecab_pretok = MecabPreTokenizer(
            mecab_dict_path, lowercase, space_replacement
        )

    def encode(
        self,
        sequence: InputSequence,
        pair: Optional[InputSequence] = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        if not is_pretokenized:
            sequence = self.mecab_pretok(sequence)
        return super().encode(sequence, pair, is_pretokenized, add_special_tokens)

    def encode_batch(
        self,
        inputs: List[EncodeInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List[Encoding]:
        if not is_pretokenized:
            # NOTE: reject Tuple[List[str], str] pattern like
            # ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
            inputs = [
                self.mecab_pretok(sequence)
                if isinstance(sequence, str)
                else tuple(map(self.mecab_pretok, sequence))
                for sequence in inputs
            ]

        return super().encode_batch(inputs, is_pretokenized, add_special_tokens)
