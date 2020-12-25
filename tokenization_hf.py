from typing import Dict, List, Optional, Union

from tokenizers import BertWordPieceTokenizer
from tokenizers.models import WordPiece


class FullTokenizer(BertWordPieceTokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        clean_text: bool = True,
        handle_chinese_chars: bool = False,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ vocab: vocab.txt for WordPiece
        """
        self.wordpieces_prefix = wordpieces_prefix
        super().__init__(
            vocab=vocab,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
            wordpieces_prefix=wordpieces_prefix,
        )

    def tokenize(self, text):
        return self.encode(text).tokens

    def tokenize_as_ids(self, text):
        return self.encode(text).ids

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
    ):
        """ Train the model using the given files """

        super().train(
            files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            wordpieces_prefix=self.wordpieces_prefix,
        )
