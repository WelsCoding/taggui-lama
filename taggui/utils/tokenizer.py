from pathlib import Path

from tokenizers import Tokenizer


class TokenizerOutput:
    """Mimics the parts of the transformers tokenizer output that are used."""
    def __init__(self, input_ids: list[int]):
        self.input_ids = input_ids


class ClipTokenizer:
    """
    A minimal CLIP tokenizer based on the `tokenizers` library. It replaces
    the `AutoTokenizer` from `transformers` (which pulled in PyTorch) for the
    Stable Diffusion token counter. The tokenizer file is the one from
    `clip-vit-base-patch32`.
    """
    def __init__(self, tokenizer_path: Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def __call__(self, text: str) -> TokenizerOutput:
        return TokenizerOutput(self.tokenizer.encode(text).ids)
