class CharTokenizer:
    def __init__(self, text: str) -> None:
        

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...