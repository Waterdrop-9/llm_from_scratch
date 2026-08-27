import re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        # a mapping from byte to id
        self.byte_to_id = {v: k for k, v in vocab.items()}
        # a rank dict merge : rank
        self.merge_to_rank = {merge: idx for idx, merge in enumerate(merges)}
        # a list of special tokens
        special_token_bytes = [special_tok.encode("UTF-8") for special_tok in special_tokens or []]
        for special_token in special_token_bytes:
            if special_token not in self.byte_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = special_token
                self.byte_to_id[special_token] = token_id
        self.special_tokens = sorted(
            set(special_token_bytes),
            key=lambda token: (-len(token), token),
        )


    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_text_segment(text)

        special_pattern = re.compile(
            "|".join(re.escape(token.decode("UTF-8")) for token in self.special_tokens)
        )
        token_ids = []
        last_end = 0
        for match in special_pattern.finditer(text):
            token_ids.extend(self._encode_text_segment(text[last_end:match.start()]))
            special_token = match.group().encode("UTF-8")
            token_ids.append(self.byte_to_id[special_token])
            last_end = match.end()
        token_ids.extend(self._encode_text_segment(text[last_end:]))
        return token_ids

    def _encode_text_segment(self, text: str) -> list[int]:
        tokens = [bytes([byte]) for byte in text.encode("UTF-8")]

        while True:
            best_merge = None
            best_rank = float("inf")
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_to_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_merge = pair
                    best_rank = rank

            if best_merge is None:
                break

            merged_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_merge:
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens

        token_ids = []
        for token in tokens:
            if token not in self.byte_to_id:
                raise ValueError(f"Token {token!r} not found in vocabulary.")
            token_ids.append(self.byte_to_id[token])
        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        bytes_output = []
        for id in ids:
            if id not in self.vocab:
                raise ValueError(f"ID {id} not found in vocabulary.")
            bytes_output.append(self.vocab[id])
        return b"".join(bytes_output).decode("UTF-8", errors="replace")
