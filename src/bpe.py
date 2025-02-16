from input import Input
from base_tokenizer import BaseTokenizer


class Tokenizer(BaseTokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = BaseTokenizer.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = BaseTokenizer.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"""merge {i+1}/{num_merges}: {pair} ->
                      {idx} ({vocab[idx]}) had {stats[pair]} occurrences""")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = BaseTokenizer.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = BaseTokenizer.merge(ids, pair, idx)
        return ids


if __name__ == '__main__':
    input = Input('input.txt')
    tokenizer = Tokenizer()
    tokenizer.train(input.text, 400, True)
    print(tokenizer.merges)
    print(tokenizer.vocab)
    test = "this is a text I have now for you there"
    print(test)
    test_encoded = tokenizer.encode(test)
    print(test_encoded)
    test_decoded = tokenizer.decode(test_encoded)
    print(test_decoded)
    print(test == test_decoded)
