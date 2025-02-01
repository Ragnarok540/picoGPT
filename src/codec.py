from input import Input


class Codec:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, string):
        return [self.stoi[c] for c in string]

    def decode(self, int_list):
        return ''.join([self.itos[i] for i in int_list])


if __name__ == '__main__':
    i = Input('input.txt')
    c = Codec(i.chars)
    print(c.encode('hi there'))
    print(c.decode(c.encode('hi there')))
