from input import Input
from codec import Codec
import torch


class Loader:
    def __init__(self, input: Input, codec: Codec, train: float = 0.9):
        self.text = input.text
        self.data = torch.tensor(codec.encode(self.text), dtype=torch.long)
        self.n = int(train*len(self.data))
        self.train_data = self.data[:self.n]
        self.val_data = self.data[self.n:]
        self.batch_size = 64 #32
        self.block_size = 256 #8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)

        return x, y


if __name__ == '__main__':
    torch.manual_seed(1337)
    input = Input('input.txt')
    codec = Codec(input.chars)
    loader = Loader(input, codec)
    print(loader.data.shape, loader.data.dtype)
    print(loader.data[:1000])
    print(loader.get_batch('train'))
