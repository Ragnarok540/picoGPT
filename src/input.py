class Input:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)


if __name__ == '__main__':
    input = Input('input.txt')
    print(input.chars)
    print(input.vocab_size)
