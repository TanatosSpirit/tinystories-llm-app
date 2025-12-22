import pickle

class BPETokenizer:
    def __init__(self, path_to_load):
        self.use_bounds_tokens = True
        self.pair2token = {}
        self.token2pair = {}
        self._load(path_to_load)

    def _init_vocabulary(self, token2pair = None, pair2token = None):
        self.token2pair, self.pair2token = token2pair, pair2token
        self.special_tokens = {0,1,2}

        base_chars = ['\n'] + [chr(i) for i in range(32, 127)]
        self.id_to_ascii_code = {i + len(self.special_tokens): ord(char) for i, char in enumerate(base_chars)}

        self.min_idx = len(self.special_tokens)
        self.max_idx = self.min_idx + len(base_chars)

    def encode(self, charCodes):
        if self.use_bounds_tokens:
            charCodes = charCodes + [1]

        while len(charCodes) >= 2:
            pairs = {}
            for i in range(len(charCodes) - 1):
                pair = (charCodes[i], charCodes[i + 1])
                if pair in self.pair2token:
                    if pair not in pairs:
                        pairs[pair] = []
                    pairs[pair].append(i)

            if not pairs:
                break

            best_pair = min(pairs.keys(), key=lambda p: self.pair2token[p])
            best_token = self.pair2token[best_pair]

            a, b = best_pair
            new_tokens = []
            i = 0
            while i < len(charCodes):
                if i < len(charCodes) - 1 and charCodes[i] == a and charCodes[i + 1] == b:
                    new_tokens.append(best_token)
                    i += 2
                else:
                    new_tokens.append(charCodes[i])
                    i += 1
            charCodes = new_tokens

        return charCodes

    def decode(self, tokens: list[int]) -> str:
        base_tokens = []

        for token in tokens:
            stack = [token]

            while stack:
                current = stack.pop()

                if self.min_idx <= current < self.max_idx:
                    base_tokens.append(self.id_to_ascii_code[current])
                elif current in {0, 1, 2}:
                    continue
                else:
                    a, b = self.token2pair[current]

                    stack.append(b)
                    stack.append(a)

        text = ''.join(chr(t) for t in base_tokens)
        return text

    def _load(self, path: str):
        print(f"Загрузка словарей {path}...")
        with open(path, 'rb') as f:
            vocab_mappings = pickle.load(f)

        self._init_vocabulary(vocab_mappings['token2pair'], vocab_mappings['pair2token'])
        print(f"Готово! Загружено {len(vocab_mappings)} словарей из кэша.")
