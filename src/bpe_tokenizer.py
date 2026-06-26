class CompleteBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}

    def train(self, text):
        # 1. Pre-tokenize: Split text into words and add end-of-word symbol '</w>'
        words = text.split()
        word_freqs = {}
        for word in words:
            # e.g., 'hello' -> ('h', 'e', 'l', 'l', 'o', '</w>')
            token_tuple = tuple(list(word) + ['</w>'])
            word_freqs[token_tuple] = word_freqs.get(token_tuple, 0) + 1

        # Initialize base vocabulary with unique characters
        unique_chars = set()
        for word_tuple in word_freqs:
            unique_chars.update(word_tuple)
        
        self.vocab = {char: idx for idx, char in enumerate(sorted(list(unique_chars)))}
        
        # 2. Iteratively merge the most frequent pairs
        while len(self.vocab) < self.vocab_size:
            pair_counts = {}
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
            
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]
            
            # Record the merge rule and update vocab
            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)
            
            # Update our word frequency dictionary with the merged pair
            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                new_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == best_pair:
                        new_tuple.append(new_token)
                        i += 2
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_tuple)] = freq
            word_freqs = new_word_freqs

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text):
        # Converts raw text into a list of token IDs
        tokens = []
        for word in text.split():
            word_tokens = list(word) + ['</w>']
            # Apply learned merges in order
            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i+1]) == pair:
                        word_tokens[i:i+2] = [merged]
                    else:
                        i += 1
            tokens.extend([self.vocab[t] for t in word_tokens if t in self.vocab])
        return tokens

    def decode(self, ids):
        # Converts token IDs back into raw text
        text = "".join([self.inverse_vocab[idx] for idx in ids])
        return text.replace('</w>', ' ')
        