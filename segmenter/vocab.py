import logging

logger = logging.getLogger(__name__)


class VocabDictionary:
    def __init__(self, unk="<unk>", pad="<pad>"):
        self.dictionary = {}
        self.tokens = []
        self.unk_symbol = unk
        self.pad_symbol = pad
        self.unk_index = self.add_symbol(self.unk_symbol)
        self.pad_index = self.add_symbol(self.pad_symbol)

    def __len__(self):
        return len(self.dictionary)

    def add_symbol(self, symbol):
        index = len(self.tokens)
        if symbol not in self.tokens:
            self.tokens.append(symbol)
            self.dictionary[symbol] = index
            return index
        else:
            return self.dictionary[symbol]

    def create_from_count_file(self, path, vocab_max_size, word_min_frequency):
        """
        Loads the dictionary from a file containing a list of counts
        <symbol0> <count0>
        ...
        <symbolN> <countN>
        """

        symbols_and_counts_tuple_list = []
        with open(path, encoding="utf-8") as f:
            total_words = 0
            kept_words = 0
            for line in f:
                total_words += 1
                symbol, count = line.strip().split()
                if int(count) >= word_min_frequency:
                    symbols_and_counts_tuple_list.append((symbol, int(count)))
                    kept_words += 1
        if word_min_frequency > 0:
            logger.info(f"{total_words - kept_words} words out of the original {total_words} have been excluded due "
                        f"to not reaching the minimum frequency {word_min_frequency}")

        if vocab_max_size is not None and vocab_max_size > len(symbols_and_counts_tuple_list):
            init_size = len(symbols_and_counts_tuple_list)
            symbols_and_counts_tuple_list = sorted(symbols_and_counts_tuple_list, key=lambda x: x[1], reverse=True)

            symbols_and_counts_tuple_list = symbols_and_counts_tuple_list[:vocab_max_size]
            logger.info(f"Vocabulary max size exceeded. Original size: {init_size}, cut down to size {vocab_max_size}")

        for symbol, _ in symbols_and_counts_tuple_list:
            self.add_symbol(symbol)

    def get_index(self, token):
        return self.dictionary.get(token, self.unk_index)
