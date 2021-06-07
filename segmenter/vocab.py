class VocabDictionary:
    def __init__(self, include_special=True,unk="<unk>", pad="<pad>"):
        self.dictionary = {}
        self.tokens = []
        if include_special:
            self.unk = unk
            self.pad = pad
            self.pad_index = self.add_symbol(self.pad)
            self.unk_index = self.add_symbol(self.unk)
        else:
            self.pad_index = None
            self.unk_index = None
    def __len__(self):
        return len(self.dictionary)

    def load_from_index_file(self, path):
        """
        Loads the dictionary from a file in format
        <symbol0> <index0>
        ...
        <symbolN> <indexN>
        TODO: Test this method
        """
        raise Exception
        with open(path, encoding="utf-8") as f:
            for line in f:
                symbol, index = line.strip().split()
                self.dictionary[symbol] = int(index)

    def add_symbol(self,symbol):
        index = len(self.tokens)
        if symbol not in self.tokens:
            self.tokens.append(symbol)
            self.dictionary[symbol] = index
            return index
        else:
            return self.dictionary[symbol]

    def create_from_count_file(self, path, max_size=None):
        """
        Loads the dictionary from a file containing a list of counts
        <symbol0> <count0>
        ...
        <symbolN> <countN>
        """

        symbols_and_counts_tuple_list=[]
        with open(path, encoding="utf-8") as f:
            for line in f:
                symbol, count = line.strip().split()
                symbols_and_counts_tuple_list.append((symbol, int(count)))

        if max_size is not None and max_size > len(symbols_and_counts_tuple_list):
            symbols_and_counts_tuple_list = sorted(symbols_and_counts_tuple_list, key=lambda x: x[1], reverse=True)

            symbols_and_counts_tuple_list = symbols_and_counts_tuple_list[:max_size]

        for symbol, _ in symbols_and_counts_tuple_list:
            self.add_symbol(symbol)

    def get_index(self, token):
        return self.dictionary.get(token, self.unk_index)
