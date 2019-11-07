class VocabDictionary:
    def __init__(self, unk=1):
        self.dictionary = {}
        self.unk = unk

    def load_from_index_file(self, path):
        """
        Loads the dictionary from a file in format
        <symbol0> <index0>
        ...
        <symbolN> <indexN>
        TODO: Test this method
        """

        with open(path, encoding="utf-8") as f:
            for line in f:
                symbol, index = line.strip().split()
                self.dictionary[symbol] = int(index)

    def load_from_count_file(self, path, max_size=None):
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

        i = 1
        for symbol, _ in symbols_and_counts_tuple_list:
            self.dictionary[symbol] = i
            i += 1

    def get_index(self, token):
        return self.dictionary.get(token, self.unk)
