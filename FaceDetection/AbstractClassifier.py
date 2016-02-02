import pickle


class AbstractClassfier:
    def __init__(self):
        pass

    def load(self, path):
        with open(path, 'rb') as infile:
            list_ = pickle.load(infile)
            self.from_list(list_)

    def save(self, path):
        with open(path, 'wb') as outfile:
            list_ = self.to_list()
            pickle.dump(list_, outfile, pickle.HIGHEST_PROTOCOL)

    def to_list(self):
        raise NotImplementedError

    def from_list(self, mat):
        raise NotImplementedError

    def classify(self, x):
        raise NotImplementedError

    def classify_vec(self, vec):
        raise NotImplementedError