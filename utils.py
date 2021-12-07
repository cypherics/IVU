import pickle


def load_pickle(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

