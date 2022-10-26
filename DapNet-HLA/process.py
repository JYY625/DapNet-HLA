def load_data_bicoding(Path):
    data = np.loadtxt(Path, dtype=list)
    data_result = []
    for seq in data:
        seq = str(seq.strip('\n'))
        data_result.append(seq)
    return data_result


def transform_token2index(sequences):
    token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))
    print(token2index)

    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)
    return token_list, max_len


def make_data_with_unified_length(token_list, max_len):
    token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))
    print(token2index)
    data = []
    for i in range(len(token_list)):
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])
    return data

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    # np.random.seed(42)
    random.shuffle(index)
    new_X = X[index]
    new_y = y[index]
    return new_X, new_y


def round_pred(pred):
    list_result = []
    for i in pred:
        if i > 0.5:
            list_result.append(1)
        elif i <= 0.5:
            list_result.append(0)
    return torch.tensor(list_result)

