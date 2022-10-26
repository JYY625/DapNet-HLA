def load_train_val_bicoding(path_pos_data, path_neg_data):
    sequences_pos = load_data_bicoding(path_pos_data)
    sequences_neg = load_data_bicoding(path_neg_data)

    token_list_pos, max_len_pos = transform_token2index(sequences_pos)
    token_list_neg, max_len_neg = transform_token2index(sequences_neg)
    max_len = max(max_len_pos, max_len_neg)


    Positive_X = make_data_with_unified_length(token_list_pos, max_len)
    Negitive_X = make_data_with_unified_length(token_list_neg, max_len)

    data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])

    np.random.seed(42)
    np.random.shuffle(data_train)

    X = np.array([_[:-1] for _ in data_train])
    y = np.array([_[-1] for _ in data_train])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

def load_in_torch_fmt(X_train, y_train):
    X_train = torch.from_numpy(X_train).long()
    y_train = torch.from_numpy(y_train).long()

    return X_train, y_train

def ytest_ypred_to_file(y, y_pred, out_fn):
    with open(out_fn, 'w') as f:
        for i in range(len(y)):
            f.write(str(y[i]) + '\t' + str(y_pred[i]) + '\n')


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out