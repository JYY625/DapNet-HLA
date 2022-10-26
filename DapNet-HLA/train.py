def train(train_loader, val_loader):
    model = Emb_CNNGRU_ATT().to(device)

    criterion = nn.BCELoss(size_average=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []

    best_acc = 0
    patience = 0
    patience_limit = 200

    epoch_list = []
    torch_val_best, torch_val_y_best = torch.tensor([]), torch.tensor([])

    for epoch in range(EPOCH):
        repres_list, label_list = [], []

        torch_train, torch_train_y = torch.tensor([]), torch.tensor([])
        torch_val, torch_val_y = torch.tensor([]), torch.tensor([])

        model.train()  # !!! Train
        correct = 0
        train_loss = 0
        for step, (train_x, train_y) in enumerate(train_loader):

            train_x = Variable(train_x, requires_grad=False).to(device)  # ----cuda
            train_y = Variable(train_y, requires_grad=False).to(device)  # ----cuda
            fx, presention = model(train_x)  # torch.Size([256, 1])
            loss = criterion(fx.squeeze(), train_y.type(torch.FloatTensor).to(device))  # B

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            repres_list.extend(presention.cpu().detach().numpy())
            label_list.extend(train_y.cpu().detach().numpy())

            pred = round_pred(fx.data.cpu().numpy()).to(device)  # B
            correct += pred.eq(train_y.view_as(pred)).sum().item()

            train_loss += loss.item() * len(train_y)

            # pred_prob = fx[:,1] 									#C
            pred_prob = fx  # B
            torch_train = torch.cat([torch_train, pred_prob.data.cpu()], dim=0)
            torch_train_y = torch.cat([torch_train_y, train_y.data.cpu()], dim=0)

            if (step + 1) % 10 == 0:
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, EPOCH,
                                                                    step + 1, len(X_train) // BATCH_SIZE,
                                                                    loss.item()))

        train_losses.append(train_loss / len(X_train))
        epoch_list.append(epoch)

        accuracy_train = 100. * correct / len(X_train)
        # ：Epoch: 1, Loss: 0.64163, Training set accuracy: 908/1426 (63.675%)
        # print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
        #     epoch + 1, loss.item(), correct, len(X_train), accuracy_train))
        train_acc.append(accuracy_train)

        model.eval()  # !!! Valid
        val_loss = 0
        correct = 0
        repres_list_valid, label_list_valid = [], []

        with torch.no_grad():
            for step, (valid_x, valid_y) in enumerate(val_loader):  # val_loader
                valid_x = Variable(valid_x, requires_grad=False).to(device)
                valid_y = Variable(valid_y, requires_grad=False).to(device)

                optimizer.zero_grad()  # --->>
                y_hat_val, presention_valid = model(valid_x)

                loss = criterion(y_hat_val.squeeze(), valid_y.type(torch.FloatTensor).to(device)).item()  # B
                val_loss += loss * len(valid_y)

                repres_list_valid.extend(presention_valid.cpu().detach().numpy())
                label_list_valid.extend(valid_y.cpu().detach().numpy())

                pred_val = round_pred(y_hat_val.data.cpu().numpy()).to(device)  # B

                # get the index of the max log-probability
                correct += pred_val.eq(valid_y.view_as(pred_val)).sum().item()

                # pred_prob = y_hat.max(1, keepdim = True)[0]
                # pred_prob_val = y_hat_val[:,1] 					#C
                pred_prob_val = y_hat_val  # B
                torch_val = torch.cat([torch_val, pred_prob_val.data.cpu()], dim=0)
                torch_val_y = torch.cat([torch_val_y, valid_y.data.cpu()], dim=0)

        val_losses.append(val_loss / len(X_valid))  # all loss / all sample
        accuracy_valid = 100. * correct / len(X_valid)
        val_acc.append(accuracy_valid)
        # 保存------------------------------------------------->>>
        cur_acc = accuracy_valid
        # np.mean([True,True,False])-->0s.6666666666666666
        is_best = bool(cur_acc > best_acc)
        best_acc = max(cur_acc, best_acc)

        if is_best:
            torch_val_best = torch_val
            torch_val_y_best = torch_val_y
            torch_train_best = torch_train
            torch_train_y_best = torch_train_y
            valid_index_all = calculateScore(torch_val_y_best, torch_val_best)
            train_index_all = calculateScore(torch_train_y, torch_train)
            valid_index = get_index(valid_index_all)
            train_index = get_index(train_index_all)

            print("\n")

    return train_index, valid_index, torch_val_best, torch_val_y_best, torch_train_best, torch_train_y_best

