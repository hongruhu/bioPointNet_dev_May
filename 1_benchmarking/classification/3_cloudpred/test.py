def cloudpred_train(Xtrain, Xvalid, centers=2, class_num=2):
    X = np.concatenate([x for (x, *_) in Xtrain])
    gm = []
    for X, *_ in Xtrain:
        gm.append(X)
    gm = np.concatenate(gm)
    model = sklearn.mixture.GaussianMixture(n_components = centers, covariance_type='diag')
    gm = model.fit(gm)
    component = [cloudpred.cloudpred.Gaussian(torch.Tensor(gm.means_[i, :]),
                          torch.Tensor(1. / gm.covariances_[i, :])) for i in range(centers)]
    mixture = cloudpred.cloudpred.Mixture(component, gm.weights_)
    classifier = cloudpred.cloudpred.DensityClassifier(mixture, centers, class_num)
    X = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xtrain])
    y = torch.LongTensor([y for (X, y, *_) in Xtrain])
    Xv = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xvalid])
    yv = torch.LongTensor([y for (X, y, *_) in Xvalid])
    print('iteration')
    # Set weights of classifier
    for lr in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
        print('?')
        optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.modules.CrossEntropyLoss()
        best_loss = float("inf")
        best_model = copy.deepcopy(classifier.pl.state_dict())
        print("Learning rate: " + str(lr))
        for i in tqdm(range(1000)):   
            z = classifier.pl(X)
            loss = criterion(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            zv = classifier.pl(Xv)
            loss = criterion(zv, yv)
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(classifier.pl.state_dict())
        print('done')
        classifier.pl.load_state_dict(best_model)
    print('finally done')
    return classifier