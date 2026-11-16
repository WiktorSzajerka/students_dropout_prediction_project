import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from saint_main.models import SAINT
from saint_main.data_openml import data_prep_train, data_prep_test, DataSetCatCon
from saint_main.utils import count_parameters, classification_scores, mean_sq_error
from saint_main.augmentations import embed_data_mask


def train_saint(
    data_path_train,
    data_path_test,
    categorical_columns=["Marital status", "Application mode", "Course", "Daytime/evening attendance",
               "Previous qualification", "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs",
               "Debtor","Tuition fees up to date", "Gender", "Scholarship holder", "International"],
    task="multiclass",             # 'binary' | 'multiclass' | 'regression'
    cont_embeddings='MLP',     # 'MLP' | 'Noemb' | 'pos_singleMLP'
    embedding_size=32,
    transformer_depth=6,
    attention_heads=8,
    attention_dropout=0.1,
    ff_dropout=0.1,
    attentiontype='colrow',    # 'col' | 'colrow' | 'row' | 'justmlp' | 'attn' | 'attnmlp'
    optimizer='AdamW',         # 'AdamW' | 'Adam' | 'SGD'
    scheduler='cosine',        # 'cosine' | 'linear' (used only with SGD)
    lr=0.0001,
    epochs=100,
    batchsize=256,
    savemodelroot='./saint_models',
    run_name='testrun',
    set_seed=1,
    dset_seed=5,
    datasplit=(.82, .18),
    num_workers=4,
    eval_every=5,
    vision_dset=False,
    # --- pretraining ---
    pretrain=False,
    pretrain_epochs=50,
    pt_tasks=('contrastive', 'denoising'),  # 'contrastive' | 'contrastive_sim' | 'denoising'
    pt_aug=(),                              # 'mixup' | 'cutmix'
    pt_aug_lam=0.1,
    mixup_lam=0.3,
    train_mask_prob=0,
    mask_prob=0,
    ssl_avail_y=0,
    pt_projhead_style='diff',               # 'diff' | 'same' | 'nohead'
    nce_temp=0.7,
    lam0=0.5,
    lam1=10,
    lam2=1,
    lam3=10,
    final_mlp_style='sep',                  # 'common' | 'sep'
):
    """Train SAINT on a CSV file. Returns a dict with the results of the best epoch
    (selected on the validation set) and the path of the saved model."""

    if task not in ('binary', 'multiclass', 'regression'):
        raise ValueError(f"task must be 'binary', 'multiclass' or 'regression', got {task!r}")

    # Bundle everything into `opt` - SAINT_pretrain and get_scheduler expect an options object.
    opt = SimpleNamespace(
        data_path=data_path_train, data_path_test=data_path_test, categorical_columns=categorical_columns, task=task,
        cont_embeddings=cont_embeddings, embedding_size=embedding_size,
        transformer_depth=transformer_depth, attention_heads=attention_heads,
        attention_dropout=attention_dropout, ff_dropout=ff_dropout,
        attentiontype=attentiontype, optimizer=optimizer, scheduler=scheduler,
        lr=lr, epochs=epochs, batchsize=batchsize, savemodelroot=savemodelroot,
        run_name=run_name, set_seed=set_seed, dset_seed=dset_seed,
        vision_dset=vision_dset, pretrain=pretrain, pretrain_epochs=pretrain_epochs,
        pt_tasks=list(pt_tasks), pt_aug=list(pt_aug), pt_aug_lam=pt_aug_lam,
        mixup_lam=mixup_lam, train_mask_prob=train_mask_prob, mask_prob=mask_prob,
        ssl_avail_y=ssl_avail_y, pt_projhead_style=pt_projhead_style, nce_temp=nce_temp,
        lam0=lam0, lam1=lam1, lam2=lam2, lam3=lam3, final_mlp_style=final_mlp_style,
    )
    opt.dtask = 'reg' if opt.task == 'regression' else 'clf'

    modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, opt.run_name)
    os.makedirs(modelsave_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    torch.manual_seed(opt.set_seed)

    print('Loading and processing the dataset, it might take some time.')
    (cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid,
     train_mean, train_std) = data_prep_train(
        opt.data_path, opt.dset_seed, opt.categorical_columns,
        opt.task, datasplit=list(datasplit))
    X_test, y_test = data_prep_test(opt.data_path, opt.categorical_columns,
            opt.task)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    ##### Setting some hyperparams based on inputs and dataset
    _, nfeat = X_train['data'].shape
    if nfeat > 100:
        opt.embedding_size = min(8, opt.embedding_size)
        opt.batchsize = min(64, opt.batchsize)
    if opt.attentiontype != 'col':
        opt.transformer_depth = 1
        opt.attention_heads = min(4, opt.attention_heads)
        opt.attention_dropout = 0.8
        opt.embedding_size = min(32, opt.embedding_size)
        opt.ff_dropout = 0.8

    print(nfeat, opt.batchsize)
    print(opt)

    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=num_workers)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=num_workers)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=num_workers)

    if opt.task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(y_train['data'][:, 0]))

    # Appending 1 for CLS token, this is later used to generate embeddings.
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=opt.embedding_size,
        dim_out=1,
        depth=opt.transformer_depth,
        heads=opt.attention_heads,
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=y_dim,
    )

    if y_dim == 2 and opt.task == 'binary':
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2 and opt.task == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(device)
    elif opt.task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise ValueError(
            f"Task/label mismatch: task={opt.task!r} but y has {y_dim} distinct classes.")

    model.to(device)

    if opt.pretrain:
        from pretraining import SAINT_pretrain
        model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device)

    ## Choosing the optimizer
    sched = None
    if opt.optimizer == 'SGD':
        optim_ = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler
        sched = get_scheduler(opt, optim_)
    elif opt.optimizer == 'Adam':
        optim_ = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optim_ = optim.AdamW(model.parameters(), lr=opt.lr)
    else:
        raise ValueError(f"Unknown optimizer {opt.optimizer!r}")

    best_valid_accuracy = 0
    best_test_auroc = 0
    best_test_accuracy = 0
    best_valid_rmse = float('inf')
    best_test_rmse = None
    best_epoch = None
    model_path = os.path.join(modelsave_path, 'bestmodel.pth')

    print('Training begins now.')
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for data in trainloader:
            optim_.zero_grad()
            # x_categ: categorical data, x_cont: continuous data, y_gts: ground truth,
            # cat_mask: ones like x_categ plus an extra column (CLS token) set to 0,
            # con_mask: ones like x_cont.
            x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in data[:5]]

            # Convert the data to embeddings
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, opt.vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # Select only the representation of the CLS token and apply the MLP to get predictions.
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            if opt.task == 'regression':
                loss = criterion(y_outs, y_gts)
            else:
                loss = criterion(y_outs, y_gts.squeeze())
            loss.backward()
            optim_.step()
            if sched is not None:
                sched.step()
            running_loss += loss.item()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                if opt.task in ['binary', 'multiclass']:
                    accuracy, auroc = classification_scores(
                        model, validloader, device, opt.task, opt.vision_dset)
                    test_accuracy, test_auroc = classification_scores(
                        model, testloader, device, opt.task, opt.vision_dset)

                    print('[EPOCH %d] TRAIN LOSS: %.4f' % (epoch + 1, running_loss))
                    print('[EPOCH %d] VALID ACCURACY: %.3f' %
                          (epoch + 1, accuracy))
                    print('[EPOCH %d] TEST ACCURACY: %.3f' %
                          (epoch + 1, test_accuracy))

                    # Model selection on validation accuracy (as in the original script)
                    if accuracy > best_valid_accuracy:
                        best_valid_accuracy = accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), model_path)
                else:
                    valid_rmse = mean_sq_error(model, validloader, device, opt.vision_dset)
                    test_rmse = mean_sq_error(model, testloader, device, opt.vision_dset)
                    print('[EPOCH %d] TRAIN LOSS: %.4f' % (epoch + 1, running_loss))
                    print('[EPOCH %d] VALID RMSE: %.3f' % (epoch + 1, valid_rmse))
                    print('[EPOCH %d] TEST RMSE: %.3f' % (epoch + 1, test_rmse))
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), model_path)
            model.train()

    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' % total_parameters)

    results = {
        'task': opt.task,
        'best_epoch': best_epoch,
        'total_parameters': total_parameters,
        'cat_dims': len(cat_idxs),
        'con_dims': len(con_idxs),
        'model_path': model_path,
    }
    if opt.task == 'binary':
        print('AUROC on best model:  %.3f' % best_test_auroc)
        results.update(test_auroc=best_test_auroc, test_accuracy=best_test_accuracy,
                       valid_accuracy=best_valid_accuracy)
    elif opt.task == 'multiclass':
        print('Accuracy on best model:  %.3f' % best_test_accuracy)
        results.update(test_accuracy=best_test_accuracy, test_auroc=best_test_auroc,
                       valid_accuracy=best_valid_accuracy)
    else:
        print('RMSE on best model:  %.3f' % best_test_rmse)
        results.update(test_rmse=best_test_rmse, valid_rmse=best_valid_rmse)
    return results


if __name__ == '__main__':
    # Example usage - adjust to your data
    results = train_saint(
        data_path_train='data/splits/fold0test.csv',
        data_path_test='data/splits/fold0train.csv',
        categorical_columns=["Marital status", "Application mode", "Course", "Daytime/evening attendance",
               "Previous qualification", "Mother's qualification", "Father's qualification",
               "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs",
               "Debtor","Tuition fees up to date", "Gender", "Scholarship holder", "International"],
        task='multiclass',
        epochs=10,
        run_name='testrun',
    )
    print(results)