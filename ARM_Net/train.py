import os
import sys
import copy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from ARM_Net.models import create_model, get_config, to_device
from ARM_Net.utils import logger, AverageMeter
from ARM_Net.data_loader import csv_loader


# ----------------------------------------------------------------------------
# Hyper-parameter sampling (replaces Ray's search space + BasicVariantGenerator)
# ----------------------------------------------------------------------------
def resolve_config(space, rng):
    """
    Turn a search space into one concrete config (plain python values).
    Supported entries:
      - Ray-style domains (tune.choice / tune.uniform / tune.loguniform ...): anything with .sample()
      - {'grid_search': [...]}  -> random pick (this script does random search, not a full grid)
      - list / tuple            -> random choice
      - nested dict             -> resolved recursively
      - anything else           -> constant
    """
    out = {}
    for k, v in space.items():
        if hasattr(v, 'sample') and callable(v.sample):
            out[k] = v.sample()
        elif isinstance(v, dict) and 'grid_search' in v:
            options = v['grid_search']
            out[k] = options[rng.integers(len(options))]
        elif isinstance(v, dict):
            out[k] = resolve_config(v, rng)
        elif isinstance(v, (list, tuple)):
            out[k] = v[rng.integers(len(v))]
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, criterion, device, metric='acc', nclass=2):
    net.eval()
    loss_avg = AverageMeter()
    all_labels, all_preds = [], []
    for batch, labels in loader:
        batch, labels = batch.to(device), labels.to(device)
        outputs = net(batch)
        loss_avg.update(criterion(outputs, labels).item())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
    y_true, y_pred = np.concatenate(all_labels), np.concatenate(all_preds)

    if metric == 'acc':
        score = float((y_true == y_pred).mean())
    elif metric == 'auc':
        try:
            score = roc_auc_score(y_true, y_pred)
        except ValueError:
            score = 0.0
    elif metric == 'f1':
        score = f1_score(y_true, y_pred, average='binary' if nclass == 2 else 'macro')
    else:
        raise ValueError(f'unknown metric: {metric}')
    return score, loss_avg.avg


def train_one_epoch(net, loader, optimizer, criterion, device):
    net.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()


def run_trial(config, train_loaders, test_loader, model_name, max_epochs, metric, patience, plogger):
    """
    Train one config. Returns the statistics of the epoch with the best validation score
    (same as Ray's 'last reported result' in the original code, because it only reported improving epochs).
    """
    train_loader, val_loader = train_loaders
    cfg = dict(config)
    cfg['nclass'], cfg['nfeat'] = train_loader.nclass, train_loader.nfeat
    cfg['model'] = model_name

    net = create_model(cfg, plogger)
    device = to_device(net)
    size = sum(p.data.nelement() for p in net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    best, bad_epochs = None, 0
    for epoch in range(max_epochs):
        train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(net, val_loader, criterion, device, metric, cfg['nclass'])
        test_acc, test_loss = evaluate(net, test_loader, criterion, device, metric, cfg['nclass'])
        print(f'epoch-{epoch} val_acc-{val_acc:.4f} test_acc-{test_acc:.4f} config: {config}')

        if best is None or val_acc > best['val_acc']:
            best = dict(val_acc=val_acc, val_loss=val_loss, test_acc=test_acc,
                        test_loss=test_loss, size=size, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                        full_config=cfg)   # includes nclass, nfeat, model
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:   # patience=0 -> stop at the first epoch without improvement
                break
    return best


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def train_arm_net(
    num_samples: int = 1,          # number of random hyper-parameter configs per repeat
    exp_name: str = 'test',        # experiment name used to store logs
    stdout: str = 'log.txt',       # redirect print() output to this file in the log dir, '' / None to disable
    model: str = 'armnet',         # model type
    train_data_dir: str = r'data\transformed_data.csv',  # path to the train CSV file (features..., target in last column)
    test_data_dir: str = r'data\transformed_data.csv',  # path to the test CSV file (features..., target in last column)
    metric: str = 'acc',           # evaluation metric: acc / auc / f1
    valid_perc: float = 0.2,       # validation split taken from the train set
    repeat: int = 5,               # number of repeats with seeds [seed, seed + repeat)
    seed: int = 2021,              # seed for reproducibility
    max_epochs: int = 100,         # max training epochs per config
    patience: int = 0,             # early stopping patience (epochs without val improvement)
):
    args = dict(locals())
    log_dir = f'log/{exp_name}/'
    os.makedirs(log_dir, exist_ok=True)
    plogger = logger(f'{log_dir}stdout.log', True, True).info
    plogger(args)
    if stdout:
        # utf-8 explicitly: on Windows the default cp1250 cannot encode some characters
        sys.stdout = open(f'{log_dir}{stdout}', 'w', encoding='utf-8', buffering=1)

    space = get_config(model)

    repeat_acc, repeat_loss, repeat_rho, repeat_size = [], [], [], []
    for s in range(seed, seed + repeat):
        torch.manual_seed(s)
        np.random.seed(s)
        rng = np.random.default_rng(s)

        # data is loaded once per repeat and shared by all trials (split depends on the seed)
        loaders = csv_loader(train_data_dir, valid_perc=valid_perc, workers=0, seed=s)
        test_loader = csv_loader(test_data_dir, valid_perc=0, workers=0, seed=s)


        records = []
        best_val = -float('inf')
        for trial_id in range(num_samples):
            config = resolve_config(space, rng)
            try:
                result = run_trial(config, loaders, test_loader, model, max_epochs, metric, patience, plogger)
            except Exception as e:
                plogger(f'Trial {trial_id} failed for config {config}: {e!r}')
                continue
            result.update(trial_id=trial_id, config=config)
            if result['val_acc'] > best_val:
                best_val = result['val_acc']
                for r in records: r.pop('state', None)   # free weights of the previous best
            else:
                result.pop('state')
            records.append(result)

        if not records:
            plogger(f'Repeat-{s}\tno successful trials')
            continue

        # store the stats of all trials to csv
        df = pd.DataFrame([
            {'trial_id': r['trial_id'],
             **{f'config/{k}': v for k, v in r['config'].items()},
             **{k: r[k] for k in ('val_acc', 'val_loss', 'test_acc', 'test_loss', 'size', 'epoch')}}
            for r in records
        ])
        df.to_csv(f'{log_dir}results_{s}.csv', index=False)

        best = max(records, key=lambda r: r['val_acc'])
        train_loader = loaders[0]
        torch.save({
            'model_state': best['state'],
            'config': best['full_config'],
            'metric': metric,
            'val_acc': best['val_acc'],
            'test_acc': best['test_acc'],
            'scaler_mean': train_loader.transformer.mean_.tolist(),
            'scaler_scale': train_loader.transformer.scale_.tolist(),
            'classes': train_loader.label_encoder.classes_.tolist(),
        }, f'{log_dir}best_model_{s}.pt')

        plogger(f'Repeat-{s}\tbest trial id: {best["trial_id"]}\tconfig: {best["config"]}\t'
                f'parameter size: {best["size"]}\tbest epoch: {best["epoch"]}\t'
                f'val loss: {best["val_loss"]} val acc: {best["val_acc"]}\t'
                f'test loss: {best["test_loss"]} test acc: {best["test_acc"]}')

        # spearman correlation between val_acc and test_acc (optional, needs >= 2 varying trials)
        if len(df) >= 2 and df['val_acc'].nunique() > 1 and df['test_acc'].nunique() > 1:
            rho, pval = stats.spearmanr(df['val_acc'], df['test_acc'])
        else:
            rho, pval = float('nan'), float('nan')
        plogger(f'Repeat-{s}\tspearman correlation: {rho}({pval})\n')

        repeat_acc.append(best['test_acc']); repeat_loss.append(best['test_loss'])
        repeat_rho.append(rho); repeat_size.append(best['size'])

    if not repeat_acc:
        plogger('No successful runs.')
        return
    plogger(f'Final report results: loss {np.mean(repeat_loss):.4f}/{np.std(repeat_loss):.4f}\t '
            f'accuracy {np.mean(repeat_acc):.4f}/{np.std(repeat_acc):.4f}\t'
            f'spearman correlation {np.nanmean(repeat_rho):.4f}/{np.nanstd(repeat_rho):.4f}\t'
            f'avg parameter size {np.mean(repeat_size):.4f}/{np.std(repeat_size):.4f}')



if __name__ == "__main__":
    train_arm_net(num_samples=1, train_data_dir=r"data\splits\fold0train.csv" ,test_data_dir=r"data\splits\fold0test.csv", max_epochs=2)