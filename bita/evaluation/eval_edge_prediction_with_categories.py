import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

@torch.no_grad()
def eval_edge_prediction_with_categories(
    model,
    negative_edge_sampler,
    data,
    n_neighbors=20,
    device=None,
    edge_criterion=None,
    category_criterion=None,
    ignore_classes=(),
):
    """Link-prediction and category metrics over `data`.

    `ignore_classes` (UNKNOWN: edges without a label) are scored for link
    prediction like any edge but left out of every CATEGORY metric, and the
    head is never credited or blamed for predicting them: argmax runs over the
    labelled classes only.

    Per-class accuracy of a class ABSENT from `data` is NaN, not 0.0. A 0.0
    there read as "the head never recalls this class" -- the health check
    reported it as a dead class -- when the split simply holds none of it.
    """
    model.eval()
    if device is None:
        device = model.device

    batch_size = 200
    num_instances = len(data.sources)
    num_batches = (num_instances + batch_size - 1) // batch_size

    all_pos_scores, all_neg_scores = [], []
    all_true_labels, all_pred_labels = [], []
    all_category_logits = []

    total_edge_loss = 0.0
    total_category_loss = 0.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(num_instances, start_idx + batch_size)

        sources_batch = data.sources[start_idx:end_idx]
        destinations_batch = data.destinations[start_idx:end_idx]
        edge_idxs_batch = data.edge_idxs[start_idx:end_idx]
        timestamps_batch = data.timestamps[start_idx:end_idx]
        categories_batch = data.labels[start_idx:end_idx]

        size = len(sources_batch)
        # Same-capture negatives where the sampler knows the captures (utils.RandEdgeSampler).
        _, negatives_batch = negative_edge_sampler.sample(size, sources=sources_batch, destinations=destinations_batch)

        pos_score, neg_score, category_logits = model.compute_edge_probabilities_and_categories(
            sources_batch,
            destinations_batch,
            negatives_batch,
            timestamps_batch,
            edge_idxs_batch,
            n_neighbors=n_neighbors
        )

        pred_categories = torch.argmax(category_logits, dim=1).cpu().numpy()
        # Arrays, not .extend(): a list of numpy scalars is ~40 B per edge,
        # twice over -- ~1.6 GB across a 20M-edge validation split.
        all_true_labels.append(np.array(categories_batch))
        all_pred_labels.append(pred_categories)
        all_category_logits.append(category_logits.cpu().numpy())
        all_pos_scores.append(pos_score.cpu().numpy())
        all_neg_scores.append(neg_score.cpu().numpy())

        # Edge loss
        if edge_criterion is not None:
            pos_label = torch.ones_like(pos_score, device=device)
            neg_label = torch.zeros_like(neg_score, device=device)
            edge_loss = edge_criterion(pos_score, pos_label) + edge_criterion(neg_score, neg_label)
            total_edge_loss += edge_loss.item()

        # Category loss
        if category_criterion is not None:
            true_labels_tensor = torch.tensor(categories_batch, dtype=torch.long, device=device)
            cat_loss = category_criterion(category_logits, true_labels_tensor)
            total_category_loss += cat_loss.item()

    # Aggregate
    # Same values and dtypes np.array(list of the scalars) produced.
    y_true = np.concatenate(all_true_labels) if all_true_labels else np.array([])
    y_pred = np.concatenate(all_pred_labels) if all_pred_labels else np.array([])
    category_logits = np.concatenate(all_category_logits)
    pos_scores = np.concatenate(all_pos_scores)
    neg_scores = np.concatenate(all_neg_scores)

    ignore = [int(c) for c in ignore_classes if 0 <= int(c) < category_logits.shape[1]]
    if ignore:
        category_logits[:, ignore] = -np.inf
        keep = ~np.isin(y_true, ignore)
        y_true = y_true[keep]
        category_logits = category_logits[keep]
        y_pred = np.argmax(category_logits, axis=1) if len(category_logits) else y_pred[:0]
    y_scores = torch.softmax(torch.tensor(category_logits), dim=1).numpy()

    # Binary classification (link prediction)
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    auc_score = roc_auc_score(all_labels, all_scores)
    avg_precision = average_precision_score(all_labels, all_scores)

    if len(y_true) == 0:
        # Every edge of this split is unlabelled: link metrics only.
        nan, C = float("nan"), category_logits.shape[1]
        per = {c: nan for c in range(C) if c not in ignore}
        return (avg_precision, auc_score, nan, nan, nan,
                total_category_loss / num_batches if category_criterion else None,
                {'edge_loss': total_edge_loss / num_batches if edge_criterion else None},
                dict(per), nan, nan, nan, nan, nan, nan, nan, nan, y_true, y_scores,
                dict(per), dict(per), dict(per), dict(per))

    # Hits@K and MRR, vectorised.
    #
    # These were Python loops over every evaluation sample, each doing its own
    # argsort and an O(C) membership scan. At a million interactions that is the
    # dominant cost of an epoch. The rank of the true class is just one plus the
    # number of classes scoring strictly higher, which needs no sort at all.
    #
    # Tie handling: ranks are optimistic (a tie with the true class does not
    # push it down). The previous argsort-based version broke ties by class
    # index. Exact ties between float softmax outputs are vanishingly rare, and
    # test_eval_metrics_vectorisation.py pins the two against each other.
    row = np.arange(len(y_true))
    true_class_score = y_scores[row, y_true]
    correct_ranks = 1 + (y_scores > true_class_score[:, None]).sum(axis=1)

    hits_at_1 = float(np.mean(correct_ranks <= 1))
    hits_at_3 = float(np.mean(correct_ranks <= 3))
    hits_at_5 = float(np.mean(correct_ranks <= 5))
    mrr = float(np.mean(1.0 / correct_ranks))

    # Overall classification metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Confusion matrix for FPR/FNR/TPR/TNR
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(category_logits.shape[1]))
    fpr, fnr, tpr, tnr = {}, {}, {}, {}
    TP_total, FP_total, FN_total, TN_total = 0, 0, 0, 0
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

        fpr[i] = FP / (FP + TN + 1e-9)
        fnr[i] = FN / (FN + TP + 1e-9)
        tpr[i] = TP / (TP + FN + 1e-9)
        tnr[i] = TN / (TN + FP + 1e-9)

    # Global (macro) confusion-based metrics
    fpr_macro = FP_total / (FP_total + TN_total + 1e-9)
    fnr_macro = FN_total / (FN_total + TP_total + 1e-9)
    tpr_macro = TP_total / (TP_total + FN_total + 1e-9)
    tnr_macro = TN_total / (TN_total + FP_total + 1e-9)

    # Per-class metrics
    num_classes = category_logits.shape[1]
    precision_by_class = {}
    recall_by_class = {}
    f1_by_class = {}
    auc_by_class = {}
    mrr_by_class = {}
    accuracy_by_class = {}

    for c in range(num_classes):
        if c in ignore:
            continue
        true_indices = (y_true == c)
        if true_indices.sum() == 0:
            accuracy_by_class[c] = float("nan")     # absent from this split, not "never recalled"
        else:
            correct_preds = (y_pred[true_indices] == c).sum()
            accuracy_by_class[c] = correct_preds / true_indices.sum()

    for c in range(num_classes):
        if c in ignore:
            continue
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        y_score_class = y_scores[:, c]

        precision_by_class[c] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall_by_class[c] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_by_class[c] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # Undefined with one class present: NaN, without asking sklearn (which
        # warned on every such slice, every epoch, and returned NaN anyway).
        if 0 < y_true_bin.sum() < len(y_true_bin):
            auc_by_class[c] = roc_auc_score(y_true_bin, y_score_class)
        else:
            auc_by_class[c] = float('nan')

        # Per-class MRR, vectorised.
        #
        # This was the single worst hot spot in the file: `np.where(ranks_c ==
        # idx)[0][0]` is a full O(N) scan of the ranking, run once per positive
        # sample, inside a loop over classes -- O(N x P x C) overall, which on a
        # million-row evaluation set stalls an epoch for many minutes.
        #
        # `np.where(ranks_c == idx)[0][0]` is just the position of sample `idx`
        # in the ordering, i.e. the inverse permutation of the argsort. Building
        # that inverse once costs O(N log N) and gives bit-identical results.
        true_indices = np.where(y_true_bin == 1)[0]
        if len(true_indices) > 0:
            order = np.argsort(-y_score_class)
            position = np.empty(len(order), dtype=np.int64)
            position[order] = np.arange(len(order))
            mrr_c = float(np.mean(1.0 / (position[true_indices] + 1)))
        else:
            mrr_c = 0.0
        mrr_by_class[c] = mrr_c

    return (
        avg_precision,          # 0
        auc_score,              # 1
        mrr,                    # 2
        recall_macro,           # 3
        acc,                    # 4
        total_category_loss / num_batches if category_criterion else None,  # 5
        {'edge_loss': total_edge_loss / num_batches if edge_criterion else None},  # 6
        accuracy_by_class,      # 7
        f1_macro,               # 8
        fpr_macro,              # 9
        fnr_macro,              # 10
        tpr_macro,              # 11
        tnr_macro,              # 12
        hits_at_1,              # 13
        hits_at_3,              # 14
        hits_at_5,              # 15
        y_true,                 # 16
        y_scores,               # 17
        precision_by_class,     # 18
        auc_by_class,           # 19
        mrr_by_class,           # 20
        recall_by_class         # 21
        )

