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
    category_criterion=None
):
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
        _, negatives_batch = negative_edge_sampler.sample(size)

        pos_score, neg_score, category_logits = model.compute_edge_probabilities_and_categories(
            sources_batch,
            destinations_batch,
            negatives_batch,
            timestamps_batch,
            edge_idxs_batch,
            n_neighbors=n_neighbors
        )

        pred_categories = torch.argmax(category_logits, dim=1).cpu().numpy()
        all_true_labels.extend(categories_batch)
        all_pred_labels.extend(pred_categories)
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
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    category_logits = np.concatenate(all_category_logits)
    pos_scores = np.concatenate(all_pos_scores)
    neg_scores = np.concatenate(all_neg_scores)

    y_scores = torch.softmax(torch.tensor(category_logits), dim=1).numpy()

    # Binary classification (link prediction)
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    auc_score = roc_auc_score(all_labels, all_scores)
    avg_precision = average_precision_score(all_labels, all_scores)

    # Hits@K
    hits_at_k = lambda k: np.mean([
        y_true[i] in np.argsort(-y_scores[i])[:k] for i in range(len(y_true))
    ])
    hits_at_1 = hits_at_k(1)
    hits_at_3 = hits_at_k(3)
    hits_at_5 = hits_at_k(5)

    # MRR
    ranks = np.argsort(-y_scores, axis=1)
    correct_ranks = np.array([np.where(ranks[i] == y_true[i])[0][0] + 1 for i in range(len(y_true))])
    mrr = np.mean(1 / correct_ranks)

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
        true_indices = (y_true == c)
        if true_indices.sum() == 0:
            accuracy_by_class[c] = 0.0
        else:
            correct_preds = (y_pred[true_indices] == c).sum()
            accuracy_by_class[c] = correct_preds / true_indices.sum()

    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        y_score_class = y_scores[:, c]

        precision_by_class[c] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall_by_class[c] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_by_class[c] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        try:
            auc_by_class[c] = roc_auc_score(y_true_bin, y_score_class)
        except:
            auc_by_class[c] = float('nan')

        ranks_c = np.argsort(-y_score_class)
        true_indices = np.where(y_true_bin == 1)[0]
        if len(true_indices) > 0:
            mrr_c = np.mean([
                1 / (np.where(ranks_c == idx)[0][0] + 1) for idx in true_indices
            ])
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

