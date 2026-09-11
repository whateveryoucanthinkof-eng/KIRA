def find_best_threshold_per_class(y_true, y_scores, class_labels, metric='f1'):
    thresholds_by_class = {}

    for cls in np.unique(class_labels):
        mask = class_labels == cls
        true = y_true[mask]
        scores = y_scores[mask]

        best_thresh, best_score = 0.5, 0.0
        if len(np.unique(true)) < 2:
            thresholds_by_class[cls] = 0.5
            continue

        for t in np.linspace(0.01, 0.99, 99):
            pred = (scores >= t).astype(int)
            score = f1_score(true, pred, zero_division=0)
            if score > best_score:
                best_thresh = t
                best_score = score
        thresholds_by_class[cls] = best_thresh

    return thresholds_by_class


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.alpha is not None:
            at = torch.ones_like(targets, dtype=torch.float).to(inputs.device) * (1 - self.alpha)
            at[targets == 1] = self.alpha
            loss *= at

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

import math
import logging
import time
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from google.colab import drive
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)

# Dictionary to replace argparse
args = {
    'data': 'new_df',
    'bs': 128,
    'prefix': '',
    'n_degree': 10,
    'n_head': 2,
    'n_epoch': 50,
    'n_layer': 1,
    'lr': 0.0001,
    'patience': 5,
    'n_runs': 1,
    'drop_out': 0.1,
    'gpu': 0,
    'node_dim': 100,
    'time_dim': 100,
    'backprop_every': 1,
    'use_memory': False,
    'embedding_module': 'graph_attention',
    'message_function': 'identity',
    'memory_updater': 'gru',
    'aggregator': 'bigru_transformer',
    'memory_update_at_end': False,
    'message_dim': 100,
    'memory_dim': 9,
    'different_new_nodes': True,
    'uniform': False,
    'randomize_features': False,
    'use_destination_embedding_in_message': False,
    'use_source_embedding_in_message': False,
    'dyrep': False,
    'num_categories' : 4
}

BATCH_SIZE = args['bs']
NUM_NEIGHBORS = args['n_degree']
NUM_NEG = 1
num_categories = 4
NUM_EPOCH = args['n_epoch']
NUM_HEADS = args['n_head']
DROP_OUT = args['drop_out']
GPU = args['gpu']
DATA = args['data']
NUM_LAYER = args['n_layer']
LEARNING_RATE = args['lr']
NODE_DIM = args['node_dim']
TIME_DIM = args['time_dim']
USE_MEMORY = args['use_memory']
MESSAGE_DIM = args['message_dim']
MEMORY_DIM = args['memory_dim']

# Initialize dictionaries to track category loss and accuracy by type
category_loss_by_type = {i: [] for i in range(num_categories)}
category_accuracy_by_type = {i: [] for i in range(num_categories)}

# Define paths in Google Drive
saved_models_path = '/content/drive/My Drive/saved_models/'
saved_checkpoints_path = '/content/drive/My Drive/saved_checkpoints/'
log_path = '/content/drive/My Drive/log/'

# Create directories if they don't exist
Path(saved_models_path).mkdir(parents=True, exist_ok=True)
Path(saved_checkpoints_path).mkdir(parents=True, exist_ok=True)
Path(log_path).mkdir(parents=True, exist_ok=True)

# Set up paths for saving models and checkpoints
MODEL_SAVE_PATH = f'{saved_models_path}{args["prefix"]}-{args["data"]}.pth'
get_checkpoint_path = lambda epoch: f'{saved_checkpoints_path}{args["prefix"]}-{args["data"]}-{epoch}.pth'

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{log_path}{str(time.time())}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(new_df, edge_features, node_features,
                              different_new_nodes_between_val_and_test=args['different_new_nodes'],
                              randomize_features=args['randomize_features'])
#DATA,different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features
# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args['uniform'])

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args['uniform'])

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args['n_runs']):
    results_path = "results/{}_{}.pkl".format(args['prefix'], i) if i > 0 else "results/{}.pkl".format(args['prefix'])
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = ExtendedTGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args['memory_update_at_end'],
              embedding_module_type=args['embedding_module'],
              message_function=args['message_function'],
              aggregator_type=args['aggregator'],
              memory_updater_type=args['memory_updater'],
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args['use_destination_embedding_in_message'],
              use_source_embedding_in_message=args['use_source_embedding_in_message'],
              dyrep=args['dyrep'], num_categories=num_categories)

    criterion = torch.nn.BCELoss()
    category_criterion = FocalLoss(alpha=0.25, gamma=2.0) #torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    new_nodes_val_accuracy = []
    new_nodes_val_mrr = []
    val_aps = []
    val_accuracy = []
    val_mrrs = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args['patience'])
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for k in range(0, num_batch, args['backprop_every']):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args['backprop_every']):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                categories_batch = train_data.labels[start_idx:end_idx]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                #    def compute_edge_probabilities_and_categories(
                #self, source_nodes, destination_nodes, negative_nodes,
                #pos_edge_times, neg_edge_times, edge_idxs, n_neighbors=20):
                pos_prob, neg_prob, category_logits = tgn.compute_edge_probabilities_and_categories(
                    sources_batch,
                    destinations_batch,
                    negatives_batch,
                    timestamps_batch,
                    edge_idxs_batch,              #
                    n_neighbors=20                #
                )
                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                category_loss = category_criterion(category_logits, torch.tensor(categories_batch, dtype=torch.long, device=device))

            #loss /= args['backprop_every']
            loss = (loss + category_loss) / args['backprop_every']

            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Track category loss by type
            categories_batch_tensor = torch.tensor(categories_batch, dtype=torch.long, device=device)
            for category in range(num_categories):
              mask = (categories_batch == category)
              if mask.any():
                category_loss_by_type[category].append(category_criterion(
                    category_logits[mask], categories_batch_tensor[mask]).item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        (val_ap,
        val_auc,
        val_mrr,
        val_recall,
        val_category_accuracy,
        val_category_loss,
        val_category_loss_by_type,
        val_category_accuracy_by_type,
        val_f1,
        val_fpr,
        val_fnr,
        val_tpr,
        val_tnr,
        val_hits_at_1,
        val_hits_at_3,
        val_hits_at_5,
        val_all_labels,
        val_all_scores,
        val_category_precision_by_type,
        val_category_auc_by_type,
        val_category_mrr_by_type,
        val_category_recall_by_type)= eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=val_rand_sampler, data=val_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion)

        #criterion = torch.nn.BCELoss()
        #category_criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Update the tracking dictionaries for validation
        for category in range(num_categories):
            # Check if the key exists in val_category_loss_by_type and val_category_accuracy_by_type
            if category in val_category_loss_by_type and category in val_category_accuracy_by_type:
                category_loss_by_type[category].append(val_category_loss_by_type[category])
                category_accuracy_by_type[category].append(val_category_accuracy_by_type[category])
            else:
                # Optionally add NaN or skip
                category_loss_by_type[category].append(np.nan)
                category_accuracy_by_type[category].append(np.nan)


        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        (nn_val_ap,
        nn_val_auc,
        nn_val_mrr,
        nn_val_recall,
        nn_val_category_accuracy,
        nn_val_category_loss,
        nn_val_category_loss_by_type,
        nn_val_category_accuracy_by_type,
        nn_val_f1,
        nn_val_fpr,
        nn_val_fnr,
        nn_val_tpr,
        nn_val_tnr,
        nn_val_hits_at_1,
        nn_val_hits_at_3,
        nn_val_hits_at_5,
        nn_val_all_labels,
        nn_val_all_scores,
        nn_val_category_precision_by_type,
        nn_val_category_auc_by_type,
        nn_val_category_mrr_by_type,
        nn_val_category_recall_by_type) = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=val_rand_sampler, data=new_node_val_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        new_nodes_val_accuracy.append(nn_val_category_accuracy)
        new_nodes_val_mrr.append(nn_val_mrr)
        val_aps.append(val_ap)
        val_accuracy.append(val_category_accuracy)
        val_mrrs.append(val_mrr)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "new_nodes_val_mrr":new_nodes_val_mrr,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        df_r = pd.DataFrame({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "new_nodes_val_mrr":new_nodes_val_mrr,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
        })

        df_r.to_csv('/content/training_metrics_edgcl.csv', index=True)

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
            'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
        logger.info(
            'val mrr: {}, new node val mrr: {}'.format(val_mrr, nn_val_mrr))
        logger.info(
            'val category accuracy: {}, new node val category accuracy: {}'.format(val_category_accuracy, nn_val_category_accuracy))
        logger.info(
            'val category loss: {}, new node val category loss: {}'.format(val_category_loss, nn_val_category_loss))

        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test - np.mean(val_ap), np.mean(val_auc), category_accuracy, predicted_positive_edges
    tgn.embedding_module.neighbor_finder = full_ngh_finder

    (test_ap,
     test_auc,
     test_mrr,
     test_recall,
     category_accuracy_test,
     test_category_loss,
     test_category_loss_by_type,
     test_category_accuracy_by_type,
     test_f1,
     test_fpr,
     test_fnr,
     test_tpr,
     test_tnr,
     test_hits_at_1,
     test_hits_at_3,
     test_hits_at_5,
     test_all_labels,
     test_all_scores,
     test_category_precision_by_type,
     test_category_auc_by_type,
     test_category_mrr_by_type,
     test_category_recall_by_type)  = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=test_rand_sampler, data=test_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion)

    for category in range(num_categories):
      if category in test_category_loss_by_type and category in test_category_accuracy_by_type:
        category_loss_by_type[category].append(test_category_loss_by_type[category])
        category_accuracy_by_type[category].append(test_category_accuracy_by_type[category])
      else:
        category_loss_by_type[category].append(np.nan)
        category_accuracy_by_type[category].append(np.nan)




    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # Test on unseen nodes # np.mean(val_ap), np.mean(val_auc), category_accuracy, predicted_positive_edges, category_loss
    (nn_test_ap,
     nn_test_auc,
     nn_test_mrr,
     nn_test_recall,
     category_accuracy_nn_test,
     category_loss_nn_test,
     category_loss_by_type_nn_test,
     category_accuracy_by_type_nn_test,
     nn_test_f1,
     nn_test_fpr,
     nn_test_fnr,
     nn_test_tpr,
     nn_test_tnr,
     nn_test_hits_at_1,
     nn_test_hits_at_3,
     nn_test_hits_at_5,
     nn_test_all_labels,
     nn_test_all_scores,
     nn_test_category_precision_by_type,
     nn_test_category_auc_by_type,
     nn_test_category_mrr_by_type,
     nn_test_category_recall_by_type) = eval_edge_prediction_with_categories(model=tgn,
                                                   negative_edge_sampler=nn_test_rand_sampler,
                                                   data=new_node_test_data,
                                                   n_neighbors=NUM_NEIGHBORS,
                                                   edge_criterion=criterion,
                                                   category_criterion= category_criterion)

    logger.info(
        'Test statistics: Old nodes -- auc: {}, ap: {}, mrr: {}, recall : {}'.format(test_auc, test_ap, test_mrr, test_recall))
    logger.info(
        'Test statistics: New nodes -- auc: {}, ap: {}, mrr: {}, recall: {}'.format(nn_test_auc, nn_test_ap, nn_test_mrr, nn_test_recall))
    logger.info('Test statistics: Old nodes -- accuracy: {}, loss: {}'.format(category_accuracy_test, test_category_loss))
    logger.info('Test statistics: New nodes -- accuracy: {}, loss: {}'.format(category_accuracy_nn_test, category_loss_nn_test))
    logger.info('Test statistics: Old nodes -- accuracy_types: {}, loss_types: {}'.format(test_category_accuracy_by_type,test_category_loss_by_type))
    logger.info('Test statistics: New nodes -- accuracy_types: {}, loss_types: {}'.format(category_accuracy_by_type_nn_test, category_loss_by_type_nn_test))
    logger.info('Test statistics: Old nodes -- f1: {}, fpr: {}, fnr: {}, tpr: {}, tnr: {}'.format(test_f1, test_fpr, test_fnr, test_tpr, test_tnr))
    logger.info('Test statistics: New nodes -- f1: {}, fpr: {}, fnr: {}, tpr: {}, tnr: {}'.format(nn_test_f1, nn_test_fpr, nn_test_fnr, nn_test_tpr, nn_test_tnr))
    logger.info('Test statistics: Old nodes -- hits@1: {}, hits@3: {}, hits@5: {}'.format(test_hits_at_1, test_hits_at_3, test_hits_at_5))
    logger.info('Test statistics: New nodes -- hits@1: {}, hits@3: {}, hits@5: {}'.format(nn_test_hits_at_1, nn_test_hits_at_3, nn_test_hits_at_5))
    logger.info('Test statistics: Old nodes -- precision_types: {}, auc_types: {}, mrr_types: {}, recall_types: {}'.format(test_category_precision_by_type, test_category_auc_by_type, test_category_mrr_by_type, test_category_recall_by_type))
    logger.info('Test statistics: New nodes -- precision_types: {}, auc_types: {}, mrr_types: {}, recall_types: {}'.format(nn_test_category_precision_by_type, nn_test_category_auc_by_type, nn_test_category_mrr_by_type, nn_test_category_recall_by_type))
    #logger.info('Test statistics: Old nodes -- all labels: {}, all scores: {}'.format(test_all_labels, test_all_scores))
    #logger.info('Test statistics: New nodes -- all labels: {}, all scores: {}'.format(nn_test_all_labels, nn_test_all_scores))

    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "test_mrr": test_mrr,
        "new_node_test_ap": nn_test_ap,
        "new_node_test_mrr":nn_test_mrr,
        "category_loss_new_node_test": category_loss_nn_test,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    pickle.dump({
    "category_loss_by_type": category_loss_by_type,
    "category_accuracy_by_type": category_accuracy_by_type,
    }, open('results/category_metrics.pkl', 'wb'))

    #df_category_metrics = pd.DataFrame({
    #f"category_{i}_loss": category_loss_by_type[i] for i in range(num_categories)})

    #df_category_metrics.to_csv('/content/category_metrics.csv', index=True)

    df_test = pd.DataFrame({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "test_mrr": test_mrr,
        "new_node_test_ap": nn_test_ap,
        "new_node_test_mrr":nn_test_mrr,
        "category_loss_new_node_test": category_loss_nn_test,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times})

    df_test.to_csv('/content/test_metrics_edgcl.csv', index=True)

    logger.info('Saving TGN model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.memory.restore_memory(val_memory_backup)
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')

import matplotlib.pyplot as plt

# Define the fontsize for the numbers
number_fontsize = 25

# Plotting the results
plt.figure(figsize=(15, 10))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Loss', fontsize=number_fontsize)
plt.title('Training Loss over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation AP
plt.subplot(2, 2, 2)
plt.plot(val_aps, label='Validation AP')
plt.plot(new_nodes_val_aps, label='New Nodes Validation AP')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Average Precision', fontsize=number_fontsize)
plt.title('Validation AP over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation Accuracy
plt.subplot(2, 2, 3)
plt.plot(val_accuracy, label='Validation Accuracy')
plt.plot(new_nodes_val_accuracy, label='New Nodes Validation Accuracy')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Average Accuracy', fontsize=number_fontsize)
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation MRR
plt.subplot(2, 2, 4)
plt.plot(val_mrrs, label='Validation MRR')
plt.plot(new_nodes_val_mrr, label='New Nodes Validation MRR')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('MRR', fontsize=number_fontsize)
plt.title('Validation MRR over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

plt.tight_layout()
plt.show()