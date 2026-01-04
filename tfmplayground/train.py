import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from typing import Dict
from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree
import os

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device
from evaluation.metrics import roc_auc_error, average_treatment_effect

def update_top_k_checkpoints(
    ckpt_dir: str,
    model_name: str,
    training_state: dict,
    score: float,
    best_scores: list[float],
    k_best: int,
) -> list[float]:
    """Update top-k best checkpoints on disk based on score (lower is better).

    Args:
        ckpt_dir: Directory to save checkpoints
        model_name: Name of the model for file naming
        training_state: Current model state dict to save
        score: Current score (ROC AUC Error, lower is better)
        best_scores: List of current best scores in memory
        k_best: Number of top checkpoints to keep

    Returns:
        Updated best_scores list
    """
    # Find insertion index
    insert_idx = len(best_scores)
    for i, s in enumerate(best_scores):
        if score < s:
            insert_idx = i
            break

    if insert_idx < k_best:
        # Shift existing best files downwards on disk to make room
        current_count = min(len(best_scores), k_best)
        # j is 1-based rank; move best_j -> best_{j+1}
        for j in range(min(current_count, k_best - 1), insert_idx, -1):
            src = os.path.join(ckpt_dir, f"best_{j}_{model_name}.pth")
            dst = os.path.join(ckpt_dir, f"best_{j+1}_{model_name}.pth")
            if os.path.exists(src):
                os.replace(src, dst)
        # Save current model as the new best at position insert_idx+1
        new_best_path = os.path.join(ckpt_dir, f"best_{insert_idx+1}_{model_name}.pth")
        torch.save(training_state, new_best_path)
        # Update scores list in memory
        best_scores.insert(insert_idx, score)
        if len(best_scores) > k_best:
            best_scores = best_scores[:k_best]

    return best_scores

def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    ckpt: Dict[str, torch.Tensor] = None,
    multi_gpu: bool = False,
    run_name: str = 'nanoTFM',
    max_hours: float | None = None,
    model_name: str = 'NanoTabPFN',
    k_best: int = 3,
):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    ckpt_dir = 'pre_training/pre_trained_models/' + run_name
    os.makedirs(ckpt_dir, exist_ok=True)
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'
    time_budget = (max_hours * 3600) if max_hours else None

    best_scores: list[float] = []  # lower is better (ROC AUC Error)

    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_accuracy = 0.0
            total_ate = 0.0
            roc_auc_err_sum = 0.0
            for i, full_data in enumerate(prior):
                scm_dataset = full_data['scm_dataset']
                observational_data = scm_dataset.get_observational_features()
                context_labels = scm_dataset.get_context_labels()
                observational_query_labels = scm_dataset.get_observational_query_set_labels()
                y_j = full_data.get('y_j')
                y_ns = full_data.get('y_ns')
                y_ns_soft = full_data.get('y_ns_soft')
                single_eval_pos = scm_dataset.train_size

                model.train()
                optimizer.train()

                x = observational_data.to(device)
                y_obs = context_labels.unsqueeze(-1).to(device)
                data = (x, y_obs)

                output = model(data, single_eval_pos=single_eval_pos)
                output_shape = output.shape
                target_source = observational_query_labels
                if model_name == 'NanoJPFN' and y_j is not None:
                    target_source = y_j
                elif model_name == 'NanoNsPFN' and y_ns is not None:
                    target_source = y_ns
                elif model_name == 'NanoNssPFN' and y_ns_soft is not None:
                    target_source = y_ns_soft

                if classification_task:
                    # For NanoNssPFN, use soft labels (no .long() conversion)
                    if model_name == 'NanoNssPFN' and y_ns_soft is not None:
                        # target_source is [batch_size, num_samples, 2] soft labels
                        tgt = target_source.to(device)
                        
                        output = output.view(-1, output.shape[-1])
                        tgt = tgt.view(-1, tgt.shape[-1])
                    else:
                        tgt = target_source.reshape((-1,)).to(torch.long).to(device)
                        output = output.view(-1, output.shape[-1])
                else:
                    # Minimal regression support (no extra metrics)
                    tgt = target_source.to(device)

                losses = criterion(output, tgt)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients
                # Metrics (classification only)
                if classification_task:
                    output_detached = output.detach()
                    # Accuracy on CPU
                    accuracies = output_detached.argmax(dim=1).to('cpu') == observational_query_labels.reshape((-1,)).to('cpu')
                    total_accuracy += (accuracies.sum().item() / accuracies.numel())
                    del accuracies

                    with torch.inference_mode():
                        # ROC AUC Error
                        values = output_detached.view(output_shape).argmax(dim=2).to('cpu')
                        unique_classes = [torch.unique(classes) for classes in context_labels]
                        probs = output_detached.view(output_shape).softmax(dim=2).to('cpu')
                        from collections import namedtuple
                        Prediction = namedtuple("Prediction", ["values", "probs"])  # local minimal tuple
                        Results = namedtuple("Results", ["y_true", "preds", "unique_classes"])  # local minimal tuple
                        res = Results(
                            y_true=observational_query_labels.to('cpu'),
                            preds=Prediction(values=values, probs=probs),
                            unique_classes=unique_classes,
                        )
                        roc_auc = roc_auc_error(res, scm_dataset)
                        roc_auc_err_sum += float(roc_auc)
                        del values, probs, res, unique_classes

                        # ATE
                        obs_ctx_cft_q_features = scm_dataset.get_observational_context_features_and_nth_counterfactual_query_set_features(n=1).to(device)
                        data_cf = (obs_ctx_cft_q_features, y_obs)
                        preds_cf = model(data_cf, single_eval_pos=single_eval_pos)
                        values_cf = preds_cf.argmax(dim=2)
                        values_obs = output_detached.view(output_shape).argmax(dim=2)
                        cat_values = torch.cat([values_obs, values_cf], dim=1)
                        res = Results(y_true=None, preds=Prediction(values=cat_values, probs=None), unique_classes=None)
                        ate = average_treatment_effect(res, scm_dataset)
                        total_ate += float(ate)
                        del preds_cf, values_cf, values_obs, cat_values, res

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            mean_accuracy = (total_accuracy / len(prior)) if classification_task else None
            mean_ate = (total_ate / len(prior)) if classification_task else None
            mean_roc_auc_error = (roc_auc_err_sum / len(prior)) if classification_task else None

            model.eval()
            optimizer.eval()

            training_state = {
                'epoch': epoch,
                'architecture': {
                    'num_layers': int((model.module if multi_gpu else model).num_layers),
                    'embedding_size': int((model.module if multi_gpu else model).embedding_size),
                    'num_attention_heads': int((model.module if multi_gpu else model).num_attention_heads),
                    'mlp_hidden_size': int((model.module if multi_gpu else model).mlp_hidden_size),
                    'num_outputs': int((model.module if multi_gpu else model).num_outputs)
                },
                'model': (model.module if multi_gpu else model).state_dict(),
                'optimizer': optimizer.state_dict()
            }
            latest_path = os.path.join(ckpt_dir, f'model_{model_name}_ckpt.pth')
            torch.save(training_state, latest_path)

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        (model.module if multi_gpu else model),
                        dist=criterion,
                        accuracy=mean_accuracy,
                        ate=mean_ate,
                        roc_auc_error=mean_roc_auc_error,
                        model_name=model_name,
                    )
                else:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        (model.module if multi_gpu else model),
                        accuracy=mean_accuracy,
                        ate=mean_ate,
                        roc_auc_error=mean_roc_auc_error,
                        model_name=model_name,
                    )

            if (mean_loss is not None) and k_best > 0:
                best_scores = update_top_k_checkpoints(
                    ckpt_dir, model_name, training_state,
                    float(mean_loss), best_scores, k_best
                )

            if time_budget is not None:
                time_budget -= (time.time() - epoch_start_time)
                if time_budget <= 0:
                    print(f"Reached time budget of {max_hours} hours. Stopping training.")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss
