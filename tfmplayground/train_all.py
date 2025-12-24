import os
import time
import torch
import schedulefree
from collections import namedtuple
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List
from pfns.bar_distribution import FullSupportBarDistribution

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device
from experiments.mc.metrics import roc_auc_error, average_treatment_effect

Prediction = namedtuple("Prediction", ["values", "probs"])
Results = namedtuple("Results", ["y_true", "preds", "unique_classes"])

def train(models: List[NanoTabPFNModel], prior: DataLoader, criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
          epochs: int, max_hours: float, model_idx_to_name: dict, accumulate_gradients: int = 1, lr: float = 1e-4, device: torch.device = None,
          callbacks: list[Callback] = None, ckpt: Dict[str, torch.Tensor] = None, multi_gpu: bool = False,
          run_name: str = 'nanoTFM'):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        models: (List[NanoTabPFNModel]) list of models to train on the same prior
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
    ckpt_dir = 'pre_trained_models/'+run_name
    os.makedirs(ckpt_dir, exist_ok=True)
    if multi_gpu:
        models = [nn.DataParallel(m) for m in models]
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    # Keep models on CPU by default; move to device one-at-a-time when training
    optimizers = [schedulefree.AdamWScheduleFree(m.parameters(), lr=lr, weight_decay=0.0) for m in models]

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'
    time_budget = max_hours * 3600
    print(f"Starting training for up to {max_hours} hours ({time_budget} seconds)...")
    min_roc_auc_errors = [float('inf') for _ in models]
    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            # Train each model independently; set train mode when moved to device
            total_losses = [0.0 for _ in models]
            total_accuracies = [0.0 for _ in models]
            total_ates = [0.0 for _ in models]
            roc_auc_errors = [0.0 for _ in models]
            for i, full_data in enumerate(prior):
                experiment_dataset = full_data['experiment_dataset']
                observational_data = experiment_dataset.get_observational_features()
                context_labels = experiment_dataset.get_context_labels()
                observational_query_labels = experiment_dataset.get_observational_query_set_labels()

                y_j = full_data['y_j']
                y_ns = full_data['y_ns']
                single_eval_pos = experiment_dataset.train_size
                # Keep batch tensors on CPU; move per-model to device inside inner loop
                data_cpu = (
                    observational_data,
                    context_labels.unsqueeze(-1)
                )

                targets_list = [
                    observational_query_labels,
                    # y_j,
                    # y_ns
                ]

                for idx, (m, opt) in enumerate(zip(models, optimizers)):
                    m.to(device)
                    m.train()
                    opt.train()
                    x = data_cpu[0].to(device)
                    y_obs = data_cpu[1].to(device)
                    # if (torch.isnan(x).any() or torch.isnan(y_obs).any()):
                    #     # Offload and skip this model for this batch
                    #     m.eval(); m.to('cpu'); torch.cuda.empty_cache()
                    #     continue
                    # else:
                    data = (x, y_obs)

                    output = m(data, single_eval_pos=single_eval_pos)
                    output_shape = output.shape 
                    tgt = targets_list[idx].to(device)
                    tgt = tgt.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                    # print(f"Shapes - output: {output.shape}, tgt: {tgt.shape}, idx: {idx}")
                    losses = criterion(output, tgt)
                    loss = losses.mean() / accumulate_gradients
                    loss.backward()
                    total_losses[idx] += loss.cpu().detach().item() * accumulate_gradients

                    # # Detach logits to free autograd graph before metric computations
                    output_detached = output.detach()
                    # Compute accuracy on CPU to reduce GPU memory pressure
                    accuracies = output_detached.argmax(dim=1).to('cpu') == targets_list[0].reshape((-1,)).to('cpu')
                    total_accuracies[idx] += (accuracies.sum().item() / accuracies.numel())
                    del accuracies

                    # m.eval()
                    # # Use inference_mode (less overhead than no_grad and disables version counters)
                    with torch.inference_mode():
                        # Compute ROC AUC Error
                        # Move minimal data needed for metrics to CPU
                        values = output_detached.view(output_shape).argmax(dim=2).to('cpu')
                        unique_classes = [torch.unique(classes) for classes in context_labels]
                        probs = output_detached.view(output_shape).softmax(dim=2).to('cpu')
                        res = Results(
                            y_true=targets_list[0].to('cpu'),
                            preds=Prediction(
                                values=values,
                                probs=probs
                            ),
                            unique_classes=unique_classes
                        )

                        roc_auc = roc_auc_error(res, experiment_dataset)
                        roc_auc_errors[idx] += float(roc_auc)
                        # Free CPU tensors used for ROC AUC
                        del values, probs, res, unique_classes

                        # Compute ATE
                        obs_ctx_cft_q_features = experiment_dataset.get_observational_context_features_and_nth_counterfactual_query_set_features(n=1).to(device)
                        data_cf = (obs_ctx_cft_q_features, y_obs)
                        preds_cf = m(data_cf, single_eval_pos=single_eval_pos)
                        values_cf = preds_cf.argmax(dim=2)
                        values_obs = output_detached.view(output_shape).argmax(dim=2)
                        cat_values = torch.cat([values_obs, values_cf], dim=1)
                        res = Results(
                            y_true=None,
                            preds=Prediction(values=cat_values, probs=None),
                            unique_classes=None
                        )
                        ate = average_treatment_effect(res, experiment_dataset)
                        total_ates[idx] += float(ate)
                        # Free GPU tensors used for ATE
                        del preds_cf, values_cf, values_obs, cat_values, res

                    if (i + 1) % accumulate_gradients == 0:
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                        opt.step()
                        opt.zero_grad()

                    # Explicitly free per-batch tensors and move model off GPU
                    # del output_detached, output, tgt, x, y_obs, data
                    m.to('cpu')
                    # Optionally clear cached allocator blocks (helps when memory is fragmented)
                    torch.cuda.empty_cache()

            end_time = time.time()
            mean_losses = [tl / len(prior) for tl in total_losses]
            mean_accuracies = [ta / len(prior) for ta in total_accuracies]
            mean_ates = [ta / len(prior) for ta in total_ates]
            mean_roc_auc_errors = [rae / len(prior) for rae in roc_auc_errors]

            for idx, (m, opt) in enumerate(zip(models, optimizers)):
                # Save checkpoint if this is the lowest roc_auc_error so far
                # if mean_roc_auc_errors[idx] < min_roc_auc_errors[idx]:
                #     min_roc_auc_errors[idx] = mean_roc_auc_errors[idx]
                training_state = {
                    'epoch': epoch,
                    'architecture': {
                        'num_layers': int((m.module if multi_gpu else m).num_layers),
                        'embedding_size': int((m.module if multi_gpu else m).embedding_size),
                        'num_attention_heads': int((m.module if multi_gpu else m).num_attention_heads),
                        'mlp_hidden_size': int((m.module if multi_gpu else m).mlp_hidden_size),
                        'num_outputs': int((m.module if multi_gpu else m).num_outputs)
                    },
                    'model': (m.module if multi_gpu else m).state_dict(),
                    'optimizer': opt.state_dict()
                }
                torch.save(training_state, os.path.join(ckpt_dir, f'model_{model_idx_to_name[idx]}_ckpt.pth'))

                # Callbacks per model (kept minimal): report using this model
                for callback in callbacks:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_losses[idx],
                        (m.module if multi_gpu else m),
                        accuracy=mean_accuracies[idx],
                        ate=mean_ates[idx],
                        roc_auc_error=mean_roc_auc_errors[idx],
                        model_name=model_idx_to_name[idx],
                    )

            time_budget -= (time.time() - epoch_start_time)
            if time_budget <= 0:
                print(f"Reached time budget of {max_hours} hours. Stopping training.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return [(m.module if multi_gpu else m) for m in models], total_losses
