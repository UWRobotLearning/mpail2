import torch
from typing import Mapping, Dict

#
# Logging stats
#

def task_stats(stats: Dict[str, list], task_stat_name) -> Mapping[str, float]:
    """
    Return success or failure using infos collected during rollouts.
    """

    if task_stat_name in ['push_success']:
        # Check for any reward of 1.0 in the episodes stats
        success = any(s == 1.0 for s in stats['Episode_Reward/push_sparse_reward'])
        return {f"Result/{task_stat_name}": success}

    if task_stat_name in ['pick_success']:
        # Check for any reward of 1.0 in the episodes stats
        success = any(s == 1.0 for s in stats['Episode_Reward/pick_and_place_sparse_reward'])
        return {f"Result/{task_stat_name}": success}


#
# Train stats
#

def compute_disc_train_stats(
    disc_output: torch.Tensor,
    labels: torch.Tensor,
    disc_loss: torch.Tensor,
    from_probs: bool = False,
) -> Mapping[str, float]:
    """
    Taken from: https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/adversarial/common.py#L27

    Train statistics for GAIL/AIRL discriminator.

    Args:
        disc_logits: discriminator logits produced by
            `AdversarialTrainer.logits_expert_is_high`.
        labels: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        A mapping from statistic names to float values.
    """
    with torch.no_grad():
        disc_logits = torch.logit(disc_output) if from_probs else disc_output
        # Logits of the discriminator output; >0 for expert samples, <0 for generator.
        bin_is_generated_pred = disc_logits < 0
        # Binary label, so 0 is for generator, 1 is for expert.
        bin_is_generated_true = labels == 0
        bin_is_expert_true = torch.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(torch.sum(int_is_generated_true))
        n_labels = float(len(labels))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - torch.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = torch.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = torch.mean(correct_vec.float())

        _n_pred_expert = torch.sum(torch.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Pytorchon ints
            expert_acc = _n_pred_expert.item() / float(n_expert)

        _n_pred_gen = torch.sum(torch.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = torch.distributions.Bernoulli(logits=disc_logits)
        entropy = torch.mean(label_dist.entropy())

    return {
        "disc_loss": float(torch.mean(disc_loss)),
        "disc_acc": float(acc),
        "disc_acc_expert": float(expert_acc),  # accuracy on just expert examples
        "disc_acc_gen": float(generated_acc),  # accuracy on just generated examples
        # entropy of torche predicted label distribution, averaged equally across
        # botorch classes (if torchis drops torchen disc is very good or has given up)
        "disc_entropy": float(entropy),
        # true number of expert demos and predicted number of expert demos
        "disc_proportion_expert_true": float(pct_expert),
        "disc_proportion_expert_pred": float(pct_expert_pred),
        "n_expert": float(n_expert),
        "n_generated": float(n_generated),
    }

class Stats:
    def __init__(self):
        self.stats = {}

    def update(self, stats: dict):
        for key, value in stats.items():
            if key not in self.stats:
                self.stats[key] = []
            self.stats[key].append(value)

    def mean(self):
        return {f"{key}": sum(values) / len(values) for key, values in self.stats.items()}