import torch


def w_reinforce_loss(log_probs, rewards, psr_lambda, token_mask=None):
    """Compute weighted REINFORCE for +1/-1 rewards.

    Args:
        log_probs: Tensor of shape (batch_size, sequence_length), token-level log-probabilities.
        rewards: Tensor of shape (batch_size,), values in {+1, -1}.
        psr_lambda: Scalar weight for positive-sample reinforcement.
        token_mask: Optional tensor of shape (batch_size, sequence_length) to mask valid tokens.
    """
    if token_mask is not None:
        token_mask = token_mask.to(log_probs.dtype)
        denom = token_mask.sum(dim=-1).clamp_min(1.0)
        seq_log_probs = (log_probs * token_mask).sum(dim=-1) / denom
    else:
        seq_log_probs = log_probs.sum(dim=-1)

    pos_mask = rewards == 1.0
    neg_mask = rewards == -1.0

    psr_loss = -(seq_log_probs[pos_mask]).mean() if pos_mask.any() else torch.tensor(0.0, device=log_probs.device)
    nsr_loss = (seq_log_probs[neg_mask]).mean() if neg_mask.any() else torch.tensor(0.0, device=log_probs.device)
    return (psr_lambda * psr_loss) + nsr_loss


class StepLambdaScheduler:
    def __init__(self, initial_lambda=0.01, step_size=100, lambda_multiplier=10.0, max_lambda=1.0):
        """Increase lambda every step_size iterations by lambda_multiplier."""
        self.current_lambda = initial_lambda
        self.step_size = step_size
        self.lambda_multiplier = lambda_multiplier
        self.max_lambda = max_lambda
        self.iteration = 0

    def step(self):
        self.iteration += 1
        if self.step_size > 0 and self.iteration % self.step_size == 0:
            self.current_lambda = min(self.current_lambda * self.lambda_multiplier, self.max_lambda)
        return self.current_lambda

    def get_lambda(self):
        return self.current_lambda


class RatioLambdaScheduler:
    def __init__(self, initial_lambda=0.01, target_positive_ratio=0.2, update_rate=0.1, max_lambda=1.0, min_lambda=0.0):
        """Adapt lambda using batch positive ratio to stabilize PSR/NSR balance."""
        self.current_lambda = initial_lambda
        self.target_positive_ratio = target_positive_ratio
        self.update_rate = update_rate
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda

    def step(self, positive_ratio):
        error = self.target_positive_ratio - float(positive_ratio)
        self.current_lambda = self.current_lambda + self.update_rate * error
        self.current_lambda = min(max(self.current_lambda, self.min_lambda), self.max_lambda)
        return self.current_lambda

    def get_lambda(self):
        return self.current_lambda