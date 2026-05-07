import os
import sys
import torch
import torch.nn.functional as F
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import extract_answer_math
from grader import math_equal

# A100/RTX 4000+ Optimization: Enable TF32 for much faster matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------------------------------------------------------------------
# Math functions
# -------------------------------------------------------------------------------------
def w_reinforce_loss(log_probs, rewards, psr_lambda, nsr_beta=1.0, token_mask=None):
    """Compute weighted REINFORCE for +1/-1 rewards."""
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
    return (psr_lambda * psr_loss) + (nsr_beta * nsr_loss)

class StepLambdaScheduler:
    def __init__(self, initial_lambda=0.01, step_size=100, lambda_multiplier=10.0, max_lambda=1.0):
        self.current_lambda = initial_lambda
        self.step_size = step_size
        self.lambda_multiplier = lambda_multiplier
        self.max_lambda = max_lambda
        self.iteration = 0

    def step(self, pos_ratio=None):
        self.iteration += 1
        if self.step_size > 0 and self.iteration % self.step_size == 0:
            self.current_lambda = min(self.current_lambda * self.lambda_multiplier, self.max_lambda)
        return self.current_lambda, 1.0

    def get_lambda(self):
        return self.current_lambda, 1.0

class RatioLambdaScheduler:
    def __init__(self, initial_lambda=0.01, target_positive_ratio=0.2, update_rate=0.1, max_lambda=1.0, min_lambda=0.0):
        self.current_lambda = initial_lambda
        self.target_positive_ratio = target_positive_ratio
        self.update_rate = update_rate
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda

    def step(self, positive_ratio):
        error = self.target_positive_ratio - float(positive_ratio)
        self.current_lambda = self.current_lambda + self.update_rate * error
        self.current_lambda = min(max(self.current_lambda, self.min_lambda), self.max_lambda)
        return self.current_lambda, 1.0

    def get_lambda(self):
        return self.current_lambda, 1.0

class TheoryLambdaScheduler:
    def __init__(self, total_steps, initial_lambda=0.0, final_lambda=1.0, initial_beta=1.0, final_beta=0.1):
        self.total_steps = total_steps
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.step_idx = 0
        self.current_lambda = initial_lambda
        self.current_beta = initial_beta

    def step(self, *args):
        self.step_idx += 1
        progress = min(self.step_idx / self.total_steps, 1.0)
        
        # Linear increase for PSR (lambda)
        self.current_lambda = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        
        # Exponential decay for NSR (beta)
        if self.initial_beta > 0 and self.final_beta > 0:
            decay_factor = (self.final_beta / self.initial_beta) ** progress
            self.current_beta = self.initial_beta * decay_factor
        else:
            self.current_beta = self.final_beta
            
        return self.current_lambda, self.current_beta

    def get_lambda(self):
        return self.current_lambda, self.current_beta

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, *args, lambda_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_scheduler = lambda_scheduler or StepLambdaScheduler()

    def loss(self, old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns):
        # advantages has shape (batch_size, seq_len). Each token has the seq_reward.
        seq_rewards = advantages[:, 0] # Extract the sequence reward
        
        # Calculate the positive ratio
        pos_ratio = (seq_rewards == 1.0).float().mean()
        
        # Step the scheduler
        current_lambda, current_beta = self.lambda_scheduler.step(pos_ratio)
            
        loss_p = w_reinforce_loss(logprobs, seq_rewards, current_lambda, current_beta, mask)
        # Dummy value loss since AutoModelForCausalLMWithValueHead requires it for backward
        loss_v = torch.tensor(0.0, device=loss_p.device, requires_grad=True)
        
        stats = {
            "loss/policy": loss_p.detach(), 
            "policy/lambda": torch.tensor(current_lambda, device=loss_p.device),
            "policy/beta": torch.tensor(current_beta, device=loss_p.device),
            "policy/pos_ratio": pos_ratio.detach(),
            "policy/returns_mean": seq_rewards.mean().detach(),
            "loss/value": loss_v.detach(),
            "loss/total": loss_p.detach(),
            "loss/entropy": torch.tensor(0.0, device=loss_p.device),
            "policy/entropy": torch.tensor(0.0, device=loss_p.device),
            "policy/advantages": advantages.detach(),
            "policy/advantages_mean": advantages.mean().detach(),
            "policy/returns": returns.detach(),
            "policy/approxkl": torch.tensor(0.0, device=loss_p.device),
            "policy/policykl": torch.tensor(0.0, device=loss_p.device),
            "policy/clipfrac": torch.tensor(0.0, device=loss_p.device),
            "policy/ratio": torch.tensor(1.0, device=loss_p.device),
            "val/error": torch.tensor(0.0, device=loss_p.device),
            "val/vpred": torch.tensor(0.0, device=loss_p.device),
            "val/var": torch.tensor(1.0, device=loss_p.device),
            "val/clipfrac": torch.tensor(0.0, device=loss_p.device),
            "returns/var": torch.tensor(1.0, device=loss_p.device)
        }
        return loss_p, loss_v, stats

    def compute_advantages(self, values: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor):
        # Extract the sequence level reward from self.current_token_level_scores
        seq_rewards = self.current_token_level_scores.sum(dim=-1)
        
        # Return advantages matching token shape so TRL passes them cleanly to loss()
        advantages = seq_rewards.unsqueeze(-1).expand_as(rewards)
        return values, advantages, advantages

# -------------------------------------------------------------------------------------
# Data Formatting & Main Loop
# -------------------------------------------------------------------------------------

def format_reward(query, response_text, ground_truth):
    answer = extract_answer_math(response_text)
    is_correct = math_equal(answer, ground_truth, timeout=True)
    return 1.0 if is_correct else -1.0 # Dynamic lambda requires +1 / -1

def build_dataset(tokenizer, data_path):
    ds = load_dataset("parquet", data_files={"train": data_path})["train"]
    
    def tokenize(sample):
        if isinstance(sample.get('prompt', ''), list):
            prompt_text = tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = (
                "<|im_start|>system\n"
                "You are a helpful assistant.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{sample.get('prompt', '')}\n"
                "Please reason step by step and put your final answer within \\boxed{}.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
        sample["input_ids"] = tokenizer.encode(prompt_text, add_special_tokens=False)
        sample["query"] = prompt_text
        
        if "reward_model" in sample and "ground_truth" in sample["reward_model"]:
            gt_data = sample["reward_model"]["ground_truth"]
            if isinstance(gt_data, dict) and "target" in gt_data:
                sample["ground_truth"] = str(gt_data["target"])
            else:
                sample["ground_truth"] = str(gt_data)
        else:
            sample["ground_truth"] = ""
            
        return sample
        
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids"], output_all_columns=True)
    return ds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_files", type=str, default="./data/math/train.parquet")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts per batch")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="PPO mini batch size")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--max_response_length", type=int, default=1024)
    
    # Lambda scheduler args
    parser.add_argument("--lambda_schedule", type=str, default="step", choices=["step", "ratio", "theory"])
    parser.add_argument("--initial_lambda", type=float, default=0.01)
    parser.add_argument("--lambda_step_size", type=int, default=100)
    parser.add_argument("--lambda_multiplier", type=float, default=10.0)
    parser.add_argument("--max_lambda", type=float, default=1.0)
    parser.add_argument("--min_lambda", type=float, default=0.0)
    parser.add_argument("--target_positive_ratio", type=float, default=0.2)
    parser.add_argument("--ratio_update_rate", type=float, default=0.1)
    
    # Theory scheduler specific args
    parser.add_argument("--initial_beta", type=float, default=1.0)
    parser.add_argument("--final_beta", type=float, default=0.1)
    parser.add_argument("--final_lambda", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, 
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        ppo_epochs=args.epochs,
        remove_unused_columns=False,
    )
    # We set gamma to 1.0, and cliprange arbitrarily since we don't use PPO loss
    config.vf_coef = 0.0
    config.ent_coef = 0.0
    config.cliprange = 0.2
    config.cliprange_value = 0.2
    config.gamma = 1.0

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name, 
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        use_cache=False,
        trust_remote_code=True
    )
    model.pretrained_model.gradient_checkpointing_enable()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name, 
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        use_cache=False,
        trust_remote_code=True
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    train_file = args.train_files
    if train_file.startswith("["):
        import ast
        train_file = ast.literal_eval(train_file)[0]

    dataset = build_dataset(tokenizer, train_file)

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    if args.lambda_schedule == "step":
        scheduler = StepLambdaScheduler(args.initial_lambda, args.lambda_step_size, args.lambda_multiplier, args.max_lambda)
    elif args.lambda_schedule == "ratio":
        scheduler = RatioLambdaScheduler(args.initial_lambda, args.target_positive_ratio, args.ratio_update_rate, args.max_lambda, args.min_lambda)
    elif args.lambda_schedule == "theory":
        # Estimate total steps from dataset size and epochs
        total_steps = (len(dataset) // args.batch_size) * args.epochs
        scheduler = TheoryLambdaScheduler(total_steps, args.initial_lambda, args.final_lambda, args.initial_beta, args.final_beta)

    trainer = CustomPPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        lambda_scheduler=scheduler
    )



    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_response_length,
        "use_cache": True,
        "batch_size": args.mini_batch_size,
    }

    # Force TRL to keep our string columns!
    trainer.dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)

    for epoch, batch in enumerate(trainer.dataloader):
        query_tensors = batch["input_ids"]
        query_tensors = [q.clone().detach().to(dtype=torch.long) if isinstance(q, torch.Tensor) else torch.tensor(q, dtype=torch.long) for q in query_tensors]

        print(f"Generating responses for batch of {len(query_tensors)} prompts... (This takes a moment)")
        response_tensors, ref_response_tensors = trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
            
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = []
        token_level_scores_list = []
        for i, response_str in enumerate(responses):
            gt = batch["ground_truth"][i]
            score = format_reward(batch["query"][i], response_str, gt)
                
            r = torch.tensor(score, dtype=torch.float32)
            rewards.append(r)
            
            ts = torch.zeros(len(response_tensors[i]), dtype=torch.float32)
            ts[-1] = score
            token_level_scores_list.append(ts)
            
        trainer.current_token_level_scores = torch.nn.utils.rnn.pad_sequence(
            [ts.flip(0) for ts in token_level_scores_list], batch_first=True, padding_value=0.0
        ).flip(1).to(model.pretrained_model.device)
        
        rewards_tensors = [r.to(model.pretrained_model.device) for r in rewards]

        stats = trainer.step(query_tensors, response_tensors, rewards_tensors)
        
        batch["response"] = responses
        trainer.log_stats(stats, batch, rewards_tensors)
        
        avg_reward = sum([r.sum().item() for r in rewards_tensors])/len(rewards_tensors)
        current_lambda, current_beta = scheduler.get_lambda()
        print(f"Step {epoch} | Policy Loss: {stats.get('loss/policy', 0):.4f} | Lambda: {current_lambda:.4f} | Beta: {current_beta:.4f} | Reward: {avg_reward:.4f}")
        torch.cuda.empty_cache()

    trainer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
