<div align="center">

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.01347)
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/tianhongzxy/status/1929596099154633036)
[![Hugging Face](https://img.shields.io/badge/RLVR_Decomposed-fcd022?style=for-the-badge&logo=Huggingface&logoColor=000)](https://huggingface.co/collections/TianHongZXY/rlvr-decomposed-683c0cd7151b769d8ea5915c)

</div>

## News
- **[2025/10/27]** An updated version is out! It includes additional experiments and results (Llama model, λ ablation, and more). Check it out [here](https://arxiv.org/pdf/2506.01347)! ✨
- **[2025/09/18]** Our paper is accepted to NeurIPS 2025! 🎉
- **[2025/06/01]** We release our [paper](https://arxiv.org/pdf/2506.01347) and code. 🚀
## Quick Start
### Installation
This repository now includes a PyTorch-native PSR/NSR/W-REINFORCE training path with dynamic λ scheduling.

Start from a custom environment:
```
conda create -y -n rlvr python=3.10.14 && conda activate rlvr
pip install -e .
pip install torch transformers datasets
pip install latex2sympy2
pip install fire
```
Optional for faster generation/evaluation: install `vllm`.

## Training
PSR, NSR, W-REINFORCE (PyTorch): specify `advantage` in `run_qwen2.5-math-7b_psr_nsr.sh` or `run_qwen3-4b_psr_nsr.sh`.
For weighted mode, dynamic λ is controlled by `initial_lambda`, `lambda_schedule`, `lambda_step_size`, `lambda_multiplier`, and `max_lambda`.
```
bash run_qwen2.5-math-7b_psr_nsr.sh
```

Direct PyTorch entrypoint:
```
python train_pytorch_psr_nsr.py --help
```

Single-node Slurm launch (Tyrone profile, 2 GPUs on one node):
```
GPUS_PER_NODE=2 bash run_qwen2.5-math-7b_psr_nsr.sh
GPUS_PER_NODE=2 bash run_qwen2.5-math-7b_ppo.sh
GPUS_PER_NODE=2 bash run_qwen2.5-math-7b_grpo.sh
```

PPO (PyTorch)
```
bash run_qwen2.5-math-7b_ppo.sh
```
GRPO (PyTorch)
```
bash run_qwen2.5-math-7b_grpo.sh
```

Direct PyTorch entrypoints:
```
python train_pytorch_ppo.py --help
python train_pytorch_grpo.py --help
```

## Evaluation
Specify `MODEL_PATH` and `OUTPUT_DIR` in `eval.sh`, then `bash eval.sh`.

Cluster-aware defaults in `eval.sh`:
- `NUM_GPUS` defaults to `GPUS_PER_NODE` (default `2`).
- Override as needed: `NUM_GPUS=2 BATCH_SIZE=512 NUM_GENERATION=256 bash eval.sh`.

Calculate Pass@k: `python calculate_metrics --file_path <file_to_evaluate>`

## Troubleshoot
- PyTorch training OOM: reduce `--batch_size`, `--num_generations`, and `--max_response_length` in the PSR/NSR/PPO/GRPO run scripts.
- DDP startup hangs: verify `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `GPUS_PER_NODE`, and `SLURM_NODEID` are consistent across nodes.

 ## Citation

If you find our paper or code useful, please consider cite our work:

```bibtex
@article{zhu2025rlvr-decomposed,
  title={The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning},
  author={Zhu, Xinyu and Xia, Mengzhou and Wei, Zhepei and Chen, Wei-Lin and Chen, Danqi and Meng, Yu},
  journal={arXiv preprint arXiv:2506.01347},
  year={2025}
}
```
