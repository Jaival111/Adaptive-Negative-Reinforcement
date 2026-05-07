import os
import json
import argparse
from typing import List
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from utils import extract_answer_math
from grader import math_equal
os.environ["NCCL_DEBUG"] = "WARN"


def resolve_model_path(model_path: str) -> str:
    # Training saves to <output_dir>/final by default; accept either root or final path.
    final_path = os.path.join(model_path, "final")
    if os.path.isdir(final_path) and os.path.exists(os.path.join(final_path, "config.json")):
        return final_path
    return model_path


def _read_base_model_name(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("_name_or_path", "") or "")
    except Exception:
        return ""


def resolve_tokenizer_source(args) -> str:
    # Prefer explicit override; useful when checkpoint tokenizer metadata is incompatible.
    if getattr(args, "tokenizer_name", None):
        return args.tokenizer_name
    base_model_name = _read_base_model_name(args.model_name)
    if base_model_name:
        return base_model_name
    return args.model_name


def prepare_data(example, prompt_key):
    qwen_boxed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    example['prompt'] = qwen_boxed_prompt.replace("{input}", example[prompt_key])

    return example


def init_vllm_engine(args):

    tokenizer_source = resolve_tokenizer_source(args)
    llm = LLM(
        args.model_name,
        tokenizer=tokenizer_source,
        tensor_parallel_size=args.num_gpus,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=args.num_generation,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    return llm, sampling_params


def init_hf_engine(args):
    tokenizer = None
    tokenizer_error = None
    tokenizer_sources = [resolve_tokenizer_source(args), args.model_name]
    base_model_name = _read_base_model_name(args.model_name)
    if base_model_name and base_model_name not in tokenizer_sources:
        tokenizer_sources.append(base_model_name)

    for src in tokenizer_sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
            break
        except Exception as e_fast:
            try:
                tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True, use_fast=False)
                break
            except Exception as e_slow:
                tokenizer_error = (src, e_fast, e_slow)

    if tokenizer is None:
        src, e_fast, e_slow = tokenizer_error
        raise RuntimeError(
            f"Failed to load tokenizer from '{src}'. Fast tokenizer error: {repr(e_fast)}; "
            f"slow tokenizer error: {repr(e_slow)}. Install sentencepiece/tiktoken or provide tokenizer files."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def apply_hf_safety_limits(args):
    # vLLM-oriented eval settings are often too large for plain HF generate.
    capped = False
    if args.batch_size > 4:
        print(f"[WARN] HF backend capping batch_size from {args.batch_size} to 4 to avoid OOM.")
        args.batch_size = 4
        capped = True
    if args.num_generation > 4:
        print(f"[WARN] HF backend capping num_generation from {args.num_generation} to 4 to avoid OOM.")
        args.num_generation = 4
        capped = True
    if args.max_tokens > 1024:
        print(f"[WARN] HF backend capping max_tokens from {args.max_tokens} to 1024 to avoid OOM.")
        args.max_tokens = 1024
        capped = True
    if capped:
        print(
            f"[INFO] HF effective settings: batch_size={args.batch_size}, "
            f"num_generation={args.num_generation}, max_tokens={args.max_tokens}"
        )


def generate_with_hf(model, tokenizer, inputs: List[str], args) -> List[List[str]]:
    encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "num_return_sequences": args.num_generation,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**encoded, **gen_kwargs)

    completion_ids = output_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    grouped = []
    for i in range(len(inputs)):
        start = i * args.num_generation
        grouped.append(decoded[start:start + args.num_generation])
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--resume_id", type=int, default=0)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="",
        help="Optional tokenizer source (e.g., Qwen/Qwen2.5-Math-1.5B).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "hf"],
        help="auto: try vllm then fallback to hf; vllm: require vllm; hf: use transformers generate.",
    )
    args = parser.parse_args()
    print(args)
    args.model_name = resolve_model_path(args.model_name)
    print(f"Resolved model path: {args.model_name}")
    if not os.path.exists(args.model_name):
        print(f"Model {args.model_name} not found. Skip.")
        return

    # Load the model and tokenizer
    print(f"Loading model {args.model_name}")
    backend = args.backend
    llm = None
    sampling_params = None
    hf_model = None
    hf_tokenizer = None
    if backend in ("auto", "vllm"):
        try:
            llm, sampling_params = init_vllm_engine(args)
            backend = "vllm"
            print("Using backend: vllm")
        except Exception as e:
            if args.backend == "vllm":
                raise
            print(f"[WARN] vllm backend failed to initialize: {repr(e)}")
            print("[WARN] Falling back to transformers backend.")
            backend = "hf"
    if backend == "hf":
        apply_hf_safety_limits(args)
        hf_model, hf_tokenizer = init_hf_engine(args)
        print("Using backend: hf")

    # Load the dataset
    datasets = args.datasets.split(",")
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, split=args.split)
        # dataset = dataset.filter(lambda example: example['level'] == 'Level 5')
        print(f"Starting from index {args.resume_id} out of {len(dataset)} examples.")
        dataset = dataset.select(range(args.resume_id, len(dataset)))
        if "math" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "solution"
        elif "aime" in dataset_name.lower() or "amc23" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "answer"
        dataset = dataset.map(lambda x: prepare_data(x, prompt_key), num_proc=args.dataset_num_proc)

        output_file = dataset_name.split("/")[-1] + '-' + args.split + '-temp_' + str(args.temperature) + "-top_p_" + str(args.top_p) + "-top_k_" + str(args.top_k) + f'{args.comment}.jsonl'
        output_dir = args.output_dir
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if local_rank == 0 and args.resume_id == 0 and os.path.exists(os.path.join(output_dir, output_file)):
            raise FileExistsError(f"Output file {output_file} already exists.")
        # Create a JSONL file to store the output
        with open(os.path.join(output_dir, output_file), 'w' if args.resume_id == 0 else 'a') as f:
            for i in tqdm(range(0, len(dataset), args.batch_size)):
                batch = dataset[i:i + args.batch_size]
                inputs = batch["prompt"]
                answers = batch[answer_key]

                # Generate the answer
                if backend == "vllm":
                    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
                    results = [[_.outputs[l].text for l in range(len(_.outputs))] for _ in outputs]
                else:
                    results = generate_with_hf(hf_model, hf_tokenizer, inputs, args)
                assert len(results[0]) == args.num_generation, f"Number of generations is not equal to {args.num_generation}, got {len(results[0])}"

                # Prepare all outputs for batch tokenization
                flat_outputs = []
                output_mapping = []  # To map back to original indices
                
                for j in range(len(results)):
                    for k in range(args.num_generation):
                        flat_outputs.append(results[j][k])
                        output_mapping.append((j, k))

                # Process the results
                output_idx = 0
                for j, (inp, q, a, r) in enumerate(zip(inputs, batch[prompt_key], answers, results)):
                    for k in range(args.num_generation):
                        qa_pair = {
                            "prompt": inp,
                            "vanilla_response": r[k],
                            "question": q,
                            "answer": a,
                            "question_id": args.resume_id + i + j,
                            "generation_id": k,
                        }
                        qa_pair["response"] = r[k]
                        output_idx += 1
                        if "math" in dataset_name.lower():
                            gold_answer = extract_answer_math(a)
                            pred_answer = extract_answer_math(qa_pair["response"])
                        elif "amc23" in dataset_name.lower() or "aime" in dataset_name.lower():
                            gold_answer = a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        # qa_pair["label"] = pred_answer == gold_answer
                        qa_pair["label"] = math_equal(pred_answer, gold_answer, timeout=True)
                        qa_pair["gold_answer"] = gold_answer
                        qa_pair["pred_answer"] = pred_answer
                        f.write(json.dumps(qa_pair) + '\n')
                f.flush()


if __name__ == "__main__":
    main()
