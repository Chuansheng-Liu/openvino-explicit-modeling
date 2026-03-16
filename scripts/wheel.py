#!/usr/bin/env python3
"""Smoke and perf test for the locally built openvino-genai wheel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from openvino import Tensor
import openvino_genai as ovg


def read_model_pair(xml_path: Path, bin_path: Path) -> tuple[str, Tensor]:
    model = xml_path.read_text(encoding="utf-8")
    weights = np.frombuffer(bin_path.read_bytes(), dtype=np.uint8).astype(np.uint8)
    return model, Tensor(weights)


def make_tokenizer(model_dir: Path) -> ovg.Tokenizer:
    tok_model, tok_weights = read_model_pair(
        model_dir / "openvino_tokenizer.xml",
        model_dir / "openvino_tokenizer.bin",
    )
    detok_model, detok_weights = read_model_pair(
        model_dir / "openvino_detokenizer.xml",
        model_dir / "openvino_detokenizer.bin",
    )
    return ovg.Tokenizer(tok_model, tok_weights, detok_model, detok_weights)


def resolve_model_paths(parser: argparse.ArgumentParser, model_arg: str) -> tuple[Path, Path, Path]:
    model_xml = Path(model_arg)
    if model_xml.suffix.lower() != ".xml":
        parser.error("--model must point to an OpenVINO .xml file")

    if not model_xml.is_file():
        parser.error(f"model xml not found: {model_xml}")

    model_bin = model_xml.with_suffix(".bin")
    if not model_bin.is_file():
        parser.error(f"expected model bin next to xml, but not found: {model_bin}")

    model_dir = model_xml.parent
    required_tokenizer_files = (
        model_dir / "openvino_tokenizer.xml",
        model_dir / "openvino_tokenizer.bin",
        model_dir / "openvino_detokenizer.xml",
        model_dir / "openvino_detokenizer.bin",
    )
    for tokenizer_file in required_tokenizer_files:
        if not tokenizer_file.is_file():
            parser.error(f"required tokenizer file not found: {tokenizer_file}")

    return model_xml, model_bin, model_dir


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampling-policy",
        choices=("greedy", "multinomial", "beam_search"),
        default="greedy",
        help="Generation policy preset based on openvino.genai Python samples",
    )
    parser.add_argument("--min-new-tokens", type=int, default=None, help="Minimum generated tokens")
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS and continue until max tokens")
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        help="Stop generation when this string is produced; can be specified multiple times",
    )
    parser.add_argument("--do-sample", action="store_true", help="Force multinomial sampling")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty")
    parser.add_argument("--rng-seed", type=int, default=None, help="Random seed for multinomial sampling")
    parser.add_argument("--num-beams", type=int, default=None, help="Beam search beam count")
    parser.add_argument("--num-beam-groups", type=int, default=None, help="Beam group count")
    parser.add_argument("--diversity-penalty", type=float, default=None, help="Beam search diversity penalty")
    parser.add_argument("--length-penalty", type=float, default=None, help="Beam search length penalty")
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=None,
        help="Number of sequences to return",
    )


def build_generation_config(args: argparse.Namespace) -> ovg.GenerationConfig:
    config = ovg.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    if args.min_new_tokens is not None:
        config.min_new_tokens = args.min_new_tokens
    if args.ignore_eos:
        config.ignore_eos = True
    if args.stop_string:
        config.stop_strings = args.stop_string

    if args.sampling_policy == "multinomial":
        config.do_sample = True
        if args.top_p is None:
            config.top_p = 0.9
        if args.top_k is None:
            config.top_k = 30
    elif args.sampling_policy == "beam_search":
        actual_num_beams = args.num_beams if args.num_beams is not None else 6
        actual_num_beam_groups = args.num_beam_groups if args.num_beam_groups is not None else 3
        config.num_beams = actual_num_beams
        if args.num_return_sequences is None:
            config.num_return_sequences = 3
        config.num_beam_groups = actual_num_beam_groups
        if args.diversity_penalty is None and actual_num_beam_groups > 1:
            config.diversity_penalty = 1.0

    if args.do_sample:
        config.do_sample = True

    scalar_fields = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "rng_seed": args.rng_seed,
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
    }
    for field_name, value in scalar_fields.items():
        if value is not None:
            setattr(config, field_name, value)

    config.validate()
    return config


def detect_generation_mode(config: ovg.GenerationConfig) -> str:
    if config.is_beam_search():
        return "beam_search"
    if config.is_multinomial():
        return "multinomial"
    if config.is_greedy_decoding():
        return "greedy"
    return "custom"


def mean_std_str(mean_std: object) -> str:
    mean = getattr(mean_std, "mean", None)
    std = getattr(mean_std, "std", None)
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.2f} +- {std:.2f}"


def print_outputs(texts: Iterable[str]) -> None:
    for index, text in enumerate(texts, start=1):
        print(f"result_{index}:")
        print(text)


def print_perf_summary(perf_metrics: object, prompt_token_size: int) -> None:
    print(f"prompt_tokens={prompt_token_size}")
    print(f"generated_tokens={perf_metrics.get_num_generated_tokens()}")
    print(f"load_time_ms={perf_metrics.get_load_time():.2f}")
    print(f"generate_duration_ms={mean_std_str(perf_metrics.get_generate_duration())}")
    print(f"tokenization_duration_ms={mean_std_str(perf_metrics.get_tokenization_duration())}")
    print(f"detokenization_duration_ms={mean_std_str(perf_metrics.get_detokenization_duration())}")
    print(f"ttft_ms={mean_std_str(perf_metrics.get_ttft())}")
    print(f"tpot_ms={mean_std_str(perf_metrics.get_tpot())}")
    print(f"throughput_tokens_per_s={mean_std_str(perf_metrics.get_throughput())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a cached OpenVINO IR xml file; the matching .bin and tokenizer files must exist beside it",
    )
    parser.add_argument("--device", default="GPU", help="Inference device, e.g. GPU or CPU")
    parser.add_argument("--prompt", default="question: what is ffmpeg?", help="Prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=2400, help="Maximum generated tokens")
    parser.add_argument("--num-warmup", type=int, default=1, help="Warmup generation iterations")
    parser.add_argument("--num-iter", type=int, default=2, help="Measured generation iterations")
    add_sampling_args(parser)
    args = parser.parse_args()

    model_xml, model_bin, model_dir = resolve_model_paths(parser, args.model)
    model, weights = read_model_pair(model_xml, model_bin)

    init_start = time.perf_counter()
    tokenizer = make_tokenizer(model_dir)
    pipe = ovg.LLMPipeline(model, weights, tokenizer, args.device)
    init_elapsed = time.perf_counter() - init_start

    config = build_generation_config(args)
    prompt_batch = [args.prompt]
    input_data = tokenizer.encode(prompt_batch)
    prompt_token_size = input_data.input_ids.get_shape()[1]

    for _ in range(args.num_warmup):
        pipe.generate(prompt_batch, config)

    generate_start = time.perf_counter()
    result = pipe.generate(prompt_batch, config)
    perf_metrics = result.perf_metrics
    for _ in range(args.num_iter - 1):
        result = pipe.generate(prompt_batch, config)
        perf_metrics += result.perf_metrics
    generate_elapsed = time.perf_counter() - generate_start

    print(f"openvino_genai={ovg.__version__}")
    print(f"device={args.device}")
    print(f"sampling_policy={args.sampling_policy}")
    print(f"generation_mode={detect_generation_mode(config)}")
    print(f"pipeline_init_sec={init_elapsed:.2f}")
    print(f"measured_generate_wall_sec={generate_elapsed:.2f}")
    print_perf_summary(perf_metrics, prompt_token_size)
    print_outputs(result.texts)


if __name__ == "__main__":
    main()
