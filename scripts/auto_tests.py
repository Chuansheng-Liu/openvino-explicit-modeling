from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PROMPT = "introduce ffmpeg in details"
PROMPT_FILE_NAME = "prompt_1k.txt"
COMMON_ARGS = ["GPU", "1", "1", "100"]

DEFAULT_MODELS_ROOT = r"D:\data\models"
DEFAULT_BUILD_TYPE = "Release"
FALLBACK_BUILD_TYPE = "RelWithDebInfo"
SUPPORTED_BUILD_TYPES: Tuple[str, str] = (DEFAULT_BUILD_TYPE, FALLBACK_BUILD_TYPE)
BUILD_TYPE_TOKEN = "__BUILD_TYPE__"
OPENVINO_GENAI_REQUIRED_DLLS: Tuple[str, ...] = (
    "openvino_genai.dll",
    "openvino_tokenizers.dll",
)
OPENVINO_RUNTIME_REQUIRED_DLLS: Tuple[str, ...] = ("openvino.dll",)

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT_DEFAULT = SCRIPT_DIR.parent
TEST_IMAGE_PATH = SCRIPT_DIR / "test.jpg"
TEST_OCR2_IMAGE_PATH = SCRIPT_DIR / "test_ocr2.png"
TEST_AUDIO_PATH = SCRIPT_DIR / "asr_zh.wav"
PROMPT_FILE_PATH = SCRIPT_DIR / PROMPT_FILE_NAME


def load_prompt(prompt_file_path: Path, fallback_prompt: str) -> str:
    try:
        content = prompt_file_path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback_prompt
    return content if content else fallback_prompt


PROMPT = load_prompt(PROMPT_FILE_PATH, DEFAULT_PROMPT)

TEXT_EXE_REL = (
    Path("openvino.genai")
    / "build"
    / "bin"
    / BUILD_TYPE_TOKEN
    / "greedy_causal_lm.exe"
)
TEXT_WORK_DIR_REL = Path("openvino") / "bin" / "intel64" / BUILD_TYPE_TOKEN
GENAI_BIN_REL = Path("openvino.genai") / "build" / "bin" / BUILD_TYPE_TOKEN
MODELING_QWEN_EXE_REL = GENAI_BIN_REL / "modeling_qwen3_vl.exe"
MODELING_QWEN3_5_EXE_REL = GENAI_BIN_REL / "modeling_qwen3_5.exe"
MODELING_DEEPSEEK_OCR2_EXE_REL = GENAI_BIN_REL / "modeling_deepseek_ocr2.exe"
MODELING_ZIMAGE_EXE_REL = GENAI_BIN_REL / "modeling_zimage.exe"
MODELING_WAN_T2V_EXE_REL = GENAI_BIN_REL / "modeling_wan_t2v.exe"
MODELING_DFLASH_EXE_REL = GENAI_BIN_REL / "modeling_dflash.exe"
MODELING_QWEN3_TTS_EXE_REL = GENAI_BIN_REL / "modeling_qwen3_tts.exe"
BENCHMARK_GENAI_EXE_REL = GENAI_BIN_REL / "benchmark_genai.exe"
MODELING_QWEN3_ASR_EXE_REL = GENAI_BIN_REL / "modeling_qwen3_asr.exe"

MODELING_ULT_EXE_REL = (
    Path("openvino.genai")
    / "build"
    / "bin"
    / BUILD_TYPE_TOKEN
    / "test_modeling_api.exe"
)
PATH_PREPEND_REL = Path("openvino.genai") / "build" / "openvino_genai"
TBB_BIN_REL_CANDIDATES: Tuple[Path, ...] = (
    Path("openvino") / "temp" / "Windows_AMD64" / "tbb" / "bin",
)

TEXT_COMMAND_ARGS = [PROMPT, *COMMON_ARGS]
MODELING_QWEN_ARGS = [str(TEST_IMAGE_PATH), "describe this picture", "GPU", "100"]
MODELING_DEEPSEEK_OCR2_ARGS = [
    str(TEST_OCR2_IMAGE_PATH),
    "<image>\n<|grounding|>Convert the document to markdown.",
    "GPU",
    "300",
]
MODELING_ZIMAGE_ARGS = [
    "a cute cat",
    "cat2.bmp",
    "GPU",
    "256",
    "256",
    "8",
    "2",
    "0.0",
]
MODELING_WAN_T2V_ARGS = [
    "a cat playing piano",
    "wan_t2v_out",
    "GPU",
    "256",
    "384",
    "33",
    "30",
    "0",
    "5.0",
    "",
    "512",
]


MODELING_DFLASH_ARGS = [
    "__DRAFT_MODEL__",
    PROMPT,
    "GPU",
    "100",
    "16",
]
MODELING_QWEN3_TTS_ARGS = [
    "我爱北京天安门",
    "qwen3_tts_out.wav",
    "GPU",
]
MODELING_QWEN3_ASR_AUDIO_ARGS = [
    "--cache-model",
    "--wav",
    str(TEST_AUDIO_PATH),
    "--device",
    "GPU",
    "--max_new_tokens",
    "200",
]
MODELING_QWEN3_ASR_TEXT_ARGS = [
    "--cache-model",
    "--text-only",
    "--prompt",
    PROMPT,
    "--device",
    "GPU",
    "--max_new_tokens",
    "200",
]
MODELING_QWEN3_5_TEXT_ARGS = [
    "--cache-model",
    "--mode",
    "text",
    "--prompt",
    PROMPT,
    "--output-tokens",
    "300",
]
QWEN3_5_35B_GREEDY_TEXT_ARGS = [
    PROMPT,
    "GPU",
    "1",
    "3",
    "300",
    "int4_asym",
    "128",
    "int4_asym",
]
QWEN3_5_35B_BENCHMARK_ARGS = [
    "-p",
    PROMPT,
    "-n",
    "1",
    "--mt",
    "300",
    "-d",
    "GPU",
]
MODELING_QWEN3_5_VL_ARGS = [
    "--cache-model",
    "--mode",
    "vl",
    "--image",
    str(TEST_IMAGE_PATH),
    "--prompt",
    "describe this picture in details: ",
    "--output-tokens",
    "300",
]

QUANT_DEFAULT_ARGS = ["int4_asym", "128", "int8_asym"]
QUANT_INT4_CHANNEL_WISE_ARGS = ["int4_asym", "-1", "int8_asym"]
QUANT_INT8_CHANNEL_WISE_ARGS = ["int8_asym", "-1", "int8_asym"]
QWEN3_5_35B_EXTRA_ENV = {
    "OV_GPU_MOE_DISABLE_ONEDNN": "1",
    "OV_GENAI_USE_MODELING_API": "1",
    "OV_GENAI_INFLIGHT_QUANT_MODE": "int4_asym",
    "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE": "128",
    "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE": "int4_asym",
}

TEST_SPECS: List[Dict[str, Any]] = [
    {
        "name": "Modeling API Unit Tests",
        "model_rel": None,
        "exe_rel": MODELING_ULT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": ["--gtest_filter=*"],
        "is_ult": True,
    },
    {
        "name": "Huggingface Qwen3-0.6B",
        "model_rel": Path("Huggingface") / "Qwen3-0.6B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "Huggingface Qwen3-4B",
        "model_rel": Path("Huggingface") / "Qwen3-4B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "Huggingface Qwen3-4B in-flight quantized (int4_asym, gs128)",
        "model_rel": Path("Huggingface") / "Qwen3-4B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_DEFAULT_ARGS,
    },
    {
        "name": "Huggingface Qwen3-4B in-flight quantized (int8_asym, channel-wise)",
        "model_rel": Path("Huggingface") / "Qwen3-4B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_INT8_CHANNEL_WISE_ARGS,
    },
    {
        "name": "Huggingface Qwen3-4B in-flight quantized (int4_asym, channel-wise)",
        "model_rel": Path("Huggingface") / "Qwen3-4B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_INT4_CHANNEL_WISE_ARGS,
    },
    {
        "name": "Huggingface Qwen3-8B in-flight quantized (int4_asym, gs128)",
        "model_rel": Path("Huggingface") / "Qwen3-8B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_DEFAULT_ARGS,
    },
    {
        "name": "Huggingface SmolLM3-3B",
        "model_rel": Path("Huggingface") / "SmolLM3-3B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "Huggingface Youtu-LLM-2B",
        "model_rel": Path("Huggingface") / "Youtu-LLM-2B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "GGUF Qwen3-0.6B-BF16",
        "model_rel": Path("gguf") / "Qwen3-0.6B-BF16.gguf",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "GGUF Qwen3-4B-BF16",
        "model_rel": Path("gguf") / "Qwen3-4B-BF16.gguf",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    {
        "name": "GGUF Qwen3-30B-A3B",
        "model_rel": Path("gguf") / "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
    },
    # {
    #     "name": "GGUF SmolLM3-3B-BF16",
    #     "model_rel": Path("gguf") / "SmolLM3-3B-BF16.gguf",
    #     "exe_rel": TEXT_EXE_REL,
    #     "work_dir_rel": TEXT_WORK_DIR_REL,
    #     "command_args": TEXT_COMMAND_ARGS.copy(),
    # },
    {
        "name": "Huggingface Qwen3-VL-2B-Instruct",
        "model_rel": Path("Huggingface") / "Qwen3-VL-2B-Instruct",
        "exe_rel": MODELING_QWEN_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN_ARGS.copy(),
    },
    {
        "name": "Huggingface Qwen3-VL-4B-Instruct",
        "model_rel": Path("Huggingface") / "Qwen3-VL-4B-Instruct",
        "exe_rel": MODELING_QWEN_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN_ARGS.copy(),
    },
    {
        "name": "Huggingface DeepSeek-OCR-2",
        "model_rel": Path("Huggingface") / "DeepSeek-OCR-2",
        "exe_rel": MODELING_DEEPSEEK_OCR2_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_DEEPSEEK_OCR2_ARGS.copy(),
    },
    {
       "name": "Huggingface Qwen3-VL-8B-Instruct-inflight-quantized (int4_asym, gs128)",
       "name": "Huggingface Qwen3-VL-8B-all-inflight-quantized (int4_asym, gs128)",
       "model_rel": Path("Huggingface") / "Qwen3-VL-8B-Instruct",
       "exe_rel": MODELING_QWEN_EXE_REL,
       "work_dir_rel": TEXT_WORK_DIR_REL,
       # [VISION_QUANT] [VISION_GS] [VISION_BACKUP] [TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP]
       "command_args": MODELING_QWEN_ARGS.copy() + [
           "int4_asym", "-1", "int8_asym",
           "int4_asym", "128", "int8_asym"
       ],
    },
    {
       "name": "Huggingface Qwen3-VL-8B-only-text-inflight-quantized (int4_asym, gs128)",
       "model_rel": Path("Huggingface") / "Qwen3-VL-8B-Instruct",
       "exe_rel": MODELING_QWEN_EXE_REL,
       "work_dir_rel": TEXT_WORK_DIR_REL,
       # [VISION_QUANT] [VISION_GS] [VISION_BACKUP] [TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP]
       "command_args": MODELING_QWEN_ARGS.copy() + [
           "none", "-1", "none",
           "int4_asym", "128", "int8_asym"
       ],
    },
    {
        "name": "Huggingface Z-Image-Turbo",
        "model_rel": Path("Huggingface") / "Z-Image-Turbo",
        "exe_rel": MODELING_ZIMAGE_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_ZIMAGE_ARGS.copy(),
    },
    #{
    #    "name": "Huggingface Z-Image-Turbo (int4_sym, gs128)",
    #    "model_rel": Path("Huggingface") / "Z-Image-Turbo",
    #    "exe_rel": MODELING_ZIMAGE_EXE_REL,
    #    "work_dir_rel": TEXT_WORK_DIR_REL,
    #    # [DUMP_DIR] [TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP] [DIT_QUANT] [DIT_GS] [DIT_BACKUP] [VAE_QUANT] [VAE_GS] [VAE_BACKUP]
    #    "command_args": MODELING_ZIMAGE_ARGS.copy() + [
    #        "int4_sym", "128", "int8_asym",
    #        "int4_sym", "128", "none",
    #        "none", "-1", "none"
    #    ],
    #},
    {
        "name": "Huggingface Wan2.1-T2V-1.3B-Diffusers",
        "model_rel": Path("Huggingface") / "Wan2.1-T2V-1.3B-Diffusers",
        "exe_rel": MODELING_WAN_T2V_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_WAN_T2V_ARGS.copy(),
    },
    {
        "name": "Huggingface Qwen3-4B DFlash (block_size=16)",
        "model_rel": Path("Huggingface") / "Qwen3-4B",
        "draft_model_rel": Path("Huggingface") / "Qwen3-4B-DFlash-b16",
        "exe_rel": MODELING_DFLASH_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_DFLASH_ARGS.copy(),
        "is_dflash": True,
    },
    {
        "name": "Huggingface Qwen3-MOE-4x0.6B-2.4B-legacy-path inflight-quantized (int4_asym, gs128)",
        "model_rel": Path("Huggingface") / "Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy(),
        "extra_env": {"OV_GENAI_INFLIGHT_QUANT_MODE": "int4_asym",
                      "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE": "128",
                      "OV_GENAI_USE_MODELING_API": "0"},
    },
    {
        "name": "Huggingface Qwen3-MOE-4x0.6B-2.4B-modeling-api inflight-quantized (int4_asym, gs128)",
        "model_rel": Path("Huggingface") / "Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_DEFAULT_ARGS,
    },
    {
        "name": "Huggingface Qwen3-MOE-4x0.6B-2.4B-modeling-api inflight-quantized (int4_asym, channel-wise)",
        "model_rel": Path("Huggingface") / "Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_INT4_CHANNEL_WISE_ARGS,
    },
    {
        "name": "Huggingface Qwen3-30B-A3B-Instruct-2507 inflight-quantized (int4_asym, gs128)",
        "model_rel": Path("Huggingface") / "Qwen3-30B-A3B-Instruct-2507",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": TEXT_COMMAND_ARGS.copy() + QUANT_DEFAULT_ARGS,
    },
    {
        "name": "Huggingface Qwen3-TTS-12Hz-1.7B-Base",
        "model_rel": Path("Huggingface") / "Qwen3-TTS-12Hz-1.7B-Base",
        "exe_rel": MODELING_QWEN3_TTS_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_TTS_ARGS.copy(),
    },
    {
        "name": "Huggingface Qwen3.5-0.8B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-0.8B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-0.8B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-0.8B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-2B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-2B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-2B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-2B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-4B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-4B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-4B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-4B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-9B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-9B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-9B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-9B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-27B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-27B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-27B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-27B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B-Base modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B-Base",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B-Base modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B-Base",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B modeling_qwen3_5 text",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_TEXT_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B modeling_qwen3_5 vl",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B",
        "exe_rel": MODELING_QWEN3_5_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_5_VL_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B greedy_causal_lm text",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B",
        "exe_rel": TEXT_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": QWEN3_5_35B_GREEDY_TEXT_ARGS.copy(),
        "extra_env": {
            "OV_GENAI_SAVE_OV_MODEL": "1",
        },
    },
    {
        "name": "Huggingface Qwen3.5-35B-A3B benchmark_genai text",
        "model_rel": Path("Huggingface") / "Qwen3.5-35B-A3B",
        "exe_rel": BENCHMARK_GENAI_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": QWEN3_5_35B_BENCHMARK_ARGS.copy(),
        "extra_env": QWEN3_5_35B_EXTRA_ENV.copy(),
        "use_named_model_arg": True,
    },
    {
        "name": "Huggingface Qwen3-ASR-0.6B",
        "model_rel": Path("Huggingface") / "Qwen3-ASR-0.6B",
        "exe_rel": MODELING_QWEN3_ASR_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_ASR_AUDIO_ARGS.copy(),
    },
        {
        "name": "Huggingface Qwen3-ASR-0.6B text-only",
        "model_rel": Path("Huggingface") / "Qwen3-ASR-0.6B",
        "exe_rel": MODELING_QWEN3_ASR_EXE_REL,
        "work_dir_rel": TEXT_WORK_DIR_REL,
        "command_args": MODELING_QWEN3_ASR_TEXT_ARGS.copy(),
    },
]

def parse_build_type(value: str) -> str:
    normalized = value.strip().lower()
    for candidate in SUPPORTED_BUILD_TYPES:
        if normalized == candidate.lower():
            return candidate
    choices = ", ".join(SUPPORTED_BUILD_TYPES)
    raise argparse.ArgumentTypeError(
        f"Unsupported build type: {value}. Choose from: {choices}."
    )


def resolve_build_type_path(path_rel: Path, build_type: str) -> Path:
    return Path(str(path_rel).replace(BUILD_TYPE_TOKEN, build_type))


def _remove_build_type_token_segment(path_rel: Path) -> Path:
    rel = str(path_rel)
    rel = rel.replace(f"{BUILD_TYPE_TOKEN}/", "")
    rel = rel.replace(f"{BUILD_TYPE_TOKEN}\\", "")
    return Path(rel)


def resolve_executable_path(root: Path, exe_rel: Path, build_type: str) -> Path:
    """Resolve exe path and auto-detect whether BUILD_TYPE_TOKEN is needed.

    Preferred layout uses BUILD_TYPE_TOKEN. If the executable is not found there,
    fall back to a no-build-type layout (e.g. .../bin/foo.exe).
    """
    primary = root / resolve_build_type_path(exe_rel, build_type)
    if primary.is_file():
        return primary

    if BUILD_TYPE_TOKEN not in str(exe_rel):
        return primary

    fallback = root / _remove_build_type_token_segment(exe_rel)
    if fallback.is_file():
        return fallback

    return primary


def format_rel_path(path_rel: Path, build_type: Optional[str] = None) -> str:
    replacement = build_type if build_type is not None else "<build-type>"
    return str(path_rel).replace(BUILD_TYPE_TOKEN, replacement)


def detect_layout_root(root: Path) -> Path:
    candidates = [root]
    if root.parent != root:
        candidates.append(root.parent)
    for candidate in candidates:
        if (candidate / "openvino").is_dir() and (candidate / "openvino.genai").is_dir():
            return candidate
    return root


def find_tbb_bin_dir(root: Path) -> Optional[str]:
    for rel_path in TBB_BIN_REL_CANDIDATES:
        candidate = root / rel_path
        if candidate.is_dir() and (candidate / "tbb12.dll").is_file():
            return str(candidate)

    tbb_glob_root = root / "openvino" / "temp"
    if tbb_glob_root.is_dir():
        for candidate in sorted(tbb_glob_root.glob("*/tbb/bin")):
            if candidate.is_dir() and (candidate / "tbb12.dll").is_file():
                return str(candidate)
    return None


def build_path_entries(root: Path, build_type: str) -> List[str]:
    path_entries = [
        str(root / PATH_PREPEND_REL),
        str(root / resolve_build_type_path(TEXT_WORK_DIR_REL, build_type)),
    ]
    tbb_bin_dir = find_tbb_bin_dir(root)
    if tbb_bin_dir:
        path_entries.insert(0, tbb_bin_dir)
    return path_entries


def collect_missing_build_artifacts(
    root: Path, tests: List[Dict[str, Any]], build_type: str
) -> List[str]:
    missing: List[str] = []
    reported: Dict[str, None] = {}

    def add_missing(description: str, path: Path) -> None:
        key = f"{description}|{path}"
        if key in reported:
            return
        reported[key] = None
        missing.append(f"{description}: {path}")

    openvino_genai_dir = root / PATH_PREPEND_REL
    if not openvino_genai_dir.is_dir():
        add_missing("OpenVINO GenAI runtime directory not found", openvino_genai_dir)
    for dll_name in OPENVINO_GENAI_REQUIRED_DLLS:
        dll_path = openvino_genai_dir / dll_name
        if not dll_path.is_file():
            add_missing("OpenVINO GenAI runtime DLL not found", dll_path)

    openvino_runtime_dir = root / resolve_build_type_path(TEXT_WORK_DIR_REL, build_type)
    if not openvino_runtime_dir.is_dir():
        add_missing(
            f"OpenVINO runtime directory not found for build type {build_type}",
            openvino_runtime_dir,
        )
    for dll_name in OPENVINO_RUNTIME_REQUIRED_DLLS:
        dll_path = openvino_runtime_dir / dll_name
        if not dll_path.is_file():
            add_missing(
                f"OpenVINO runtime DLL not found for build type {build_type}",
                dll_path,
            )

    for test in tests:
        exe_path = Path(test["exe"])
        work_dir = Path(test["work_dir"])
        if not exe_path.is_file():
            add_missing(
                f"Test executable not found for build type {build_type}",
                exe_path,
            )
        if not work_dir.is_dir():
            add_missing(
                f"Working directory not found for build type {build_type}",
                work_dir,
            )

    return missing


def format_missing_build_artifacts(build_type: str, missing: List[str]) -> str:
    lines = [f"Build type '{build_type}' is not runnable. Missing artifacts:"]
    lines.extend(f"  - {item}" for item in missing)
    return "\n".join(lines)


def build_env(
    path_entries: List[str], extra_env: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    env = os.environ.copy()
    applied_env: Dict[str, str] = {}
    original_path = env.get("PATH", "")
    joined = ";".join(entry for entry in path_entries if entry)
    env["PATH"] = f"{joined};{original_path}" if original_path else joined
    applied_env["PATH"] = env["PATH"]
    if extra_env and extra_env.get("PATH"):
        env["PATH"] = f"{extra_env['PATH']};{env['PATH']}"
        applied_env["PATH"] = env["PATH"]
    # Set OV_GENAI_USE_MODELING_API from extra_env if present, else from os.environ, else default to "1"
    if extra_env and "OV_GENAI_USE_MODELING_API" in extra_env:
        env["OV_GENAI_USE_MODELING_API"] = extra_env["OV_GENAI_USE_MODELING_API"]
    else:
        env["OV_GENAI_USE_MODELING_API"] = os.environ.get("OV_GENAI_USE_MODELING_API", "1")
    applied_env["OV_GENAI_USE_MODELING_API"] = env["OV_GENAI_USE_MODELING_API"]
    if extra_env:
        for k, v in extra_env.items():
            if k in {"PATH", "OV_GENAI_USE_MODELING_API"}:
                continue
            env[k] = v
            applied_env[k] = v
    return env, applied_env


def format_env_commands(applied_env: Dict[str, str]) -> List[str]:
    return [f"set {key}={value}" for key, value in applied_env.items()]


def extract_performance(output: str) -> str:
    labels = [
        "Prompt token size:",
        "Output token size:",
        "Load time:",
        "Generate time:",
        "Tokenization time:",
        "Detokenization time:",
        "TTFT:",
        "TPOT:",
        "Throughput:",
    ]
    perf_lines: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        for label in labels:
            if stripped.startswith(label):
                perf_lines.append(stripped)
                break
    return "\n".join(perf_lines).strip() if perf_lines else "Not found in output."


def extract_generated_text(output: str) -> str:
    marker = "Generated text:"
    idx = output.find(marker)
    if idx == -1:
        # If marker not found, return all non-empty lines
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return "\n".join(lines) if lines else "Not found in output."
    text_block = output[idx + len(marker) :].lstrip("\r\n")
    return text_block.strip() if text_block.strip() else "Not found in output."


def build_command(exe_path: str, model_path: str, command_args: List[str]) -> List[str]:
    return [exe_path, model_path, *command_args]


def command_to_string(args: List[str]) -> str:
    def quote(arg: str) -> str:
        return f"\"{arg}\"" if " " in arg or "\t" in arg else arg

    return " ".join(quote(arg) for arg in args)


def format_duration(delta: _dt.timedelta) -> str:
    total_seconds = delta.total_seconds()
    if total_seconds < 60:
        return f"{total_seconds:.2f}s"
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h{minutes}m{seconds:05.2f}s"
    return f"{minutes}m{seconds:05.2f}s"


def extract_label_value(block: str, label: str) -> str:
    marker = f"{label}:"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            value = stripped[len(marker) :].strip()
            return value if value else "N/A"
    return "N/A"


def filter_ult_output(output: str) -> str:
    """Filter out [DEBUG] lines from ULT output."""
    lines = output.splitlines()
    filtered = [line for line in lines if not line.strip().startswith("[DEBUG]")]
    return "\n".join(filtered)


def format_ult_output(output: str) -> str:
    """Format ULT output for markdown report."""
    filtered = filter_ult_output(output)
    lines = filtered.splitlines()

    # Extract summary information
    summary_lines: List[str] = []
    test_results: List[str] = []
    in_test_section = False

    for line in lines:
        stripped = line.strip()
        # Capture gtest summary lines
        if stripped.startswith("[==========]") or stripped.startswith("[----------]"):
            summary_lines.append(stripped)
            in_test_section = True
        elif stripped.startswith("[  PASSED  ]") or stripped.startswith("[  FAILED  ]"):
            summary_lines.append(stripped)
        elif stripped.startswith("[       OK ]") or stripped.startswith("[   FAIL   ]"):
            test_results.append(stripped)
        elif stripped.startswith("[ RUN      ]"):
            test_results.append(stripped)
        elif in_test_section and stripped:
            # Keep other relevant lines during test execution
            if not stripped.startswith("[DEBUG]"):
                test_results.append(stripped)

    return filtered


def parse_args() -> argparse.Namespace:
    script_root = SCRIPT_ROOT_DEFAULT
    parser = argparse.ArgumentParser(
        description="Run OpenVINO GenAI greedy_causal_lm tests and capture key outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python auto_tests.py\n"
            "  python auto_tests.py --root ..\n"
            "  python auto_tests.py --root D:\\data\\code\\Openvino_new_arch_poc\\openvino-new-arch\n"
            "  python auto_tests.py --root .. --list\n"
            "  python auto_tests.py --root .. --tests 0,1,2\n"
            "  python auto_tests.py --root .. --tests 0 1 2\n"
            "  python auto_tests.py --root .. --tests 1~5,7,8~10\n"
            "  python auto_tests.py --root .. --models-root D:\\data\\models\n"
            "  python auto_tests.py --root .. --build-type RelWithDebInfo\n"
        ),
    )
    parser.add_argument(
        "--root",
        default=str(script_root),
        help=(
            "Workspace root containing openvino and openvino.genai repos. "
            "Defaults to the parent of this script and auto-detects sibling-repo layout."
        ),
    )
    parser.add_argument(
        "--models-root",
        default=DEFAULT_MODELS_ROOT,
        help=f"Root folder path for model files (default: {DEFAULT_MODELS_ROOT}).",
    )
    parser.add_argument(
        "--build-type",
        type=parse_build_type,
        help=(
            "Build type to use. Supported values: Release, RelWithDebInfo. "
            "If omitted, try Release first, then fall back to RelWithDebInfo."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Test indices to run. Supports individual indices (0,1,2), ranges (1~5), or 'all'. "
             "Examples: '0,1,2', '1~5,7,8~10', 'all'. If omitted, run all.",
    )
    return parser.parse_args()


def list_tests(models_root: Path, build_type: Optional[str] = None) -> None:
    print(f"Models root: {models_root}")
    if build_type:
        print(f"Build type: {build_type}")
    else:
        print(
            f"Build type: {DEFAULT_BUILD_TYPE} (fallback to {FALLBACK_BUILD_TYPE} if needed)"
        )
    print("Available tests:")
    for idx, spec in enumerate(TEST_SPECS):
        model_info = spec["model_rel"] if spec["model_rel"] else "N/A (ULT)"
        exe_info = format_rel_path(spec["exe_rel"], build_type)
        print(
            f"[{idx}] {spec['name']} -> {model_info} (exe: {exe_info})"
        )


def parse_test_indices(raw_items: List[str], max_index: int) -> List[int]:
    tokens: List[str] = []
    for raw in raw_items:
        tokens.extend(part.strip() for part in raw.split(",") if part.strip())
    if not tokens:
        raise ValueError("No test indices provided.")

    if any(token.lower() == "all" for token in tokens):
        return list(range(max_index + 1))

    indices: List[int] = []
    for token in tokens:
        # Support range syntax: "1~5" expands to 1,2,3,4,5
        if "~" in token:
            parts = token.split("~")
            if len(parts) != 2:
                raise ValueError(f"Invalid range syntax: {token}")
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError as exc:
                raise ValueError(f"Invalid range syntax: {token}") from exc
            if start < 0 or start > max_index:
                raise ValueError(f"Range start out of bounds: {start}")
            if end < 0 or end > max_index:
                raise ValueError(f"Range end out of bounds: {end}")
            if start > end:
                raise ValueError(f"Range start must be <= end: {token}")
            indices.extend(range(start, end + 1))
        else:
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid test index: {token}") from exc
            if idx < 0 or idx > max_index:
                raise ValueError(f"Test index out of range: {idx}")
            indices.append(idx)
    return indices


def resolve_tests(
    root: Path, models_root: Path, indices: Optional[List[int]], build_type: str
) -> List[Dict[str, Any]]:
    selected = indices if indices is not None else list(range(len(TEST_SPECS)))
    resolved: List[Dict[str, Any]] = []
    for idx in selected:
        spec = TEST_SPECS[idx]
        # Handle tests without model (like ULT)
        if spec["model_rel"] is None:
            model_path = None
        else:
            model_path = models_root / spec["model_rel"]
        
        # Handle DFlash tests with draft model
        draft_model_path = None
        if spec.get("is_dflash") and "draft_model_rel" in spec:
            draft_model_path = models_root / spec["draft_model_rel"]
        
        resolved_test = {
            "index": str(idx),
            "name": spec["name"],
            "exe": str(resolve_executable_path(root, spec["exe_rel"], build_type)),
            "work_dir": str(root / resolve_build_type_path(spec["work_dir_rel"], build_type)),
            "model": str(model_path) if model_path else None,
            "command_args": spec["command_args"],
        }
        if "extra_env" in spec:
            resolved_test["extra_env"] = spec["extra_env"].copy()
        else:
            resolved_test["extra_env"] = {}
        if spec.get("is_ult"):
            resolved_test["is_ult"] = True
            # Qwen3VLE2E.PrefillAndDecode requires QWEN3_VL_MODEL_DIR
            qwen3_vl_candidates = [
                models_root / "Huggingface" / "Qwen3-VL-2B-Instruct",
                models_root / "Huggingface" / "Qwen3-VL-4B-Instruct",
            ]
            for cand in qwen3_vl_candidates:
                if cand.is_dir():
                    resolved_test["extra_env"]["QWEN3_VL_MODEL_DIR"] = str(cand)
                    break
        if spec.get("is_dflash"):
            resolved_test["is_dflash"] = True
            resolved_test["draft_model"] = str(draft_model_path) if draft_model_path else None
        if spec.get("use_named_model_arg"):
            resolved_test["use_named_model_arg"] = True
        resolved.append(resolved_test)
    return resolved


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    workspace_root = detect_layout_root(root)
    models_root = Path(args.models_root)

    if args.list:
        list_tests(models_root, args.build_type)
        return 0

    if not workspace_root.exists():
        print(f"Root folder not found: {workspace_root}", file=sys.stderr)
        return 2

    if args.tests:
        try:
            indices = parse_test_indices(args.tests, len(TEST_SPECS) - 1)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        indices = None

    build_failures: List[Tuple[str, List[str]]] = []
    candidate_build_types = (
        [args.build_type] if args.build_type else [DEFAULT_BUILD_TYPE, FALLBACK_BUILD_TYPE]
    )
    selected_build_type: Optional[str] = None
    tests: List[Dict[str, Any]] = []

    for candidate_build_type in candidate_build_types:
        candidate_tests = resolve_tests(
            workspace_root, models_root, indices, candidate_build_type
        )
        missing_artifacts = collect_missing_build_artifacts(
            workspace_root, candidate_tests, candidate_build_type
        )
        if not missing_artifacts:
            selected_build_type = candidate_build_type
            tests = candidate_tests
            break
        build_failures.append((candidate_build_type, missing_artifacts))

    if selected_build_type is None:
        if args.build_type:
            print(
                format_missing_build_artifacts(
                    build_failures[0][0], build_failures[0][1]
                ),
                file=sys.stderr,
            )
        else:
            print(
                "Unable to find a runnable build type. Checked Release, then RelWithDebInfo.",
                file=sys.stderr,
            )
            for failed_build_type, missing_artifacts in build_failures:
                print("", file=sys.stderr)
                print(
                    format_missing_build_artifacts(
                        failed_build_type, missing_artifacts
                    ),
                    file=sys.stderr,
                )
        return 2

    if args.build_type is None and selected_build_type != DEFAULT_BUILD_TYPE:
        print(
            f"Release artifacts are incomplete. Falling back to {selected_build_type}."
        )
    print(f"Using build type: {selected_build_type}")

    path_entries = build_path_entries(workspace_root, selected_build_type)

    timestamp = _dt.datetime.now()
    stamp_for_name = timestamp.strftime("%Y%m%d_%H%M%S")
    stamp_for_title = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"test_report_{stamp_for_name}.md"

    report_lines: List[str] = [
        f"# Test Report {stamp_for_title}",
        "",
        f"Build type: {selected_build_type}",
        "",
    ]
    total_start = _dt.datetime.now()
    timing_entries: List[Dict[str, Any]] = []
    failed_tests: List[tuple] = []

    for test in tests:
        exe_path = Path(test["exe"])
        work_dir = Path(test["work_dir"])

        if not exe_path.is_file():
            print(f"Executable not found: {exe_path}", file=sys.stderr)
            return 2
        if not work_dir.is_dir():
            print(f"Working directory not found: {work_dir}", file=sys.stderr)
            return 2

        command_args = list(test["command_args"])

        # Handle Z-Image test: add timestamp to output filename
        is_zimage_test = "Z-Image" in test["name"]
        zimage_output_filename: Optional[str] = None
        if is_zimage_test and len(command_args) >= 2:
            # command_args[1] is the output filename (e.g., "cat2.bmp")
            original_filename = command_args[1]
            name_part, ext_part = os.path.splitext(original_filename)
            zimage_output_filename = f"{name_part}_{stamp_for_name}{ext_part}"
            command_args[1] = zimage_output_filename

        # Handle Wan T2V test: add timestamp to output folder name
        is_wan_t2v_test = "Wan2.1-T2V" in test["name"]
        wan_t2v_output_folder: Optional[str] = None
        if is_wan_t2v_test and len(command_args) >= 2:
            # command_args[1] is the output folder (e.g., "wan_t2v_out")
            original_folder = command_args[1]
            wan_t2v_output_folder = f"{original_folder}_{stamp_for_name}"
            command_args[1] = wan_t2v_output_folder

        # Handle Qwen3-TTS test: add timestamp to output wav filename
        is_qwen3_tts_test = "Qwen3-TTS" in test["name"]
        qwen3_tts_output_filename: Optional[str] = None
        if is_qwen3_tts_test and len(command_args) >= 2:
            # command_args[1] is the output filename (e.g., "qwen3_tts_out.wav")
            original_filename = command_args[1]
            name_part, ext_part = os.path.splitext(original_filename)
            qwen3_tts_output_filename = f"{name_part}_{stamp_for_name}{ext_part}"
            command_args[1] = qwen3_tts_output_filename

        # Handle ULT test: no model path
        is_ult_test = test.get("is_ult", False)
        is_dflash_test = test.get("is_dflash", False)
        
        if is_ult_test:
            args_list = [str(exe_path), *command_args]
        elif is_dflash_test:
            # DFlash requires: exe target_model draft_model [args...]
            # Replace __DRAFT_MODEL__ placeholder with actual draft model path
            command_args_resolved = [
                test["draft_model"] if arg == "__DRAFT_MODEL__" else arg
                for arg in command_args
            ]
            args_list = build_command(str(exe_path), test["model"], command_args_resolved)
        elif test.get("use_named_model_arg"):
            args_list = [str(exe_path), "--model", test["model"], *command_args]
        else:
            args_list = build_command(str(exe_path), test["model"], command_args)
        cmd_line = command_to_string(args_list)
        cd_cmd = f"cd {work_dir}"

        # Build env per test
        extra_env = test.get("extra_env")
        env, applied_env = build_env(path_entries, extra_env)
        env_commands = format_env_commands(applied_env)
        env_commands.append(cd_cmd)

        print("=" * 80)
        print(f"Test {test['index']}: {test['name']}")
        print(f"Command: {cmd_line}")
        test_start = _dt.datetime.now()
        result = subprocess.run(
            args_list,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output = result.stdout or ""

        # Filter debug logs for ULT tests
        if is_ult_test:
            output = filter_ult_output(output)

        perf_block = extract_performance(output)
        gen_block = extract_generated_text(output)

        # Copy Z-Image output to reports folder
        zimage_report_path: Optional[Path] = None
        if is_zimage_test and zimage_output_filename:
            zimage_src = work_dir / zimage_output_filename
            if zimage_src.is_file():
                zimage_report_path = reports_dir / zimage_output_filename
                shutil.copy2(zimage_src, zimage_report_path)
                print(f"Z-Image output copied to: {zimage_report_path}")

        # Copy Wan T2V output folder to reports folder
        wan_t2v_report_folder: Optional[Path] = None
        wan_t2v_images: List[str] = []
        if is_wan_t2v_test and wan_t2v_output_folder:
            wan_t2v_src = work_dir / wan_t2v_output_folder
            if wan_t2v_src.is_dir():
                wan_t2v_report_folder = reports_dir / wan_t2v_output_folder
                if wan_t2v_report_folder.exists():
                    shutil.rmtree(wan_t2v_report_folder)
                shutil.copytree(wan_t2v_src, wan_t2v_report_folder)
                print(f"Wan T2V output folder copied to: {wan_t2v_report_folder}")
                # Collect all BMP images in the folder
                wan_t2v_images = sorted(
                    [f.name for f in wan_t2v_report_folder.iterdir() if f.suffix.lower() == ".bmp"]
                )

        # Copy Qwen3-TTS output to reports folder
        qwen3_tts_report_path: Optional[Path] = None
        if is_qwen3_tts_test and qwen3_tts_output_filename:
            qwen3_tts_src = work_dir / qwen3_tts_output_filename
            if qwen3_tts_src.is_file():
                qwen3_tts_report_path = reports_dir / qwen3_tts_output_filename
                shutil.copy2(qwen3_tts_src, qwen3_tts_report_path)
                print(f"Qwen3-TTS output copied to: {qwen3_tts_report_path}")

        test_duration = _dt.datetime.now() - test_start
        ttft_value = extract_label_value(perf_block, "TTFT")
        throughput_value = extract_label_value(perf_block, "Throughput")
        timing_entries.append(
            {
                "index": test["index"],
                "name": test["name"],
                "duration": test_duration,
                "ttft": ttft_value,
                "throughput": throughput_value,
            }
        )

        print(f"Return code: {result.returncode}")
        if result.returncode != 0:
            failed_tests.append((test["index"], test["name"], result.returncode))
        if is_ult_test:
            print("ULT Output:")
            print(output)
        else:
            print("Performance:")
            print(perf_block)
            print("Generated text:")
            print(gen_block)

        if is_ult_test:
            # Special markdown format for ULT tests
            report_lines.extend(
                [
                    f"## Test {test['index']}: {test['name']}",
                    "",
                    "Environment:",
                    "```text",
                    *env_commands,
                    "```",
                    "",
                    "Command:",
                    "```text",
                    cmd_line,
                    "```",
                    "",
                    f"Return code: {result.returncode}",
                    "",
                    "Test Results:",
                    "```",
                    output.strip(),
                    "```",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    f"## Test {test['index']}: {test['name']}",
                    "",
                    "Environment:",
                    "```text",
                    *env_commands,
                    "```",
                    "",
                    "Command:",
                    "```text",
                    cmd_line,
                    "```",
                    "",
                    f"Return code: {result.returncode}",
                    "",
                    "Performance:",
                    "```text",
                    perf_block,
                    "```",
                    "",
                    "Generated text:",
                    "```text",
                    gen_block,
                    "```",
                    "",
                ]
            )

        # Embed Z-Image output in report
        if is_zimage_test and zimage_report_path and zimage_report_path.is_file():
            report_lines.extend(
                [
                    "Generated image:",
                    "",
                    f"![{zimage_output_filename}]({zimage_output_filename})",
                    "",
                ]
            )

        # Embed Wan T2V output images in report (5 images per row)
        if is_wan_t2v_test and wan_t2v_output_folder and wan_t2v_images:
            report_lines.extend(
                [
                    "Generated video frames:",
                    "",
                ]
            )
            # Create HTML table with 5 images per row
            images_per_row = 5
            for i in range(0, len(wan_t2v_images), images_per_row):
                row_images = wan_t2v_images[i : i + images_per_row]
                row_cells = " ".join(
                    f'<img src="{wan_t2v_output_folder}/{img}" alt="{img}">'
                    for img in row_images
                )
                report_lines.append(f"<p>{row_cells}</p>")
            report_lines.append("")

        # Embed Qwen3-TTS output in report
        if is_qwen3_tts_test and qwen3_tts_report_path and qwen3_tts_report_path.is_file():
            report_lines.extend(
                [
                    "Generated audio:",
                    "",
                    f"[{qwen3_tts_output_filename}]({qwen3_tts_output_filename})",
                    "",
                ]
            )

    total_duration = _dt.datetime.now() - total_start
    summary_lines: List[str] = ["## Timing Summary", ""]
    if timing_entries:
        summary_lines.extend(
            [
                "| Test | TTFT | Throughput | Duration |",
                "| --- | --- | --- | --- |",
            ]
        )
        for entry in timing_entries:
            summary_lines.append(
                f"| {entry['index']}: {entry['name']} | {entry['ttft']} | {entry['throughput']} | {format_duration(entry['duration'])} |"
            )
    else:
        summary_lines.append("No tests were executed.")
    summary_lines.extend(["", f"Total duration: {format_duration(total_duration)}", ""])
    report_lines.extend(summary_lines)

    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    print("=" * 80)
    print(f"Report saved to: {report_path}")
    print("=" * 80)
    print("Timing summary:")
    if timing_entries:
        print("| Test | TTFT | Throughput | Duration |")
        print("| --- | --- | --- | --- |")
        for entry in timing_entries:
            print(
                f"| {entry['index']}: {entry['name']} | {entry['ttft']} | {entry['throughput']} | {format_duration(entry['duration'])} |"
            )
    else:
        print("No tests were executed.")
    print(f"Total run time: {format_duration(total_duration)}")

    if failed_tests:
        print("", file=sys.stderr)
        print("FAILED TESTS:", file=sys.stderr)
        for idx, name, code in failed_tests:
            print(f"  [{idx}] {name} (exit code: {code})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
