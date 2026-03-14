#!/usr/bin/env python3
"""Smoke test for exp_015: 6 problems, 2 scales, verify pipeline works."""
import sys
sys.path.insert(0, '.')

import scripts.exp_015_qwen_instruct_noise_scale as exp
exp.NUM_PROBLEMS = 6
exp.NOISE_SCALES = [0.01, 1.0]
exp.NOISE_FRACTIONS = [0.03]
exp.RESULTS_DIR = "/workspace/llm-steganography-with-ai-researcher/results/smoke_015"

import os
os.makedirs(exp.RESULTS_DIR, exist_ok=True)

exp.main()
