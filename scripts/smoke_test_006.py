#!/usr/bin/env python3
"""Smoke test for exp_006: run 3 problems at 3 SNR levels to verify pipeline."""

import os
import sys

# Patch the main script's config for smoke test
import scripts.exp_006_llama_snr_cliff as exp006

exp006.NUM_PROBLEMS = 5  # try 5, expect 2-3 valid
exp006.SNR_LEVELS_DB = [0, 14, 25]  # low, cliff region, high
exp006.RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "smoke_006")
os.makedirs(exp006.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    exp006.main()
