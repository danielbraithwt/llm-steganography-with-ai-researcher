#!/usr/bin/env python3
"""Smoke test for exp_035: run 3 problems with 2 conditions."""
import sys
sys.path.insert(0, '/workspace/llm-steganography-with-ai-researcher')
import scripts.exp_035_phi_magnitude_dose as mod

mod.NUM_PROBLEMS = 8  # try 8 to get ~3 valid
mod.CONDITIONS = [
    ("mag_k_10", "k", 1.0),
    ("mag_v_10", "v", 1.0),
]
mod.main()
