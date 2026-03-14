"""
Example template showing the interface.
Replace the run_experiment method with actual experiment code.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from base_template import ExperimentTemplate


class PGDAttackTemplate(ExperimentTemplate):
    def run_experiment(self):
        # PLACEHOLDER — replace with actual PGD attack code
        print("This is a placeholder template.", file=sys.stderr)
        print("Replace run_experiment() with your actual code.", file=sys.stderr)

        return {
            "accuracy": 0.0,
            "answer_flip_rate": 0.0,
            "text_coherence": 0.0,
            "answer_coupling_rho": 0.0,
            "text_coupling_rho": 0.0,
            "note": "PLACEHOLDER — not real results"
        }


if __name__ == '__main__':
    PGDAttackTemplate().execute()
