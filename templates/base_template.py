"""
Base class for experiment templates.
All templates should follow this interface.
"""
import argparse
import json
import sys
import os
import signal
import time


class ExperimentTemplate:
    """
    Base class. Subclass this and implement run_experiment().

    Interface contract:
    - Accepts --config (JSON string) and --output-dir and --timeout
    - Prints JSON results to stdout on success
    - Saves artifacts (plots, data) to output-dir
    - Exits 0 on success, 1 on failure, 2 on timeout
    """

    def __init__(self):
        self.args = self.parse_args()
        self.config = json.loads(self.args.config)
        self.output_dir = self.args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.args.timeout:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.args.timeout)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True,
                            help='JSON configuration string')
        parser.add_argument('--output-dir', type=str, required=True,
                            help='Directory for output artifacts')
        parser.add_argument('--timeout', type=int, default=None,
                            help='Wall-clock timeout in seconds')
        return parser.parse_args()

    def _timeout_handler(self, signum, frame):
        self.on_timeout()
        sys.exit(2)

    def on_timeout(self):
        """Override to save partial results on timeout."""
        print(json.dumps({"error": "timeout", "partial": True}))

    def run_experiment(self):
        """Override this. Return a dict of results."""
        raise NotImplementedError

    def execute(self):
        """Main entry point. Call this from if __name__ == '__main__'."""
        try:
            results = self.run_experiment()
            results["status"] = "success"
            results["config"] = self.config
            results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            print(json.dumps(results, indent=2))
            sys.exit(0)
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "config": self.config,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
            }
            print(json.dumps(error_result, indent=2))
            sys.exit(1)
