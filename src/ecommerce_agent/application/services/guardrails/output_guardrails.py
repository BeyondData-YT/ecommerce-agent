from llm_guard import scan_output
from llm_guard.output_scanners import Toxicity
from llm_guard.output_scanners.toxicity import MatchType
import logging

class OutputGuardrail:
    def __init__(self):
        self.output_scanners = [Toxicity(match_type=MatchType.FULL)]

    def validate_output(self, prompt: str, response_text: str) -> str:
        sanitized_response_text, results_valid, results_score = scan_output(self.output_scanners, prompt, response_text, fail_fast=True)
        if any(not result for result in results_valid.values()):
            logging.error(f"Output {response_text} is not valid, scores: {results_score}")
            raise ValueError(f"Output {response_text} is not valid, scores: {results_score}")
        return sanitized_response_text