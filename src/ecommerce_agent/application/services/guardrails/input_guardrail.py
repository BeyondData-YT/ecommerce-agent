from llm_guard import scan_prompt
from llm_guard.input_scanners import Toxicity, BanCompetitors
from llm_guard.input_scanners.toxicity import MatchType

import logging

class InputGuardrail:
    def __init__(self):
        self.competitors = ["Nike", "Shein", "Exito", "Dafiti", "Koaj", "Patprimo", "Arturo Calle", "H&M"]
        self.input_scanners = [Toxicity(match_type=MatchType.FULL), BanCompetitors(competitors=self.competitors)]

    def validate_input(self, prompt: str) -> str:
        sanitized_prompt, results_valid, results_score = scan_prompt(self.input_scanners, prompt, fail_fast=True)
        if any(not result for result in results_valid.values()):
            logging.error(f"Prompt {prompt} is not valid, scores: {results_score}")
            raise ValueError(f"Prompt {prompt} is not valid, scores: {results_score}")
        return sanitized_prompt
