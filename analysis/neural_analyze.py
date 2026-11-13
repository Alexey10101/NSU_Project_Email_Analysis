"""
нейросетевой анализ письма по абзацам методом текстовых возмущений.
distilbert заменяет слова, qwen оценивает изменение предсказуемости текста.
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

from model_utils import analyze_path, print_error, print_json
from parser import EmailParser, clean_email_body


class PerturbationDetector:
    mask_name = "distilbert-base-uncased"
    eval_name = "Qwen/Qwen2.5-0.5B"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_name)
        self.mask_model = AutoModelForMaskedLM.from_pretrained(self.mask_name).eval()
        self.eval_tokenizer = AutoTokenizer.from_pretrained(self.eval_name)
        self.eval_model = AutoModelForCausalLM.from_pretrained(self.eval_name).eval()

    def _get_loss(self, text):
        inputs = self.eval_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if inputs["input_ids"].shape[-1] < 2:
            return None
        with self.torch.inference_mode():
            loss = float(self.eval_model(**inputs, labels=inputs["input_ids"]).loss)
        return loss if math.isfinite(loss) else None

    def perturb_text(self, words, eligible, rng):
        index = rng.choice(eligible)
        masked = list(words)
        masked[index] = self.mask_tokenizer.mask_token
        inputs = self.mask_tokenizer(" ".join(masked), return_tensors="pt", truncation=True, max_length=512)
        positions = (inputs["input_ids"][0] == self.mask_tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(positions) != 1:
            return None

        with self.torch.inference_mode():
            logits = self.mask_model(**inputs).logits[0, positions[0]]
        for token_id in logits.topk(10).indices.tolist():
            token = self.mask_tokenizer.convert_ids_to_tokens(token_id)
            if token.isalpha() and token.casefold() != words[index].casefold():
                masked[index] = token.capitalize() if words[index][0].isupper() else token
                return " ".join(masked)
        return None

    def evaluate_paragraph(self, words, n_perturbations, rng):
        eligible = [i for i, word in enumerate(words) if len(word) > 3 and word.isalpha()]
        if not eligible:
            return None
        original_loss = self._get_loss(" ".join(words))
        if original_loss is None:
            return None

        losses = []
        for _ in range(n_perturbations):
            variant = self.perturb_text(words, eligible, rng)
            loss = self._get_loss(variant) if variant else None
            if loss is not None:
                losses.append(loss)
        return sum(losses) / len(losses) - original_loss if losses else None

    def evaluate(self, text, n_perturbations=5, seed=42):
        if n_perturbations < 1:
            raise ValueError("число возмущений должно быть положительным")
        rng = random.Random(seed)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 10]
        scores = []
        for paragraph in paragraphs or [text]:
            words = paragraph.split()[:250]
            if len(words) < 5:
                continue
            value = self.evaluate_paragraph(words, n_perturbations, rng)
            if value is not None:
                scores.append((value, " ".join(words[:20])))
        if not scores:
            return dict(neural_ai_status="insufficient_text", neural_discrepancy=None)

        maximum, excerpt = max(scores, key=lambda row: row[0])
        return dict(neural_ai_status="ok", neural_discrepancy=maximum,
                    neural_mean_discrepancy=sum(value for value, _ in scores) / len(scores),
                    most_suspicious_excerpt=excerpt)


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = PerturbationDetector()
    return _detector


def analyze_email_from_file(file_path, n_perturbations=5, seed=42):
    parsed = EmailParser().parse_from_file(file_path)
    text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)
    return dict(subject=parsed.subject, **get_detector().evaluate(text, n_perturbations, seed))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path, help="письмо или папка с eml")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--perturbations", type=int, default=5)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    if args.perturbations < 1:
        cli.error("--perturbations должно быть положительным")
    try:
        result = analyze_path(args.file, analyze_email_from_file, "neural_ai_status",
                              n_perturbations=args.perturbations, seed=args.seed)
        print_json(result, args.output)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print_error(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
