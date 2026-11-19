"""
анализирует абзацы с помощью RoBERTa-large и Qwen2.5-1.5B.
"""

import math
import random
import re
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from parser import EmailParser, clean_email_body

DEFAULT_PERTURBATIONS = 4
DEFAULT_SEED = 42

class PerturbationDetector:
    def __init__(
        self,
        eval_model_name: str = "Qwen/Qwen2.5-1.5B",
        mask_model_name: str = "roberta-large",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Загрузка {mask_model_name} и {eval_model_name} на {self.device}...", file=sys.stderr)

        self.mask_name = mask_model_name
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_name)
        self.mask_model = AutoModelForMaskedLM.from_pretrained(
            self.mask_name,
            dtype=dtype,
        ).to(self.device)
        self.mask_model.eval()

        self.eval_name = eval_model_name
        self.eval_tokenizer = AutoTokenizer.from_pretrained(self.eval_name)
        if self.eval_tokenizer.pad_token is None:
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
        self.eval_tokenizer.padding_side = "right"

        self.eval_model = AutoModelForCausalLM.from_pretrained(
            self.eval_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.eval_model.eval()
        print("модели загружены", file=sys.stderr)

    def _align_length(self, text: str, max_tokens: int = 512) -> str:
        token_ids = self.eval_tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        return self.eval_tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def _get_loss(self, text: str) -> float:
        inputs = self.eval_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        if inputs["input_ids"].shape[-1] < 2:
            return float("nan")
        with torch.no_grad():
            outputs = self.eval_model(**inputs, labels=inputs["input_ids"], use_cache=False)
        return float(outputs.loss)


    def _get_losses_batch(self, texts: List[str], batch_size: int = 4) -> List[float]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self._get_loss(texts[0])]

        losses = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = self.eval_tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.eval_model(**inputs, use_cache=False)
                shift_logits = outputs.logits[:, :-1, :].float().contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()
                shift_mask = inputs["attention_mask"][:, 1:].contiguous().float()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                token_losses = token_losses.view(shift_labels.size()) * shift_mask
                per_sample = token_losses.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)
                losses.extend(float(val) for val in per_sample.tolist())
        return losses

    def perturb_text(self, text, rng, generator, mask_ratio=0.15):
        matches = list(re.finditer(r"(?<![A-Za-z'])[A-Za-z]{4,}(?![A-Za-z'])", text))
        if len(matches) < 2:
            return text

        num_to_mask = max(1, int(len(matches) * mask_ratio))
        eligible = list(range(len(matches)))

        # Стараемся не маскировать соседние слова, чтобы сохранялся контекст с обеих сторон
        chosen_idx = []
        for idx in rng.sample(eligible, len(eligible)):
            if all(abs(idx - prev) > 1 for prev in chosen_idx):
                chosen_idx.append(idx)
                if len(chosen_idx) >= num_to_mask:
                    break
        chosen_idx.sort()

        parts = []
        last_end = 0
        for idx in chosen_idx:
            match = matches[idx]
            parts.append(text[last_end:match.start()])
            parts.append(self.mask_tokenizer.mask_token)
            last_end = match.end()
        parts.append(text[last_end:])
        masked_text = "".join(parts)

        inputs = self.mask_tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.mask_model(**inputs).logits[0]

        mask_id = self.mask_tokenizer.mask_token_id
        input_ids = inputs["input_ids"][0]
        masked_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0].tolist()

        if len(masked_positions) != len(chosen_idx):
            return text

        replacements = {}
        for idx, pos in zip(chosen_idx, masked_positions):
            orig_word = matches[idx].group(0)
            top_ids = torch.topk(logits[pos], k=20).indices.tolist()
            top_tokens = self.mask_tokenizer.convert_ids_to_tokens(top_ids)
            has_space_prefix = any(tok.startswith("Ġ") for tok in top_tokens)

            candidates = []
            candidate_logits = []
            for tok_id, tok_str in zip(top_ids, top_tokens):
                # После пробела в RoBERTa целые слова начинаются с 'Ġ', а без 'Ġ' — суффиксы/обрывки
                if has_space_prefix and not tok_str.startswith("Ġ"):
                    continue
                word = self.mask_tokenizer.decode([tok_id]).strip()
                if len(word) < 2 or not word.isalpha():
                    continue
                if word.lower() == orig_word.lower():
                    continue

                if orig_word.isupper():
                    word = word.upper()
                elif orig_word[0].isupper():
                    word = word[0].upper() + word[1:].lower()
                else:
                    word = word.lower()

                candidates.append(word)
                candidate_logits.append(logits[pos, tok_id])

            if candidates:
                probs = torch.softmax(torch.stack(candidate_logits).float(), dim=0)
                picked = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
                replacements[idx] = candidates[picked]

        out = []
        last_end = 0
        for idx in chosen_idx:
            match = matches[idx]
            out.append(text[last_end:match.start()])
            out.append(replacements.get(idx, match.group(0)))
            last_end = match.end()
        out.append(text[last_end:])
        return "".join(out)

    def evaluate_paragraph(self, paragraph, n_perturbations, rng, generator):
        aligned = self._align_length(paragraph, max_tokens=450)
        variants = [self.perturb_text(aligned, rng, generator) for _ in range(n_perturbations)]
        variants = [variant for variant in variants if variant != aligned]
        if not variants:
            return None
        original = self._get_loss(aligned)
        losses = [loss for loss in self._get_losses_batch(variants) if math.isfinite(loss)]
        if not losses or not math.isfinite(original):
            return None
        return sum(losses) / len(losses) - original

    def evaluate(self, text, n_perturbations=DEFAULT_PERTURBATIONS, seed=DEFAULT_SEED):
        if n_perturbations < 1:
            raise ValueError("число возмущений должно быть положительным")
        rng = random.Random(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, pending = [], []
        for paragraph in paragraphs:
            pending.append(paragraph)
            if sum(len(part.split()) for part in pending) >= 20:
                chunks.append("\n\n".join(pending))
                pending.clear()
        if pending:
            chunks.append("\n\n".join(pending))

        scores = []
        for paragraph in chunks:
            if len(paragraph.split()) >= 5:
                delta = self.evaluate_paragraph(paragraph, n_perturbations, rng, generator)
                if delta is not None:
                    scores.append((delta, paragraph))
        if not scores:
            return dict(neural_ai_status="insufficient_text", neural_discrepancy=None,
                        neural_ai_score=None, neural_ai_reasons=["не удалось получить замены для анализа"])

        maximum, excerpt = max(scores, key=lambda row: row[0])
        return dict(neural_ai_status="ok", neural_discrepancy=maximum, neural_ai_score=None,
                    neural_mean_discrepancy=sum(value for value, _ in scores) / len(scores),
                    neural_ai_reasons=["разница предсказуемости после замены слов"],
                    most_suspicious_excerpt=excerpt[:120])


_DETECTOR = None


def get_detector() -> PerturbationDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PerturbationDetector()
    return _DETECTOR


def analyze_email_from_file(file_path, n_perturbations=DEFAULT_PERTURBATIONS, seed=DEFAULT_SEED) -> dict:
    parsed = EmailParser().parse_from_file(file_path)
    raw = parsed.text_plain or parsed.extracted_text_from_html
    cleaned = clean_email_body(raw)
    detector = get_detector()
    features = detector.evaluate(cleaned, n_perturbations, seed)
    return {
        "subject": parsed.subject,
        "cleaned_text": cleaned,
        **features,
    }
