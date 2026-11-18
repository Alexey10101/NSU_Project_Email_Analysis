"""
Анализ текста методом возмущений (DetectGPT) с оценкой ПО АБЗАЦАМ.
- Оценщик: Qwen2.5-1.5B
- Генератор синонимов: RoBERTa-large [MASK] (с точечной заменой без порчи регистра/пунктуации)
- Число возмущений по умолчанию: N = 4
- Логика: max_discrepancy по абзацам (ловит локальные вставки ИИ)
"""

import math
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from parser import EmailParser, clean_email_body

BASE_DIR = Path(__file__).resolve().parents[1]


class PerturbationDetector:
    def __init__(
        self,
        eval_model_name: str = "Qwen/Qwen2.5-1.5B",
        mask_model_name: str = "roberta-large",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f">>> Загрузка {mask_model_name} и {eval_model_name} на {self.device}...")

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
        print(">>> Модели успешно инициализированы!\n")

    def _align_length(self, text: str, max_tokens: int = 512) -> str:
        """Обрезает слишком длинный абзац один раз, чтобы 4А и 4Б видели один и тот же текст."""
        token_ids = self.eval_tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        return self.eval_tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def _get_loss(self, text: str) -> float:
        """Считает Loss одного текста через Qwen2.5."""
        inputs = self.eval_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
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
                shift_logits = outputs.logits[:, :-1, :].contiguous()
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

    def perturb_text(self, text: str, mask_ratio: float = 0.15) -> str:
        """
        Подбирает контекстные замены через RoBERTa-large и точечно вставляет их
        в исходную строку, полностью сохраняя регистр, пробелы и пунктуацию.
        """
        matches = list(re.finditer(r"(?<![A-Za-z'])[A-Za-z]{4,}(?![A-Za-z'])", text))
        if len(matches) < 2:
            return text

        num_to_mask = max(1, int(len(matches) * mask_ratio))
        eligible = list(range(len(matches)))

        # Стараемся не маскировать соседние слова, чтобы сохранялся контекст с обеих сторон
        chosen_idx = []
        for idx in random.sample(eligible, len(eligible)):
            if all(abs(idx - prev) > 1 for prev in chosen_idx):
                chosen_idx.append(idx)
                if len(chosen_idx) >= num_to_mask:
                    break
        if not chosen_idx:
            chosen_idx = random.sample(eligible, min(num_to_mask, len(eligible)))
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
                picked = int(torch.multinomial(probs, num_samples=1).item())
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

    def evaluate_paragraph(self, paragraph: str, n_perturbations: int = 4) -> float:
        """Считает разницу потерь (Discrepancy) для одного конкретного абзаца."""
        aligned = self._align_length(paragraph, max_tokens=450)
        orig_loss = self._get_loss(aligned)
        perturbed_texts = [self.perturb_text(aligned) for _ in range(n_perturbations)]
        p_losses = self._get_losses_batch(perturbed_texts)
        avg_p_loss = sum(p_losses) / len(p_losses)
        return avg_p_loss - orig_loss

    def evaluate(self, text: str, n_perturbations: int = 4) -> Dict:
        """
        Разбивает письмо на абзацы и вычисляет метрики по каждому из них.
        Ловит точечные правки ИИ через max_discrepancy.
        """
        if not text or len(text.split()) < 5:
            return {
                "neural_discrepancy": 0.0,
                "neural_mean_discrepancy": 0.0,
                "neural_original_loss": 0.0,
                "neural_ai_score": 50.0,
                "neural_ai_status": "too_short",
                "neural_ai_reasons": ["текст слишком короткий"],
                "most_suspicious_excerpt": "",
            }

        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        meaningful_paras = [p for p in raw_paragraphs if len(p.split()) >= 20]
        if not meaningful_paras:
            meaningful_paras = [text]
        buffer = ""
        for p in raw_paragraphs:
            buffer = f"{buffer}\n\n{p}".strip() if buffer else p
            if len(buffer.split()) >= 40:
                meaningful_paras.append(buffer)
                buffer = ""
        if buffer:
            if meaningful_paras:
                meaningful_paras[-1] = f"{meaningful_paras[-1]}\n\n{buffer}"
            else:
                meaningful_paras = [buffer]
        if not meaningful_paras:
            meaningful_paras = [text]

        para_discrepancies = []
        for p in meaningful_paras:
            d_p = self.evaluate_paragraph(p, n_perturbations=n_perturbations)
            para_discrepancies.append((d_p, p))

        para_discrepancies.sort(key=lambda x: x[0], reverse=True)
        max_d, most_suspicious_para = para_discrepancies[0]
        mean_d = sum(d for d, _ in para_discrepancies) / len(para_discrepancies)

        overall_orig_loss = self._get_loss(self._align_length(text, max_tokens=512))

        # Мягкий логистический перевод без преждевременного обрезания на 99.8%
        z = (max_d - 0.08) * 6.0
        z_clamped = max(-15.0, min(15.0, z))
        prob = 1.0 / (1.0 + math.exp(-z_clamped))
        score = round(prob * 100.0, 2)

        reasons = []
        if max_d > 0.15:
            reasons.append(f"В одном из абзацев обнаружен выраженный пик вероятности LLM (d={max_d:+.3f})")
        elif max_d < 0.02:
            reasons.append(f"Все абзацы показывают стабильное человеческое распределение (max d={max_d:+.3f})")
        else:
            reasons.append(f"Умеренные показатели кривизны (max d={max_d:+.3f})")

        return {
            "neural_discrepancy": round(max_d, 4),
            "neural_mean_discrepancy": round(mean_d, 4),
            "neural_original_loss": round(overall_orig_loss, 4),
            "neural_ai_score": score,
            "neural_ai_status": "ok",
            "neural_ai_reasons": reasons,
            "most_suspicious_excerpt": most_suspicious_para[:120] + "...",
        }


_DETECTOR = None


def get_detector() -> PerturbationDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PerturbationDetector()
    return _DETECTOR


def analyze_email_from_file(file_path: str, n_perturbations: int = 4) -> dict:
    parsed = EmailParser().parse_from_file(file_path)
    raw = parsed.text_plain or parsed.extracted_text_from_html
    cleaned = clean_email_body(raw)
    detector = get_detector()
    features = detector.evaluate(cleaned, n_perturbations=n_perturbations)
    return {
        "subject": parsed.subject,
        "cleaned_text": cleaned,
        **features,
    }
