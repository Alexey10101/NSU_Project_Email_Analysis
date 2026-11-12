"""
Анализ текста методом возмущений (DetectGPT) с оценкой ПО АБЗАЦАМ.
- Оценщик: Qwen2.5-0.5B (современный вероятностный ландшафт)
- Генератор синонимов: DistilBERT [MASK]
- Число возмущений: N = 5 (для снижения дисперсии шума)
- Логика: max_discrepancy по абзацам (ловит локальные вставки ИИ)
"""

import math
import random
import re
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from parser import EmailParser, clean_email_body

BASE_DIR = Path(__file__).resolve().parents[1]


class PerturbationDetector:
    def __init__(self, eval_model_name: str = "Qwen/Qwen2.5-0.5B"):
        print(f">>> Загрузка генератора синонимов (DistilBERT) и оценщика ({eval_model_name})...")
        
        # 1. Генератор контекстных синонимов
        self.mask_name = "distilbert-base-uncased"
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_name)
        self.mask_model = AutoModelForMaskedLM.from_pretrained(self.mask_name)
        self.mask_model.eval()

        # 2. Оценщик вероятности (Qwen 2.5)
        self.eval_name = eval_model_name
        self.eval_tokenizer = AutoTokenizer.from_pretrained(self.eval_name)
        self.eval_model = AutoModelForCausalLM.from_pretrained(
            self.eval_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.eval_model.eval()
        print(">>> Модели успешно инициализированы!\n")

    def _get_loss(self, text: str) -> float:
        """Считает функцию потерь (Loss) текста через Qwen2.5."""
        inputs = self.eval_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.eval_model(**inputs, labels=inputs["input_ids"])
        return float(outputs.loss)

    def perturb_text(self, text: str, mask_ratio: float = 0.15) -> str:
        """Подбирает контекстные синонимы через [MASK]."""
        words = text.split()
        if len(words) < 5:
            return text

        if len(words) > 250:
            words = words[:250]

        num_to_mask = max(1, int(len(words) * mask_ratio))
        eligible = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if not eligible:
            return " ".join(words)

        indices = set(random.sample(eligible, min(num_to_mask, len(eligible))))
        masked = [self.mask_tokenizer.mask_token if i in indices else w for i, w in enumerate(words)]
        masked_text = " ".join(masked)

        inputs = self.mask_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.mask_model(**inputs)

        mask_id = self.mask_tokenizer.mask_token_id
        input_ids = inputs["input_ids"][0]
        masked_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]

        for pos in masked_positions:
            logits = outputs.logits[0, pos]
            top_k = torch.topk(logits, k=3).indices
            input_ids[pos] = top_k[1].item()

        return self.mask_tokenizer.decode(input_ids, skip_special_tokens=True)

    def evaluate_paragraph(self, paragraph: str, n_perturbations: int = 5) -> float:
        """Считает разницу потерь (Discrepancy) для одного конкретного абзаца."""
        orig_loss = self._get_loss(paragraph)
        p_losses = [self._get_loss(self.perturb_text(paragraph)) for _ in range(n_perturbations)]
        avg_p_loss = sum(p_losses) / len(p_losses)
        return avg_p_loss - orig_loss

    def evaluate(self, text: str, n_perturbations: int = 5) -> Dict:
        """
        Разбивает письмо на абзацы и вычисляет метрики по каждому из них.
        Ловит точечные правки ИИ через max_discrepancy.
        """
        if not text or len(text.split()) < 5:
            return {
                "neural_discrepancy": 0.0,
                "neural_original_loss": 0.0,
                "neural_ai_score": 50.0,
                "neural_ai_status": "too_short",
                "neural_ai_reasons": ["текст слишком короткий"]
            }

        # 1. Разбиваем на абзацы по двойным переносам строк
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        
        # Фильтруем слишком короткие абзацы (приветствия, подписи)
        meaningful_paras = [p for p in raw_paragraphs if len(p.split()) >= 10]
        
        # Если все абзацы короткие — берем весь текст целиком
        if not meaningful_paras:
            meaningful_paras = [text]

        # 2. Считаем Discrepancy по каждому абзацу с N=5
        para_discrepancies = []
        for p in meaningful_paras:
            d_p = self.evaluate_paragraph(p, n_perturbations=n_perturbations)
            para_discrepancies.append((d_p, p))

        # 3. Находим самый подозрительный абзац (максимальный рост потерь при синонимах)
        para_discrepancies.sort(key=lambda x: x[0], reverse=True)
        max_d, most_suspicious_para = para_discrepancies[0]
        mean_d = sum(d for d, _ in para_discrepancies) / len(para_discrepancies)

        overall_orig_loss = self._get_loss(text[:300])

        # 4. Логистический перевод max_d в проценты
        # Базовый порог для Qwen2.5 по абзацам обычно около 0.15 - 0.25
        z = (max_d - 0.20) * 7.0
        z_clamped = max(-6.0, min(6.0, z))
        prob = 1.0 / (1.0 + math.exp(-z_clamped))
        score = round(prob * 100.0, 1)

        reasons = []
        if max_d > 0.30:
            reasons.append(f"В одном из абзацев обнаружен выраженный пик вероятности LLM (d={max_d:+.3f})")
        elif max_d < 0.10:
            reasons.append(f"Все абзацы показывают стабильное человеческое распределение (max d={max_d:+.3f})")
        else:
            reasons.append(f"Умеренные показатели кривизны (max d={max_d:+.3f})")

        return {
            "neural_discrepancy": round(max_d, 4),        # Ключевой признак: максимум по абзацам
            "neural_mean_discrepancy": round(mean_d, 4),  # Среднее по абзацам
            "neural_original_loss": round(overall_orig_loss, 4),
            "neural_ai_score": score,
            "neural_ai_status": "ok",
            "neural_ai_reasons": reasons,
            "most_suspicious_excerpt": most_suspicious_para[:120] + "..."
        }


# Глобальный объект кэша
_DETECTOR = None

def get_detector() -> PerturbationDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PerturbationDetector()
    return _DETECTOR


def analyze_email_from_file(file_path: str, n_perturbations: int = 5) -> dict:
    parsed = EmailParser().parse_from_file(file_path)
    raw = parsed.text_plain or parsed.extracted_text_from_html
    cleaned = clean_email_body(raw)
    detector = get_detector()
    features = detector.evaluate(cleaned, n_perturbations=n_perturbations)
    return {
        "subject": parsed.subject,
        "cleaned_text": cleaned,
        **features
    }