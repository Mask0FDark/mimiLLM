"""Held-out generation checks for dialogue behavior and short-term memory."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .generation import answer_question
from .transformer import DecoderTransformer


_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[\wЁёА-Яа-я]+", re.UNICODE)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


@dataclass(frozen=True)
class DialogueCheckResult:
    name: str
    passed: bool
    responses: tuple[str, ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["responses"] = list(self.responses)
        values["failures"] = list(self.failures)
        return values


@dataclass(frozen=True)
class DialogueEvaluationReport:
    cases: int
    passed: int
    pass_rate: float
    min_pass_rate: float
    ok: bool
    results: tuple[DialogueCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "min_pass_rate": self.min_pass_rate,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def _dialogue_question(history: list[tuple[str, str]], question: str) -> str:
    return "".join(
        f"{old_question}\nОтвет: {old_answer}\n\nВопрос: "
        for old_question, old_answer in history
    ) + question


def _expectation_failures(expectation: dict[str, Any], response: str) -> list[str]:
    allowed = {
        "role", "exact", "contains_all", "contains_any", "forbidden",
        "min_characters", "min_cyrillic_characters",
        "max_repeated_word_fraction",
    }
    unknown = sorted(set(expectation) - allowed)
    if unknown:
        raise ValueError(f"assistant expectation has unknown fields: {unknown}")
    normalized = _normalize(response)
    failures: list[str] = []
    exact = expectation.get("exact")
    if exact is not None:
        if not isinstance(exact, str):
            raise TypeError("assistant exact expectation must be a string")
        if normalized != _normalize(exact):
            failures.append(f"expected exact response: {exact!r}")
    contains_all = expectation.get("contains_all", [])
    contains_any = expectation.get("contains_any", [])
    forbidden = expectation.get("forbidden", [])
    for name, values in (
        ("contains_all", contains_all),
        ("contains_any", contains_any),
        ("forbidden", forbidden),
    ):
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise TypeError(f"assistant {name} expectation must be a string list")
    missing = [value for value in contains_all if _normalize(value) not in normalized]
    if missing:
        failures.append(f"missing required text: {missing}")
    if contains_any and not any(_normalize(value) in normalized for value in contains_any):
        failures.append(f"none of the expected alternatives occurred: {contains_any}")
    present = [value for value in forbidden if _normalize(value) in normalized]
    if present:
        failures.append(f"forbidden text occurred: {present}")
    if "\ufffd" in response:
        failures.append("response contains an invalid UTF-8 replacement character")
    if not response.strip():
        failures.append("response is empty")
    min_characters = expectation.get("min_characters")
    min_cyrillic = expectation.get("min_cyrillic_characters")
    max_repeated = expectation.get("max_repeated_word_fraction")
    for name, value in (
        ("min_characters", min_characters),
        ("min_cyrillic_characters", min_cyrillic),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise TypeError(f"assistant {name} expectation must be a non-negative integer")
    if max_repeated is not None and (
        not isinstance(max_repeated, (int, float))
        or isinstance(max_repeated, bool)
        or not math.isfinite(max_repeated)
        or not 0.0 <= max_repeated <= 1.0
    ):
        raise TypeError(
            "assistant max_repeated_word_fraction expectation must be between 0 and 1"
        )
    if min_characters is not None and len(response.strip()) < min_characters:
        failures.append(
            f"response has fewer than {min_characters} non-padding characters"
        )
    cyrillic_characters = len(_CYRILLIC.findall(response))
    if min_cyrillic is not None and cyrillic_characters < min_cyrillic:
        failures.append(
            f"response has {cyrillic_characters} Cyrillic characters; "
            f"required at least {min_cyrillic}"
        )
    if max_repeated is not None:
        words = [_normalize(word) for word in _WORD.findall(response)]
        if not words:
            failures.append("response contains no words for repetition check")
        else:
            largest_count = max(words.count(word) for word in set(words))
            repeated_fraction = largest_count / len(words)
            if repeated_fraction > max_repeated:
                failures.append(
                    f"most frequent word occupies {repeated_fraction:.1%} of the "
                    f"response; allowed at most {max_repeated:.1%}"
                )
    return failures


def evaluate_dialogues(
    model: DecoderTransformer,
    suite_path: str | Path,
) -> DialogueEvaluationReport:
    """Runs deterministic held-out single- and multi-turn dialogue checks."""
    path = Path(suite_path)
    try:
        suite = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dialogue evaluation JSON: {exc}") from exc
    if not isinstance(suite, dict):
        raise ValueError("dialogue evaluation root must be an object")
    unknown = sorted(
        set(suite) - {"version", "min_pass_rate", "generation", "cases"}
    )
    if unknown:
        raise ValueError(f"dialogue evaluation has unknown fields: {unknown}")
    if suite.get("version") != 1:
        raise ValueError("dialogue evaluation version must be 1")
    minimum = suite.get("min_pass_rate", 1.0)
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or not 0.0 <= minimum <= 1.0
    ):
        raise ValueError("min_pass_rate must be between 0 and 1")
    generation = suite.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation settings must be an object")
    unknown_generation = sorted(
        set(generation) - {"max_new_tokens", "temperature", "top_k"}
    )
    if unknown_generation:
        raise ValueError(f"generation has unknown fields: {unknown_generation}")
    settings = {
        "max_new_tokens": generation.get("max_new_tokens", 64),
        "temperature": generation.get("temperature", 0.0),
        "top_k": generation.get("top_k", 1),
    }
    if (
        not isinstance(settings["max_new_tokens"], int)
        or isinstance(settings["max_new_tokens"], bool)
        or settings["max_new_tokens"] <= 0
    ):
        raise ValueError("generation.max_new_tokens must be a positive integer")
    if (
        not isinstance(settings["temperature"], (int, float))
        or isinstance(settings["temperature"], bool)
        or not math.isfinite(settings["temperature"])
        or settings["temperature"] < 0
    ):
        raise ValueError("generation.temperature must be a finite non-negative number")
    if (
        not isinstance(settings["top_k"], int)
        or isinstance(settings["top_k"], bool)
        or settings["top_k"] < 0
    ):
        raise ValueError("generation.top_k must be a non-negative integer")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dialogue evaluation cases must be a non-empty list")

    results: list[DialogueCheckResult] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) - {"name", "messages"}:
            raise ValueError(f"dialogue case {index} has an invalid schema")
        name = case.get("name", f"case-{index + 1}")
        messages = case.get("messages")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"dialogue case {index} needs a name")
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            raise ValueError(f"dialogue case {name} needs complete user/assistant pairs")
        history: list[tuple[str, str]] = []
        responses: list[str] = []
        failures: list[str] = []
        for turn in range(0, len(messages), 2):
            user = messages[turn]
            assistant = messages[turn + 1]
            if (
                not isinstance(user, dict)
                or user.get("role") != "user"
                or not isinstance(user.get("content"), str)
                or not user["content"].strip()
                or not isinstance(assistant, dict)
                or assistant.get("role") != "assistant"
            ):
                raise ValueError(f"dialogue case {name} has invalid role order")
            question = user["content"].strip()
            response = answer_question(
                model,
                model.tokenizer,
                _dialogue_question(history, question),
                **settings,
            )
            responses.append(response)
            turn_failures = _expectation_failures(assistant, response)
            failures.extend(
                f"turn {turn // 2 + 1}: {failure}" for failure in turn_failures
            )
            history.append((question, response))
        results.append(
            DialogueCheckResult(
                name=name,
                passed=not failures,
                responses=tuple(responses),
                failures=tuple(failures),
            )
        )
    passed = sum(result.passed for result in results)
    pass_rate = passed / len(results)
    return DialogueEvaluationReport(
        cases=len(results),
        passed=passed,
        pass_rate=pass_rate,
        min_pass_rate=float(minimum),
        ok=pass_rate >= float(minimum),
        results=tuple(results),
    )


def save_dialogue_evaluation(
    report: DialogueEvaluationReport, path: str | Path,
) -> Path:
    """Atomically stores generated responses and every failed expectation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination
