"""Reasoning must not eat the answer's token budget.

Gemma streams its reasoning before its answer out of a single max_tokens cap.
When that cap was the answer budget, a long reasoning trace used it up and the
answer stopped mid-sentence: a 700-token summary ended after ~90 words.
"""

from hadith_sidecar.llm import REASONING_TOKENS, LlamaCppProvider


def _thinking_provider(completion):
    """A LlamaCppProvider without loading llama.cpp or any weights."""
    provider = object.__new__(LlamaCppProvider)
    provider.thinking = True
    provider._completion = completion
    return provider


def _capped_completion(seen: dict, *, reasoning_tokens: int = 640,
                       answer_words: int = 300):
    """A thinking model whose reasoning draws on the same cap as its answer.

    It reasons for `reasoning_tokens`, then writes words until the cap runs
    out, so an answer cut short means the budget was too small.
    """

    def completion(system, user, *, max_tokens, temperature, stream):
        seen["max_tokens"] = max_tokens
        spent = min(reasoning_tokens, max_tokens)
        remaining = max(0, max_tokens - spent)
        words = min(answer_words, remaining)
        reasoning = "<|channel>thought " + " ".join(["weighing"] * spent)
        answer = " ".join(["stated"] * words)
        return {"choices": [{"text": reasoning + "<channel|>" + answer}]}

    return completion


def test_answer_budget_is_untouched_without_reasoning():
    provider = object.__new__(LlamaCppProvider)
    provider.thinking = False
    assert provider._generation_budget(700, provider.thinking) == 700
    assert provider._generation_budget(512, provider.thinking) == 512


def test_reasoning_gets_its_own_allowance_on_top_of_the_answer():
    assert LlamaCppProvider._generation_budget(700, True) == 700 + REASONING_TOKENS
    assert LlamaCppProvider._generation_budget(1200, True) == 1200 + REASONING_TOKENS


def test_thinking_generation_still_produces_the_full_answer():
    seen: dict = {}
    provider = _thinking_provider(_capped_completion(seen))
    answer = provider.generate("system", "user", max_tokens=700)
    # The answer got the caller's full 700-token budget, in addition to reasoning.
    assert seen["max_tokens"] == 700 + REASONING_TOKENS
    assert len(answer.split()) == 300
    assert "weighing" not in answer


def _provider_that_returns(text: str, finish_reason: str | None):
    def completion(system, user, *, max_tokens, temperature, stream):
        return {"choices": [{"text": text, "finish_reason": finish_reason}]}

    provider = object.__new__(LlamaCppProvider)
    provider.thinking = False
    provider._completion = completion
    return provider


def test_a_cut_off_generation_reports_length():
    provider = _provider_that_returns("A summary that stops mid", "length")
    assert provider.generate("system", "user", max_tokens=64) == "A summary that stops mid"
    assert provider.last_finish_reason == "length"


def test_a_finished_generation_reports_stop():
    provider = _provider_that_returns("A complete answer.", "stop")
    provider.generate("system", "user")
    assert provider.last_finish_reason == "stop"


def test_streaming_keeps_the_finish_reason_from_the_last_chunk():
    def completion(system, user, *, max_tokens, temperature, stream):
        yield {"choices": [{"delta": {"content": "A partia"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {"content": "l answer"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "length"}]}

    provider = object.__new__(LlamaCppProvider)
    provider.thinking = False
    provider._completion = completion
    seen: list[str] = []
    assert provider.generate("system", "user", on_token=lambda t, p: seen.append(t)) == "A partial answer"
    assert seen == ["A partia", "l answer"]
    assert provider.last_finish_reason == "length"
