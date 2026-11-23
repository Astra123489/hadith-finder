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
