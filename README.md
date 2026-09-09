# 12 - BoundedReflexion - a reusable pattern

> One pattern extracted from the other eleven, ready to contribute upstream.

**What it demonstrates:** Turning working code into a reusable, framework-agnostic component

**Status:** working implementation with passing tests. Built as a learning project to understand the pattern, not as a production service.

---

## Run it right now

No API key needed - every project ships with `MODEL=fake`, a deterministic
offline responder, so you can see the whole flow work before spending anything.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python -m src.main
pytest -q
```

To use a real model, edit `.env`:

```
MODEL=gpt-4o-mini            # + OPENAI_API_KEY
MODEL=claude-3-5-haiku-latest  # + ANTHROPIC_API_KEY
MODEL=ollama/llama3.1        # free, runs locally
```

## How it works

Projects 03 and 10 both contain the same idea: keep improving an output while it is getting better, under hard bounds. This repo extracts exactly that and nothing else.

`generate` and `judge` are injected by the caller, so the component has no dependency on any provider, framework or prompt. It carries its own bounds (revision cap, plateau delta, wall clock), validates its configuration on construction, and returns an `Outcome` with every attempt, the improvement delta and why it stopped.

`as_node()` adapts it to a LangGraph-style node - state dict in, state dict out - which is the shape an upstream PR would need. The tests run with no network and no LLM.

## What "done" means here

- No imports from the other projects, no provider dependency
- Bounds are validated at construction, not discovered at runtime
- Plateau, regression, timeout and cap are each a distinct stop reason
- The best attempt is returned even when the last attempt is worse
- Constraints from the judge reach the next generation
- `as_node()` returns a state-dict-in, state-dict-out callable

Every one of those lines has a test behind it in `tests/` - `pytest -q` is the
proof, not the README.

## Layout

```
src/llm.py             provider-agnostic completion, plus offline fake mode
src/fake.py            the canned responses that make MODEL=fake work
src/logging_setup.py   structured JSON logging
src/agent.py           the pattern itself
src/main.py            CLI entrypoint
tests/                 11 tests, all passing
```

## Next steps

- Benchmark it against a fixed number of revisions and record the token saving
- Read the target framework's CONTRIBUTING guide and match their test style
- Write a docs page covering when NOT to use this - reflection is not free
- Open the PR, sign the CLA, and link it here

## Reference

https://github.com/langchain-ai/langgraph/blob/main/CONTRIBUTING.md

---

Part of a 12-project agentic AI series - [github.com/dhanashalini25](https://github.com/dhanashalini25?tab=repositories)
