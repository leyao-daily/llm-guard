# Contributing

Thanks for looking. This project has an unusually small surface area on purpose,
so contributing is mostly about keeping it that way.

## The one hard rule

**No third-party runtime dependencies.** The package must import from the standard
library only. This is the project's main differentiator — it is what lets a
security team audit the whole thing in an afternoon, and it is a direct response
to supply-chain compromises in comparable tools.

If you believe a dependency is genuinely necessary, open an issue first and make
the case. The default answer is no.

Dev-only tooling is fine (though currently there is none: the tests use
`unittest`).

## Getting set up

```bash
git clone <your-fork>
cd llm-guard

# no install step, no venv needed
python3 -m llmguard seed --reset --compare-days 30
python3 -m llmguard report
python3 -m unittest discover -s tests -v
```

Requires Python 3.9+.

## Before opening a pull request

1. **All tests pass**: `python3 -m unittest discover -s tests`
2. **No new imports outside the standard library.** Quick check:
   ```bash
   grep -rhoE "^(import|from) [a-zA-Z_][a-zA-Z0-9_.]*" llmguard/ \
     | sort -u
   ```
3. **You added a test** for a bug fix, and the test fails without your fix. The
   suite is the only thing standing between this code and silently wrong billing.
4. **Docstrings explain *why*.** Comments that restate the code are noise; comments
   that record a trade-off or a past failure are the most valuable thing in the
   repository. See `docs/ARCHITECTURE.md` for the standard.

## Code conventions

- Type hints on public functions.
- `from __future__ import annotations` at the top of every module.
- British or American spelling is fine, but be consistent within a file.
- Prefer explicit `Optional[X]` over implicit `None` semantics.
- Errors on the accounting path must never break the proxied request. Wrap them
  and record the failure; a customer's traffic matters more than our metrics.

## Adding a model price

One entry in `llmguard/pricing.py`, plus update `VERIFIED_AT` if you re-checked the
whole table. Include the source in the commit message. See
`docs/PRICING_SOURCES.md` — stale prices are the single most dangerous kind of bug
in this project, because every downstream number stays plausible while being wrong.

Note that cache multipliers are **per-model** and are not a constant. Do not
"simplify" them into a shared value.

## Adding a detector

Detectors live in `llmguard/detectors.py` and must:

- only ever **report**, never block traffic,
- carry an `evidence` dict with the numbers that produced the finding, so a user
  can verify the claim rather than trust it,
- ship with a `remedy()` that says what to actually do,
- have tests covering both the true positive and the near-miss negative.

## Reporting a security issue

Do not open a public issue. See `SECURITY.md`.
