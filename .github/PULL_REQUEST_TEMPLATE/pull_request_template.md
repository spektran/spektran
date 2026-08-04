## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Schema change (triggers G2 re-check)
- [ ] Physics formula (requires DOI citation)
- [ ] Documentation
- [ ] Benchmark / baseline
- [ ] Other: ___

## Checklist

- [ ] My code follows the project's conventions (units in field names, explicit seed/rng, DOI citations)
- [ ] I have run `pytest -m "not hitran_online"` and all tests pass
- [ ] I have run `ruff check src tests` with no errors
- [ ] If this is a schema change, I updated `schema/CHANGELOG.md` and gate G2 still passes
- [ ] Gate scripts (`gates/`) are NOT modified in this PR (unless this is an independent threshold-change PR)

## Contributor License Agreement

By submitting this pull request, I confirm that:

- [ ] **Code contributions**: I agree to license my contribution under the [Apache License 2.0](../LICENSE), and I have the right to do so.
- [ ] **Data contributions** (if applicable): I agree to license submitted data under [CC BY 4.0](../LICENSE-DATA), and I have the right to do so.

I understand that my contribution will be publicly available under the terms above and that I am granting SPEKTRAN maintainers a perpetual, worldwide, non-exclusive, royalty-free license to use, reproduce, modify, and distribute my contribution as part of this project.

## Test plan

<!-- How can reviewers verify this change? -->
