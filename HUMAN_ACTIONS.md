# Pending human actions

Per plan §9, these are the only steps requiring a human (identity/credential
bound). Everything else is automated. Ordered; none blocks local development.

## 1. Platform registration (do soon — G1 name availability is perishable)

- [x] GitHub: DONE — org `spektran` created; repo pushed; Pages enabled (docs live at https://spektran.github.io/spektran/); CI green  
      (was: create org `spektran`, repo `spektran/spektran`, then
      `git remote add origin ... && git push -u origin main` from this repo.
      Enable GitHub Pages (Actions build) for the docs site.
- [ ] PyPI: register the `spektran` package name (first `twine upload` of
      an sdist/wheel built with `python -m build`, or PEP 694 reservation).
- [x] Hugging Face: token configured (user `Deepnight`); official splits + card live at https://huggingface.co/datasets/Deepnight/spektran-ch4-v0  
      Still open: create HF **org** `spektran` (web UI only) and transfer the dataset for canonical branding.
- [ ] Zenodo: link the GitHub repo (Zenodo-GitHub integration) so v1.0
      release auto-mints a DOI; `.zenodo.json` metadata is already in place.

## 2. Token configuration

- [x] `HF_TOKEN`: DONE — push executed 2026-08-04 (pushes the official v0 splits;
      run after generating data locally — see docs/quickstart.md).
- [ ] GitHub Actions secrets: none required for CI as written (public,
      hermetic); PyPI publishing can use Trusted Publishers when set up.

## 3. After registration

- [x] `push_to_hf.py`: DONE — 4 splits (7500 records) uploaded with CC BY 4.0 card.
- [ ] Update README badges/links if the final org name differs.
- [ ] Cut tag `v0.1.0` -> triggers Zenodo DOI; put the DOI into CITATION.cff.

## 4. Outward communication (Phase 3, drafts will be prepared by the agent)

- [ ] Data-descriptor paper submission (agent drafts; human signs/submits).
- [ ] Emails inviting groups to contribute validation data (agent drafts;
      human reviews and sends).
