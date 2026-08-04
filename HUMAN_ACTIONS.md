# Human actions log

All identity/credential-bound actions completed as of v0.1.0 release. This file
is now a historical record; future actions are tracked in `ROADMAP.md`.

## 1. Platform registration — ALL DONE

- [x] GitHub: org `spektran` created; repo pushed; Pages enabled (docs live at https://spektran.github.io/spektran/); CI green
- [x] PyPI: `spektran` v0.1.0 published via Trusted Publisher (OIDC, no token). Live at https://pypi.org/project/spektran/
- [x] Hugging Face: org `spektran` created; dataset transferred to `spektran/spektran-ch4-v0`. Live at https://huggingface.co/datasets/spektran/spektran-ch4-v0
- [x] Zenodo: GitHub repo linked; DOI `10.5281/zenodo.21790394` auto-minted on v0.1.0 release. Written into `CITATION.cff`.

## 2. Token configuration — ALL DONE

- [x] `HF_TOKEN`: push executed 2026-08-04 (4 splits, 7500 records, CC BY 4.0 card)
- [x] GitHub Actions: PyPI uses Trusted Publishers (OIDC), no secrets needed. `pypi` environment created on GitHub.

## 3. Release — DONE

- [x] `push_to_hf.py`: 4 splits uploaded
- [x] All links updated to `spektran/` org branding
- [x] Tag `v0.1.0` cut; Zenodo DOI minted; DOI written into `CITATION.cff`

## 4. Outward communication (Phase 3, future)

- [ ] Data-descriptor paper submission (JOSS/SoftwareX; agent drafts, human signs/submits)
- [ ] Emails inviting groups to contribute validation data
- [ ] Trademark registration for "SPEKTRAN" and "SPEKTRAN Verified"
