# Roadmap issue drafts

These files are the durable source of the 2026-07 codebase-audit roadmap. Each
file was posted verbatim as a GitHub issue; the table below maps files to issue
numbers. The **issues are the working copies** — if an issue's body is edited on
GitHub, that wins. Delete a file here when its issue is closed; delete this
directory when all 19 are done.

Phases (encoded as `phase:*` labels — this repo uses labels instead of
milestones): **A** Foundation & Hygiene → **B** Robustness & UX → **C** Video
Annotation → **D** SAM 3.

| File | Issue | Title |
|------|-------|-------|
| A1.md | [#33](https://github.com/cofade/digitalsreeni-image-annotator/issues/33) | chore: Central logging framework; migrate print() calls to logging |
| A2.md | [#34](https://github.com/cofade/digitalsreeni-image-annotator/issues/34) | fix: Eliminate silent exception swallowing; document error-handling convention |
| A3.md | [#35](https://github.com/cofade/digitalsreeni-image-annotator/issues/35) | test: ProjectController .iap save/load roundtrip + ClassController unit tests |
| A4.md | [#36](https://github.com/cofade/digitalsreeni-image-annotator/issues/36) | test: ImageController multi-dim slicing + SAMController unit tests |
| A5.md | [#37](https://github.com/cofade/digitalsreeni-image-annotator/issues/37) | test: DINOController workflow tests (mocked inference) |
| A6.md | [#38](https://github.com/cofade/digitalsreeni-image-annotator/issues/38) | build: pyproject.toml migration + dependency reconciliation |
| A7.md | [#39](https://github.com/cofade/digitalsreeni-image-annotator/issues/39) | docs: Bring README, ADRs, and intro docs current with the fork |
| A8.md | [#40](https://github.com/cofade/digitalsreeni-image-annotator/issues/40) | chore: Dev-tooling — .claude settings allowlist, senior-reviewer refresh, verify/run skill |
| B1.md | [#41](https://github.com/cofade/digitalsreeni-image-annotator/issues/41) | feat: Autosave protection before first manual save |
| B2.md | [#42](https://github.com/cofade/digitalsreeni-image-annotator/issues/42) | feat: Relative paths in .iap for project portability |
| B3.md | [#43](https://github.com/cofade/digitalsreeni-image-annotator/issues/43) | feat: Image-list folders/groups + annotation-status badges (PRD US-1 remainder) |
| B4.md | [#44](https://github.com/cofade/digitalsreeni-image-annotator/issues/44) | fix: Guard mixed pose/non-pose classes (ADR-029 known gap) |
| B5.md | [#45](https://github.com/cofade/digitalsreeni-image-annotator/issues/45) | perf: Lazy multi-dim slice loading with LRU cache |
| B6.md | [#46](https://github.com/cofade/digitalsreeni-image-annotator/issues/46) | refactor: Extract rendering/overlay + edit-gesture code from image_label.py |
| C1.md | [#47](https://github.com/cofade/digitalsreeni-image-annotator/issues/47) | feat: Video loading + frame navigation (frames as slices) |
| C2.md | [#48](https://github.com/cofade/digitalsreeni-image-annotator/issues/48) | feat: Video timeline UI + annotated-frame markers + frame export |
| D1.md | [#49](https://github.com/cofade/digitalsreeni-image-annotator/issues/49) | spike: SAM 3 availability + API verification |
| D2.md | [#50](https://github.com/cofade/digitalsreeni-image-annotator/issues/50) | feat: SAM 3 text-prompt segmentation reusing the DINO review workflow |
| D3.md | [#51](https://github.com/cofade/digitalsreeni-image-annotator/issues/51) | feat: SAM 3 video object tracking |
