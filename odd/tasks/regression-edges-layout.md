# Regression suite, silhouette edges, screen fitting
Objective: protect every feature with automated tests, then improve lighting at subject silhouettes and make the UI fit the screen.
Branch: feat/regression-edges-layout (from feat/relight-mvp) · Delivery: ask-on-risk · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 (verbatim): "crea una serie de test para probar cada funcionalidad de la aplicacion, y asi asegurar que cada cambio que se haga no se rompa alguna parte de la aplicacion"
- S2 (verbatim): "Mejorar los bordes de la silueta de una persona u objeto para la iluminacion"
- S3 (verbatim): "Las interfaces no se ajustan bien a la pantalla o se salen"
- Evidence S2: depth from DAv2 (518 px, upsampled) is blurred at silhouettes; background pixels next to hair are lit as subject (halo) and hair strands cast thick ghost shadows (scratchpad crops 2026-10-16).
- Evidence S3: Scene tab content min width 401 px > 360 px panel viewport (numeric fields clipped); window has a fixed 1600x980 size regardless of the screen.

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S1 | inline | done | 234 tests pass; new: test_engine_params, test_controller, test_ui_panels, test_ui_canvas_interaction, test_main_window, test_source_hygiene |
| T2 | S2 | inline | pending | |
| T3 | S3 | inline | pending | |

Route rationale: sequential single writer; no parallel disjoint units launched together.

## Log
- L1 (2026-10-16) User: "Bien, ahora, crea una serie de test para probar cada funcionalidad de la aplicacion, y asi asegurar que cada cambio que se haga no se rompa alguna parte de la aplicacion.. voy avisando estas 2 cosas que quiero que tomes en cuentas: 1.- Mejorar los bordes de la silueta de una persona u objeto para la iluminacion 2.- Las interfaces no se ajustan bien a la pantalla o se salen"
- L2 Screen: DELL AW2523HF 1920x1080, available 1920x1032, DPR 1.0.
- L3 Decision S2: subject matting with hustvl/vitmatte-small-composition-1k (Apache-2.0, native Transformers, no remote code) using a trimap derived from depth; BiRefNet rejected (needs trust_remote_code). Prototype: ~110 ms at 1.5 MP after warm-up, resolves individual hair strands.
- L4 T1 found real bugs: main_window.py strings were double-encoded (mojibake from an earlier cp1252 conversion; also in the built exe) -> repaired + hygiene test; opening a photo marked the document dirty (first depth estimate) -> fixed + test. QSettings moved to an INI file so tests can isolate it.
