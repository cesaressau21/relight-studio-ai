# Background replacement with light matching
Objective: place the separated subject on a new background photo and propose lights that match it.
Branch: feat/background-replace (from main 13a1ffc) · Delivery: ask-on-risk · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 (from the agreed suggestion, verbatim): "Cambiar el fondo e igualar la luz: con un PNG recortado (o con el recorte automático), poner un fondo nuevo y que la app proponga luces con el color y la dirección de ese fondo."
- S2 (verbatim, follow-up description): "cargar una foto de fondo, colocar al sujeto encima y que la app proponga luces con el color y la dirección del fondo nuevo [...] Después siempre puedes ajustarlas a mano."
- S3 Subject outline comes from the existing separation: PNG transparency, AI matte or fast matte. Without a separated subject the background cannot be applied and the UI says so.
- S4 Edges: semi-transparent outline pixels (hair) must not keep the old background's color (color decontamination).
- S5 Non-destructive and saved in the project; export includes the composite (opaque result).
- Assumption: the new background is shown as a photo with its own lighting (fit to cover, optional blur and brightness); scene lights relight the subject.

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S2 | inline | done | core/background.py; tests/test_background.py 13 RED→GREEN |
| T2 | S1, S3, S4 | inline | in progress | |
| T3 | S1, S3, S5 | inline | pending | |
| T4 | S1-S5 | inline | pending | |

Route rationale: one writer, sequential dependent units (T2 uses T1 types, T3 uses T2).

## Log
- L1 (2026-10-18) User: "Dale, empieza con el cambio de fondo" (after choosing suggestion 3 in the feature list).
