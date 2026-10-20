# Color-aware delighting
Objective: remove the color of the original light from the subject, not only its brightness, so new lights and backgrounds are not tinted by the old light.
Branch: feat/color-delighting (from main e0ad75e) · Delivery: ask-on-risk · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 (verbatim, agreed next step): "corregir el **color de la luz original**. Hoy la app solo le quita brillo al sujeto, no color, así que en tu foto el tono verde azulado sigue sobre la piel aunque pongas un atardecer detrás."
- S2 The original light color can vary across the photo (sample 2: teal on the left, orange on the right); the estimate is a smooth color map, not a single color.
- S3 With "keep original light" at 1 and no lights, the output still equals the photo exactly (existing invariant).
- S4 User control: a strength slider (0 = current behavior) and an optional neutral reference picked on the canvas; saved in projects, undoable.
- S5 Matching lighting to a background removes the old light color.
- Assumption: the subject's average color is kept unless the user picks a neutral reference (object color and light color cannot be told apart from one photo).

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S2, S3 | inline | done | scene.light_tint (normal fit + slow spatial field, soft limit) + engine original_light_color; tests/test_light_color.py 10 RED→GREEN; suite 399 |
| T2 | S4, S5 | inline | pending | |
| T3 | S1-S5 | inline | pending | |

Route rationale: one writer, sequential dependent units (T2 needs T1's engine terms).

## Log
- L1 (2026-10-19) User: "Dale, empecemos por eso" (after the proposal quoted in S1).
- L2 Prototype on sample 2: a purely spatial blur left teal shadows (bright orange rim dominated the average); fitting log-chromaticity to the normal separates the two colored lights. Saturated tints soft-limited (x2.5) to avoid a green fringe on the neck.
