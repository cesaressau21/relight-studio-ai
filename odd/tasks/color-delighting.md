# Color-aware delighting
Objective: remove the color of the original light from the subject, not only its brightness, so new lights and backgrounds are not tinted by the old light.
Branch: feat/color-delighting (from main e0ad75e) · Delivery: ask-on-risk · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 (verbatim, agreed next step): "corregir el **color de la luz original**. Hoy la app solo le quita brillo al sujeto, no color, así que en tu foto el tono verde azulado sigue sobre la piel aunque pongas un atardecer detrás."
- S2 The original light color can vary across the photo (sample 2: teal on the left, orange on the right); the estimate is a smooth color map, not a single color.
- S3 With "Light color" at 0 (default), "keep original light" at 1 and no lights, the output still equals the photo exactly (existing invariant). With "Light color" on, the kept original light keeps its brightness but loses its color.
- S4 User control: a strength slider (0 = current behavior) and an optional neutral reference picked on the canvas; saved in projects, undoable.
- S5 Matching lighting to a background removes the old light color.
- Assumption: the subject's average color is kept unless the user picks a neutral reference (object color and light color cannot be told apart from one photo).

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S2, S3 | inline | done | scene.light_tint (normal fit + slow spatial field, soft limit) + engine original_light_color; tests/test_light_color.py 10 RED→GREEN; suite 399 |
| T2 | S4, S5 | inline | done | Scene → Light color slider, Pick neutral/Clear, canvas `neutral` tool, match lighting sets light_color 1, presets keep it; tests/test_light_color_editor.py 8 RED→GREEN; suite 408 |
| T3 | S1-S5 | inline | done | channel-starved fill, fine pass along the light axis, subject-only map on new backgrounds, lazy estimate (+0 ms when unused, +85 ms on first use at 1500x1025); tests 11 → 12; suite 410; native capture of sample 2 on the sunset: teal spot and red neck cast gone, lips kept |

Route rationale: one writer, sequential dependent units (T2 needs T1's engine terms).

## Log
- L1 (2026-10-19) User: "Dale, empecemos por eso" (after the proposal quoted in S1).
- L2 Prototype on sample 2: a purely spatial blur left teal shadows (bright orange rim dominated the average); fitting log-chromaticity to the normal separates the two colored lights. Saturated tints soft-limited (x2.5) to avoid a green fringe on the neck.
- L3 Decision: presets keep light_color and neutral_point (they describe the photo, not the look).
- L4 Native check: teal remained after T2. Cause 1: the kept 25% original light re-added its teal; S3 rewritten so the kept light is neutral when Light color is on. Cause 2: under the teal spot skin red is ~10% of normal (information lost); raising the limit added fringes, so starved pixels borrow the surrounding albedo color, keeping their brightness. Edge fringe on the new background fixed by a subject-only map built from decontaminated colors.
- L5 Neutral point is a user reference: picking the peach scarf as 'neutral' turns the subject cool, as expected; tooltip asks for something white or gray.
