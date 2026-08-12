# Demonstration videos

Three kinds of clip, all built from the same dose panel: a planning CT in grey, the
dose as a heat map over it, and the target volume and organs at risk as outlines.

Rebuild every one with `scripts/make_videos.sh`, or individually:

```bash
python -m deepdosesens.visualization.make_dose_video --case best median worst
python -m deepdosesens.visualization.make_sensitivity_video
python -m deepdosesens.visualization.make_robustness_video --scenario all
```

## 1. Dose sweep — `dose_sweep_<case>.mp4`

Two panels: the reference dose the treatment plan delivers, and the dose the model
predicts for the same slice. One frame per axial slice, sweeping **caudal to
cranial**, so the clip walks up through the head. The case's overall dose score is
printed in the subtitle.

Cases are selected on measured properties rather than by name, and the selector
prints the full ranking of all 20 test cases before it picks:

| Selector | Ranks by |
| --- | --- |
| `largest-ptv` | target volume in cc |
| `most-oar-interaction` | how many organs at risk the plan pushes above 20 Gy |
| `best-streaks` | correlation of the high-frequency structure left after subtracting a blurred copy — does the prediction reproduce the plan's radial streaks, or only its smooth gradient? Restricted to demanding cases, since the best streak correlation overall belongs to a small target with no organ at risk in play |
| `best` / `median` / `worst` | dose score |

The defaults are `most-oar-interaction largest-ptv best-streaks`. Where two selectors
land on the same standout case the second takes its next choice, so three selectors
give three different cases. A case whose target misses every organ at risk is flagged
in the output as an easy case: the dose is one blob and no competing constraint is in
play, so it shows nothing about the model.

## 2. Optic nerve sensitivity — `optic_nerve_sensitivity.mp4`

The experiment both papers rest on. Two panels per contour variant: the reference plan
re-optimised for that contour, beside the model's prediction for it. The variant's
outline is solid cyan and the original dashed in the same colour, since they are the
same structure in two states. Mean dose to the contour is printed under each panel and
the shift relative to the original contour is in the subtitle, for the plan and for the
prediction — the two quantities the papers correlate.

The view is cropped to the optic nerve region and the slice is chosen among those
where all ten variants have voxels. **Caveat:** at 128³ the nerve is only a few
voxels across, and much of the difference between variants is out of plane, so the
contours can look nearly identical in a single slice while the mean doses differ by
several Gy. The numbers carry the argument; the panels carry the dose context.

## 3. Robustness to target shape — `robustness_concave.mp4`, `robustness_multiple.mp4`

Three panels: the reference dose, the initial model's prediction, and the prediction
of the model retrained with cases of that shape. Only the model changes across
columns, so the columns isolate the effect of the extra training data. Per-panel
dose scores are printed underneath.

Two filters apply to the automatic case choice:

* cases the retrained model was trained on are skipped — detectable because their
  dose score collapses to a fraction of the initial model's;
* `DLDP_118` is skipped because its archived dose mask has empty slices, which would
  render as blank bands and read as a bug rather than as model behaviour.

The console prints every candidate with its shape, its scores and the reason for any
exclusion.

## Appearance

| Element | Choice |
| --- | --- |
| CT | grey, window level 40 / width 400 HU |
| Dose | `inferno`, 20–65 Gy by default, transparent below the lower limit |
| Dose opacity | ramps from 0.32 at the lower limit to 0.78 at the upper |
| Target volume | white, heaviest line |
| Organs at risk | one hue per organ; left and right share a hue |
| Contours | pale halo behind every line |

The dose map is perceptually uniform and runs dark-to-bright, so hot reads as high
dose. Opacity rises with dose because a flat overlay hides the CT everywhere the
dose is non-trivial — ramping it keeps the anatomy readable in the low-dose wash
while the high-dose region stays close to opaque.

Left and right instances of an organ share a hue deliberately: which side is which
is never ambiguous, since the image places them either side of the midline, and
nine hues read far better than seventeen. The organ palette is validated for
colour-vision deficiency (worst adjacent-pair CVD ΔE 9.1, normal-vision ΔE 19.6);
every structure is also named in the legend, so identity never rests on colour
alone. Contours carry a pale halo so a dark hue stays visible against dark low-dose
purple and a light one against bright high-dose yellow.

`--dose-min` and `--dose-max` set the heat map limits, `--fps` the sweep speed, and
`--crf` the encoder quality (higher is smaller). `--keep-frames` leaves the rendered
PNGs behind, which is what lets `scripts/make_videos.sh` produce the repo-sized
copies without rendering twice.
