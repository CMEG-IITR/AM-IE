---
name: vpp
description: "Extraction guidance for Vat Photopolymerization additive manufacturing papers."
---

# Vat Photopolymerization (VPP)

## Synonyms

- **Exposure Time**: cure time, curing time, UV curing, irradiation time, UV exposure duration
- **Light Intensity**: UV intensity, irradiance, power density
- **XY Resolution**: print resolution, XY resolution, planar resolution
- **Z Resolution**: layer resolution, Z resolution, vertical resolution

## Extraction notes

- VPP papers often report intensity in mW/cm^2 rather than W — do not convert units, report as stated.
- Layer cure time and light intensity are the two parameters most often paired together.
- Cure/curing time is very often reported as a duration alone without the word 'time' attached (e.g. '40 s' or 'a base curing time of 40 s') — check numeric durations near any mention of curing/UV/exposure even if not explicitly labeled a parameter.
- XY resolution and Z (layer) resolution are frequently reported in micrometers as a print-quality spec, separate from layer thickness — extract both if given. Watch for the parenthetical phrasing pattern '(XY around 50 µm)' or '(XYZ around 15 µm)', where the value sits inside a qualitative descriptor like 'Good resolution (...)' rather than next to the parameter name directly.
- Resin viscosity is often described qualitatively ('low viscosity resin preferred') as well as with a numeric value — capture the qualitative statement too if no number is given, flagging confidence as medium/low.
