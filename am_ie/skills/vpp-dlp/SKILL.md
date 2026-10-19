---
name: vpp-dlp
description: "Extraction guidance for Digital Light Processing additive manufacturing papers."
---

# Digital Light Processing (VPP-DLP)

## Taxonomy sections

- Process Parameters — DLP
- Feedstock Properties — Resin
- Feedstock Properties — Ceramic-Loaded Resin
- Final Shape / Part Geometry

## Synonyms

- **Exposure Time**: projection time, pixel exposure time
- **Ceramic Solid Loading (vol%)**: ceramic content, ceramic content by volume, volume fraction of ceramic
- **Particle Size D50 (µm)**: typical particle size, particle size, powder particle size
- **Size Range (µm)**: particle sizes, particle size range, typical particle sizes
- **Ceramic Type (Alumina/Zirconia)**: ceramic material, alumina, zirconia
- **Cure Wavelength (nm)**: irradiation wavelength, blue-light wavelength

## Extraction notes

- DLP projects a full layer at once (per-layer exposure time), unlike laser-scanned SLA — do not confuse layer exposure time with per-point/scan-based timing terminology from SLA papers.
- In ceramic DLP papers, report stated ceramic content/loading and particle-size ranges as feedstock properties when those values are explicit. Do not treat a suspension-layer thickness as the realized printed layer thickness.
- Map Projector Resolution (µm/px) only to a physical pixel size stated in µm/px or an equivalent explicit pixel-size phrase; do not map a projector's pixel-count array (for example, 1920 × 1080 pixels) to this parameter.
- Check both shaping and post-processing excerpts: ceramic material type, debinding cycle, and sintering temperature are in scope when the paper states them.
