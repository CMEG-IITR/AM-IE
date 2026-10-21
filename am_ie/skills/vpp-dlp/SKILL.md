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
- Check each in-scope taxonomy parameter independently, including values embedded in general process descriptions, captions, and post-processing sentences; one sentence may support multiple distinct parameters.
- Keep ceramic feedstock values distinct: particle-size range, D50, solid loading, cure wavelength, and cure depth are separate parameters. Report only the ones explicitly supported by the cited text.
- For layer thickness, report the intended realized layer thickness or explicit range. A doctor-blade gap, recoating clearance, suspension-film height, or coating thickness is not the realized layer thickness unless the paper says it is.
- Treat explicit bounds such as “porosity below X%” as a reported porosity value and retain the bound in the value. Check atmosphere statements in debinding and sintering sections.
- Assign a named ceramic such as alumina only to a row whose cited text directly connects that ceramic to the parameter. Do not copy a generic suspension name onto unrelated geometry, dimensional, debinding, or sintering values.
- Captions and figure descriptions can explicitly identify both the ceramic composition and the printed part geometry; extract those as separate values when stated.
- Keep debinding schedule values separate: emit distinct rows for a heating-rate range and dwell-time range when both appear in one sentence. Map the furnace atmosphere to Atmosphere (Ar/N2/Vacuum), not to Debinding Cycle.
- A phrase such as “a [ceramic] [part] made of [named ceramic]” can support both Ceramic Type and Part Geometry; keep each value tied to that exact caption or sentence.
- Extract an explicitly named process atmosphere under Atmosphere (Ar/N2/Vacuum), including when it appears in a debinding sentence alongside schedule details.
