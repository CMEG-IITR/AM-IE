---
name: mex
description: "Extraction guidance for Material Extrusion additive manufacturing papers."
---

# Material Extrusion (MEX)

## Synonyms

- **Nozzle Temperature**: extrusion temperature, print head temperature, hot end temperature
- **Print Speed**: extrusion speed, printing speed, travel speed
- **Machine Make/Model**: printer model, FDM printer, printer manufacturer
- **Machine/Robot Type**: FDM printer, multi-material 3D printer, printer type
- **Build Orientation**: printing direction, print orientation, orientation of printing

## Extraction notes

- Bed temperature and nozzle temperature are usually both reported — don't stop after finding one.
- Check Materials/Methods for the printer's manufacturer and model (for example, "FDM printer (Felix 3.0, ...)") and extract it as Machine Make/Model when the excerpt names it.
- Use Machine/Robot Type for a generic description such as "multi-material 3D printer with dual nozzles"; use Machine Make/Model only for an identified make or model.
- Build Orientation can be stated indirectly: for example, "printing direction was kept parallel to the slit, i.e. XY direction" reports XY direction.
- Do not report generic printer capability ranges (for example, "FDM printers allow printing layers ranging from 50 μm to 500 μm") as settings used for the paper's specimens.
- Infill percentage and pattern are often in a separate sentence from thermal parameters.
