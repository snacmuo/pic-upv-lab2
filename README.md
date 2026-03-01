# upvfab_design_tools

Small Python library for photonic waveguide/coupler mode analysis and EME propagation, extracted from the lab notebook workflow.

## Install (editable)

```bash
pip install -e .
```

## Main API

```python
from upvfab_design_tools import (
    MMI_EME,
    DC_EME,
    waveguide,
    waveguide_array,
    my_plot_mode,
)
```

## Notebook Migration

In `20_COUPLERS_v2025_Student.ipynb`, replace local class/function definition cells with imports from `upvfab_design_tools`.

## Notes

- API keeps compatibility aliases like `waveguide_Array`.
- The package includes light bug fixes from the original script while preserving behavior.
