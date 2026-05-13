# Python Usage (legacy — superseded)

> **This page documented the low-level `mgcv_rust.GAM` class and the
> `fit_auto` / `fit_formula` / `add_cubic_spline` helpers.** Most users
> should now reach for the high-level `mgcv_rust.Gam` wrapper. See:
>
> - [`docs/GETTING_STARTED.md`](GETTING_STARTED.md) — tutorial.
> - [`README.md`](../README.md) — overview + quick reference.
>
> The low-level `GAM` class is still exported (`from mgcv_rust import GAM`)
> for callers who want direct control over the Rust core; its API is
> unchanged. The notes below describe the legacy three-fit-method
> surface for that path and are kept for historical reference only.

---

## Three fit methods on the low-level `GAM` class

```python
from mgcv_rust import GAM
import numpy as np

n = 300
x = np.linspace(0, 1, n).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + 0.5 * np.random.randn(n)

gam = GAM()
gam.fit_auto(x, y, k=[15], method="GCV")  # automatic, k per predictor column
# or
gam.fit_formula(x, y, formula="s(0, k=15)", method="GCV")
# or
gam.add_cubic_spline("x", num_basis=15, x_min=0.0, x_max=1.0)
gam.fit(x, y, method="GCV")

preds = gam.predict(x)
```

These remain available; just prefer `Gam` for new code.
