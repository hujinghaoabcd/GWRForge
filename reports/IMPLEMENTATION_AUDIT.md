# Implementation Audit

- Authoritative model dependency: `pyGWRx==0.1.2`.
- LG-GWR public calls: `LGGWR(...).fit(X, y, coords, attributes)` and `predict_result(...)`.
- GR-GWR public calls: `GRGWR(...).fit(X, y, coords)` and `predict_result(...)`.
- Standard baseline: `GWR(...).fit(...)` from the same package.
- Bundled data are loaded only through `pygwrx.io.load_dataset`.
- Experiment-specific code is limited to data-generating processes, spatial folds, metric calculation, result normalization, plotting, and reporting.
- The pyGWRx source tree is not copied or modified by this repository.
- Generated outputs, wheels, caches, and environment-specific artifacts are excluded from version control.
