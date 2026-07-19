# Methods and Experiment Plan

## 1. Research separation

LG-GWR and GR-GWR are evaluated as two independent method studies. They share data loading, metrics, spatial blocking, reproducibility controls, and reporting code, but their hypotheses and structural recovery criteria remain separate.

The overall evidence structure is:

> Real-data fitting as the main analysis, controlled simulations for structural recovery, ablation and sensitivity analysis, and limited prediction experiments as supplementary validation.

## 2. LG-GWR questions

1. Does separable LG-GWR degenerate numerically to geographic GWR when attribute weighting is disabled?
2. Does LG-GWR remain comparable to GWR in Euclidean scenario L1?
3. Does it recover anisotropic or geography–attribute geometry in L2–L4?
4. Does it avoid exploiting irrelevant attributes in L5?
5. Does shared geometry reveal its limitation in parameter-specific scenario L6?

Primary structural evidence includes coefficient bias and RMSE, coefficient correlation, pairwise-distance correlation, kNN overlap and Jaccard, convergence, restart stability, and metric contributions. Real-data fitting should report AICc, R², adjusted R², RSS, RMSE, MAE, ENP, and residual Moran's I.

## 3. GR-GWR questions

1. Does one-regime GR-GWR degenerate numerically to standard GWR?
2. Does GR-GWR recover connected abrupt regimes in R2–R4?
3. Does boundary control reduce cross-boundary smoothing and fragmentation?
4. Does it avoid manufacturing regimes in R1 and smooth-transition R5?
5. Does its connectivity assumption become limiting in disconnected-mechanism R6?

Primary structural evidence includes coefficient RMSE, ARI, NMI, boundary precision/recall/F1, selected regime count, connectivity, convergence, and stability. Real-data fitting should additionally report regime sizes, within-regime coefficient variance, between-regime differences, connected components, and boundary counts.

## 4. Simulation design

The full profile uses a 20 × 20 calibration grid (`n=400`), 100 independent target locations, two standard-normal predictors, Gaussian noise with standard deviation 0.5, and 100 Monte Carlo repetitions per scenario. All random seeds are deterministic and recorded in result tables.

## 5. Real-data experiments

Bundled pyGWRx data are loaded through `pygwrx.io.load_dataset`. The empirical study should contain two layers:

1. Full-sample fitting and interpretation as the main paper evidence.
2. A smaller repeated spatial-block or held-out experiment as supplementary evidence against severe overfitting.

Housing uses a deterministic spatially stratified `n=2,000` sample and writes selected indices to the generated result directory.

LG-GWR datasets: EWHP, Crime, HIV, Housing, and Dublin Voter as an interpretation supplement. GR-GWR datasets: Georgia, Dublin Voter, and Housing. Columbus is retained for correctness and small-sample visualization.

## 6. Baselines

The runnable core includes OLS and pyGWRx GWR for every scenario, plus LG-GWR Joint/Separable or GR-GWR as appropriate. The full profile enables pyGWRx MGWR and SGWR. External SGWR-GD, GNNWR, Bayesian Cluster GWR, and oracle-regime models require independently validated implementations and must not be misrepresented as pyGWRx models.

## 7. Statistical analysis

After complete execution, use repetition- or fold-level paired Wilcoxon tests, Holm correction, bootstrap 95% confidence intervals, effect sizes, failure rates, convergence rates, and rank summaries. Pilot profiles are intended for software validation and must not be used for formal significance claims.
