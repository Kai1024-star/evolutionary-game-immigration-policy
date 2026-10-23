# Source provenance and interpretation

The source folder was supplied by Kai Zhang as undergraduate-thesis material. The English repository was prepared on 23 October 2026.

## File mapping

| Archived source | What it implemented | English counterpart |
|---|---|---|
| `replicator_sim_AB_no_pandas.py` | Government spillover payoffs and clipped RK4 | `models.py`, model `government` |
| `replicator_compare_jp.py` | Same government payoffs, Japanese plots and on/off comparison | `reproduce.py`, government comparison |
| `replicator_paper_ready_jp.py` | Different firm-cost spillover payoffs, mutation and Euler | `models.py`, model `firm` |
| `government_outputs/*.csv` | Four original cases and 23-point C3 scan | `results/government_*.csv` |
| `firm_outputs/*.csv` | Four original cases, 17-point C3 and 13-point S2 scans | `results/firm_*.csv` |

The original scripts and saved tables were copied without modification. `source_snapshot/manifest.json` records their SHA-256 digests. Duplicate Japanese figures and platform metadata were not required for the English release; the figures were regenerated in English.

## What was added for this release

- English explanations of the payoff logic and mathematical derivations.
- A shared English implementation with explicit separation of the two specifications.
- Analytic Jacobians and Model A corner-stability calculations.
- A positivity-preserving logit-coordinate adaptive reference for Model A.
- An adaptive reference for Model B, source-CSV checks and tests.
- A diagnostic showing the material effect of the original population floor.

The supplied RK4 loop performed one unused update after its final recorded sample; the English implementation omitted that unused update. It also required T/dt to be an integer, preventing a mismatch between a plotted linspace and the actual integration step. Neither change altered the original default recorded time grid.

## Claim boundaries

The payoffs were illustrative inputs. Simulation output should not be described as observed migration behaviour, causal identification, fitted policy efficacy or an optimal policy. The eight pure corners do not exhaust every possible equilibrium for arbitrary parameters. Mutation makes those corners nonstationary in Model B. Eigenvalues describe local linear behavior; zero real parts and the cost kink require further analysis.

The original file named an initial-state scenario “path dependence”; this did not prove multiple asymptotic attractors. The government population floor should not be described as a harmless accuracy adjustment: it can change recovery times and finite-horizon conclusions by many orders of magnitude.

No full thesis manuscript, MATLAB implementation or empirical dataset was present in the supplied folder. The repository documents the recoverable Python models. The original research period shown on the author's resume is distinct from this later English packaging date; Git history was not backdated.
