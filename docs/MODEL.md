---
title: Evolutionary Game Modeling of Immigration Policy
subtitle: English mathematical notes and reproducibility audit
author: Kai Zhang
date: 23 October 2026
---

# Scope and provenance

This technical companion reconstructed the mathematical content of three Python scripts supplied with an undergraduate thesis project. The files contained **two different payoff specifications**, not merely different plotting languages. This document therefore treated them separately:

1. **Government-benefit spillovers.**  
   The scripts `replicator_sim_AB_no_pandas.py` and `replicator_compare_jp.py` used the same payoffs and a clipped fourth-order Runge--Kutta scheme.
2. **Firm-cost spillovers with mutation.**  
   The script `replicator_paper_ready_jp.py` used a different payoff specification, an effective hiring cost and a small mutation term, with clipped explicit Euler updates.

The English exposition, analytic Jacobians, source-table checks and adaptive-solver diagnostics were prepared for this repository release. Original scripts and five output tables were preserved under `source_snapshot/` with SHA-256 hashes. No full thesis manuscript, MATLAB source or empirical calibration dataset was included in the supplied folder. This document is an English mathematical reconstruction from the recoverable code, rather than a sentence-by-sentence translation of an absent manuscript.

**A consequential numerical finding:** the original $10^{-9}$ population floor changed the government model's finite-horizon hiring path substantially. Reproducing the original tables and solving the unmodified continuous ODE are different checks; both are reported below.

# Players, strategies and state variables

The model represented three interacting populations. A strategy share was a population-level state, rather than a forecast probability estimated from data.

| Population | First strategy | Second strategy | State |
|---|---|---|---|
| Government | $K_1$: restrictive/contraction policy | $K_2$: expansion policy | $x$: share choosing $K_1$ |
| Firms | $M_1$: hire foreign talent | $M_2$: hire domestic talent | $y$: share choosing $M_1$ |
| Talent | $N_1$: migrate to Japan | $N_2$: alternative destination/outside option | $z$: share choosing $N_1$ |

Thus $q=(x,y,z)\in[0,1]^3$. In particular, **$x=0$ denotes expansion**, while $x=1$ denotes restriction. This orientation matters when interpreting a trajectory toward $(0,1,1)$.

Time and payoffs were illustrative model units. The parameter values were not estimated treatment effects, calibrated currency amounts or observations of real migration rates.

# Model A: government-benefit spillovers

## Expected payoffs

The supplied expected payoffs were

$$
\begin{aligned}
\pi_G(K_1)&=E_2-C_2,\\
\pi_G(K_2)&=E_1-C_1-yzS_1-zS_2+R_1yz+R_2z,\\
\pi_F(M_1)&=-C_3+zE_3+(1-x)zS_1,\\
\pi_F(M_2)&=E_4-C_4,\\
\pi_T(N_1)&=E_5-C_5+(1-x)S_2,\\
\pi_T(N_2)&=E_6.
\end{aligned}
$$

$S_1$ was a firm subsidy and $S_2$ a talent subsidy under expansion. In this specification the firm subsidy was conditional on foreign hiring **and** migration, hence $yzS_1$ in the government budget. $R_1yz$ and $R_2z$ represented government benefits associated with hiring/migration. They were distinct from a reduction in a firm's cost.

An equivalent pure-strategy payoff table follows. Entries are ordered as (government, firm, talent).

| Profile | Government | Firm | Talent |
|---|---|---|---|
| $K_1,M_1,N_1$ | $E_2-C_2$ | $E_3-C_3$ | $E_5-C_5$ |
| $K_1,M_1,N_2$ | $E_2-C_2$ | $-C_3$ | $E_6$ |
| $K_1,M_2,N_1$ | $E_2-C_2$ | $E_4-C_4$ | $E_5-C_5$ |
| $K_1,M_2,N_2$ | $E_2-C_2$ | $E_4-C_4$ | $E_6$ |
| $K_2,M_1,N_1$ | $E_1-C_1-S_1-S_2+R_1+R_2$ | $E_3-C_3+S_1$ | $E_5-C_5+S_2$ |
| $K_2,M_1,N_2$ | $E_1-C_1$ | $-C_3$ | $E_6$ |
| $K_2,M_2,N_1$ | $E_1-C_1-S_2+R_2$ | $E_4-C_4$ | $E_5-C_5+S_2$ |
| $K_2,M_2,N_2$ | $E_1-C_1$ | $E_4-C_4$ | $E_6$ |

## Replicator equations

For a two-strategy population with first-strategy share $u$, the average payoff is $\bar\pi=u\pi_1+(1-u)\pi_2$. Therefore

$$\dot u=u(\pi_1-\bar\pi)=u(1-u)(\pi_1-\pi_2).$$

Define $A=(E_2-C_2)-(E_1-C_1)$ and $d=E_5-C_5-E_6$. The three selection gaps are

$$
\begin{aligned}
G(y,z)&=A+(S_1-R_1)yz+(S_2-R_2)z,\\
F(x,z)&=-C_3-E_4+C_4+z[E_3+(1-x)S_1],\\
H(x)&=d+(1-x)S_2.
\end{aligned}
$$

The continuous-time system is

$$\dot x=x(1-x)G,\qquad \dot y=y(1-y)F,\qquad \dot z=z(1-z)H.$$

Each face of the unit cube is invariant in the exact mutation-free ODE: when one share is exactly zero or one, its derivative is zero. A small *positive* share can nevertheless become extremely small before recovering.

# Equilibria and local stability in Model A

## Stationarity and the Jacobian

The full stationarity condition is the simultaneous factorization

$$
\begin{gathered}
x\in\{0,1\}\ \text{or}\ G(y,z)=0,\\
y\in\{0,1\}\ \text{or}\ F(x,z)=0,\\
z\in\{0,1\}\ \text{or}\ H(x)=0.
\end{gathered}
$$

It includes eight corners and potentially mixed equilibria on edges, faces or in the interior. A corner catalogue alone is not a general enumeration of every equilibrium.

Writing $a=x(1-x)$, $b=y(1-y)$ and $c=z(1-z)$, the analytic Jacobian is

$$
J=\begin{pmatrix}
(1-2x)G & a(S_1-R_1)z & a[(S_1-R_1)y+S_2-R_2]\\
-bS_1z & (1-2y)F & b[E_3+(1-x)S_1]\\
-cS_2 & 0 & (1-2z)H
\end{pmatrix}.
$$

At a hyperbolic equilibrium, strictly negative eigenvalue real parts imply local asymptotic stability; any positive real part implies instability. A zero real part requires further analysis. These are local dynamical statements, not a proof of global convergence or evolutionary stability in every game-theoretic sense.

## The expansion--foreign-hiring--migration corner

At $q^*=(0,1,1)$, the Jacobian is diagonal, with eigenvalues

$$
\begin{aligned}
\lambda_x&=A+S_1-R_1+S_2-R_2,\\
\lambda_y&=C_3+E_4-C_4-E_3-S_1,\\
\lambda_z&=-(d+S_2).
\end{aligned}
$$

Consequently, strict local stability requires all three conditions:

$$
\begin{aligned}
R_1+R_2&>A+S_1+S_2,\\
C_3&<E_3+S_1-(E_4-C_4),\\
S_2&>C_5+E_6-E_5.
\end{aligned}
$$

For the supplied baseline these give eigenvalues $(-0.7,-0.6,-0.1)$, a hiring-cost bound $C_3<1.5$, and a talent-subsidy bound $S_2>0.5$. These thresholds are conditional on this corner and specification. They are not empirical policy estimates, and they must not be transferred to Model B.

## A fully interior candidate

When the denominators are nonzero, solving $G=F=H=0$ gives

$$
\begin{aligned}
x^*&=1+\frac{d}{S_2},\\
z^*&=\frac{C_3+E_4-C_4}{E_3+(1-x^*)S_1},\\
y^*&=\frac{-A/z^*-(S_2-R_2)}{S_1-R_1}.
\end{aligned}
$$

All three coordinates must lie strictly between zero and one. The baseline candidate has $x^*=1/6$, $z^*=0.825$, and $y^*\approx-0.803$, so it is infeasible. Degenerate denominators require separate analysis.

At any fully interior stationary point of this particular mutation-free specification, all three diagonal Jacobian entries vanish, so $\operatorname{tr}J=0$. Strict Hurwitz stability is therefore impossible there. This observation does not decide nonhyperbolic nonlinear stability.

# Model B: firm-cost spillovers and mutation

## Different payoff assumptions

The later source script used

$$
\begin{aligned}
C_3^{\mathrm{eff}}(z)&=\max(C_3-\eta z,0),\\
\pi_G(K_1)&=E_2-C_2,\\
\pi_G(K_2)&=E_1-C_1-yS_1-zS_2,\\
\pi_F(M_1)&=0.15E_4+zE_3+(1-x)S_1-C_3^{\mathrm{eff}}(z),\\
\pi_F(M_2)&=E_4-C_4,\\
\pi_T(N_1)&=E_5-C_5+(1-x)S_2,\qquad \pi_T(N_2)=E_6.
\end{aligned}
$$

Here the government paid $yS_1$, rather than $yzS_1$; the firm received $(1-x)S_1$, rather than $(1-x)zS_1$; government benefits $R_1,R_2$ were absent; and the firm had an additional $0.15E_4$ term. That coefficient was preserved from the script, not estimated in this release.

## Replicator dynamics with mutation

Let $\Delta=(G_B,F_B,H)$ denote the payoff gaps derived from those payoffs. For $q_i\in\{x,y,z\}$,

$$\dot q_i=r_iq_i(1-q_i)\Delta_i+\mu_i(1/2-q_i).$$

The mutation term is linear mean reversion toward one half. At a zero face it contributes $+\mu_i/2$; at a one face it contributes $-\mu_i/2$. With positive mutation, the corners are **not** equilibria and the exact vector field points inward on every face.

Away from the cost kink, write $D=\partial\Delta/\partial q$. Then

$$
J_B=\operatorname{diag}(r_i(1-2q_i)\Delta_i-\mu_i)
+\operatorname{diag}(r_iq_i(1-q_i))D,
$$

where

$$
D=\begin{pmatrix}
0&S_1&S_2\\
-S_1&0&E_3+\eta\,\mathbf{1}_{\{C_3>\eta z\}}\\
-S_2&0&0
\end{pmatrix}.
$$

At $C_3=\eta z$ the maximum is nondifferentiable. The implementation refuses a smooth Jacobian there instead of silently choosing a derivative. The supplied baseline and scan ranges remain on the positive-effective-cost branch throughout the unit cube.

# Parameter dictionary and baselines

| Parameter | Interpretation | Model A | Model B |
|---|---|---:|---:|
| $E_1,E_2$ | Expansion / restriction government benefits | 2.0, 1.5 | 2.60, 1.70 |
| $C_1,C_2$ | Expansion / restriction government costs | 0.6, 0.2 | 0.35, 0.20 |
| $E_3$ | Benefit conditional on foreign talent arriving | 2.0 | 1.20 |
| $E_4,C_4$ | Domestic-hiring benefit and cost | 1.8, 0.5 | 1.15, 0.60 |
| $C_3$ | Foreign-hiring friction cost | 0.9 | 1.05 |
| $E_5,C_5$ | Japan migration benefit and cost | 2.0, 0.6 | 1.85, 0.80 |
| $E_6$ | Outside-option payoff | 1.9 | 1.35 |
| $S_1,S_2$ | Firm/talent subsidies | 0.8, 0.6 | 0.10, 0.22 |
| $R_1,R_2$ | Government spillover benefits | 1.2, 0.8 | absent |
| $\eta$ | Hiring-cost spillover coefficient | absent | 0.25 |
| $r_x,r_y,r_z$ | Adaptation speeds | 1, 1, 1 | 1, 1, 1 |
| $\mu_x,\mu_y,\mu_z$ | Mutation coefficients | 0, 0, 0 | $(5,10,5)\times10^{-4}$ |

Both source suites used $T=200$, $h=0.05$, and a final-30% summary window. Cases started at $(0.6,0.3,0.2)$, $(0.6,0.75,0.2)$ and $(0.3,0.75,0.6)$. The policy case lowered $C_3$ to 0.6 in Model A and to 0.90 in Model B. Their different baselines and policy changes must remain explicit.

# Numerical schemes and validation

The historical Model A solver used classical RK4:

$$q_{n+1}=\operatorname{clip}_{[\epsilon,1-\epsilon]}
\left[q_n+\frac h6(k_1+2k_2+2k_3+k_4)\right],\qquad \epsilon=10^{-9},$$

with clipped intermediate states inside the vector field. Model B used

$$q_{n+1}=\operatorname{clip}_{[\epsilon,1-\epsilon]}[q_n+h f_B(q_n)].$$

RK4 and Euler have different integration errors; the clipping operator changes the numerical process further. A floor is not the same mechanism as the explicitly specified mutation term.

An independent SciPy DOP853 reference was added for this release. For Model A, positivity was protected using logit coordinates $u_i=\log(q_i/(1-q_i))$. Since $\dot q_i=q_i(1-q_i)\Delta_i$, the transformed equation is simply

$$\dot u_i=\Delta_i(\operatorname{logistic}(u)),\qquad q_i=\operatorname{logistic}(u_i).$$

This avoids replacing extremely small positive shares with an arbitrary floor. A direct absolute-tolerance solver can otherwise lose their sign and fail when the advantage later changes sign. Model B's positive mutation kept its baseline equilibrium away from the boundary, so its reference used the original state coordinates.

Verification comprised:

- matching original strategy payoffs and vector fields;
- checking both analytic Jacobians with central finite differences, including both smooth cost branches;
- checking Model A's eight corners and the eigenvalues at $(0,1,1)$;
- checking inward mutation flux and invalid parameter/grid inputs;
- comparing source-scheme paths with adaptive references;
- reproducing all five archived tables: two case summaries, two $C_3$ scans and one $S_2$ scan;
- checking the long-horizon clipping effect explicitly.

The frozen dependencies and machine-readable diagnostics are included in `requirements.txt` and `results/validation.json`. Unit tests do not replace calibration or a complete equilibrium analysis.

# Results and interpretation

## Faithful reproduction of source outputs

The historical clipped government baseline ended near $(0,1,1)$ at $T=200$. The mutation model ended near $(0.000334,0.000378,0.003112)$. Its small positive values reflected mutation, rather than exact extinction. The English code reproduced the archived output tables within their saved precision.

The source Model A hiring-cost grid contained 23 points from 0.3 to 1.4; Model B contained 17 points from 0.6 to 1.4 and 13 subsidy points from 0 to 0.6. A plotted scan was a finite grid of simulations, not proof of a unique bifurcation threshold.

The Model B baseline talent gap obeyed $H(x)=-0.30+0.22(1-x)<0$ throughout the cube. Increasing talent support or reducing foreign-hiring costs therefore addressed different channels. The baseline was not evidence that every subsidy necessarily produces high adoption.

## Clipping versus the unmodified ODE

For the same Model A baseline and initial state, the source floor and the continuous reference produced very different finite-horizon hiring shares:

| Method | $y(200)$, approximately |
|---|---:|
| Historical RK4 with floor $10^{-9}$ | 0.999999999 |
| Unmodified ODE, logit DOP853 | $1.22\times10^{-21}$ |

The continuous path reached a foreign-hiring share of approximately $1.54\times10^{-51}$ before recovering. Replacing that rare strategy with a floor of $10^{-9}$ dramatically shortened recovery. This difference persisted when the historical time step was refined, because reducing a time step does not remove the floor.

The locally stable $(0,1,1)$ corner and a long delay before reaching it are compatible: local eigenvalues do not set a uniform arrival time from every initial state. Neither finite-horizon output alone proved global convergence. The repository therefore retained both the faithful original numerical outputs and the continuous-model diagnostic, rather than presenting the former as an accurate unclipped ODE solution.

The original case named "path dependence" changed the initial state. That label alone did not establish multiple persistent attractors or genuine asymptotic path dependence. Initial-state sensitivity, transient delay and different attracting equilibria are distinct claims.

# Reproduction commands and accompanying artifacts

From the repository root, using Python 3.12 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m immigration_game.reproduce
```

The public repository contains the English figures, CSV results, source manifest and validation diagnostics:

[Evolutionary Game Modeling of Immigration Policy](https://github.com/Kai1024-star/evolutionary-game-immigration-policy).

The figures separate government spillovers, firm-cost spillovers, original-grid sensitivity and the clipping diagnostic. They are generated from illustrative payoffs, not measured policy data. The supplied MATLAB work and original manuscript, if available separately, would require their own documentation and verification.
