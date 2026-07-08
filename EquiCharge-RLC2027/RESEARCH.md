# The Price of a Fair Charge: Equitable EV Charging Access under Grid Scarcity, and the Tariff Design that Delivers It

> Target venue: **AAAI-27, AI for Social Impact (AISI) special track.**
> AISI keywords: **Energy; Mobility and Transportation; Social Welfare / Justice /
> Fairness / Equality.**
>
> This document is the research write-up that accompanies the
> [`chargax.equity`](chargax/equity) implementation and the paper
> [`paper/aaai27_equicharge.tex`](paper/aaai27_equicharge.tex). It states the
> problem, the actionable characterization, the method, the positioning relative to
> prior work, and the experimental protocol, and reports results produced by
> [`experiments/`](experiments) (which match [`results/RESULTS.md`](experiments/results/RESULTS.md)).

---

## Abstract

As electric vehicles reach lower-income drivers, the fairness of public charging
becomes an **access** question. A site draws power through a fixed grid connection,
so at busy hours it must ration. Real operators price by **tier** (memberships,
dynamic pricing, fast-vs-slow products), drivers sort into tiers by budget in a way
that **correlates with income**, and a revenue-maximizing allocation under scarcity
routes power to the higher-margin tiers — a **disparate impact** on lower-income
drivers, not deliberate income discrimination. Using an **exact offline solver on a
real-data-calibrated** environment we characterize this at optimality and reach three
findings a utility, operator, or regulator can act on. **(1) The price of fairness is a
revenue cost, not a service cost:** equalizing tier satisfaction costs ~9.5% of
margin-weighted revenue while it *raises* mean satisfaction (~0.72→0.84) and roughly
doubles the worst tier (~0.40→0.83) — though this is a redistribution the **premium
tier pays for** (it gives up ~0.11), a consequence of concave saturating utility that
we report rather than sell as free. **(2) Only income-neutral pricing removes the
disparate impact** (the tier gap 0.54→0.04, −93%); a subsidy to the poorest tier alone
does not close it, it **shifts the burden to the middle tier**. **(3) Grid capacity is
a second, independent lever**, with a metric-dependent threshold (~50 kW by Gini, ~70 kW
by the oracle gap). We release a **fair-charging benchmark** on Chargax with a corrected
two-stage oracle, the fairness layer, baselines, and a state-augmented learning
environment; no online policy reaches positive worst-decile satisfaction. Our
contribution is the **problem formulation, the actionable characterization, and the
tariff/grid levers**, with a **second pillar**: a welfare-shaped learned controller that,
judged on **inclusive** metrics (rejected=0) and against the deployed heuristics, beats
the strongest heuristic on profit, inclusive service, within-population inequality, and rejection
at once (7/10 seeds dominate all four; it rejects *fewer* drivers, so the win is not
survivorship). Paired with an honest negative — the equity weight is **not** a tunable
between-tier fairness dial but a training-stabilizing auxiliary; the between-tier disparate
impact stays an offline phenomenon the tariff, not the controller, addresses.

> **A note on the corrected result.** An earlier version reported the oracle maximin
> as a ~50% cut in mean satisfaction (segments leveled to ~0.36). That was a
> single-epigraph LP *leveling-down artifact*. The Pareto-efficient two-stage maximin
> equalizes at ~0.84 mean satisfaction with the *same delivered energy* as the profit
> optimum, so the true price of fairness is a modest **revenue** cost. This correction
> strengthens the social-impact finding rather than weakening it.

---

## 1. Contributions

1. **An actionable characterization (the social-impact core).** Using a corrected
   clairvoyant oracle on a real-data-calibrated environment we show (a) the price of
   fairness is paid in the operator's **revenue** (~10%), not in aggregate service —
   equalizing *raises* mean satisfaction and doubles the worst segment; (b) the
   **rate-design finding** that only income-neutral pricing removes the price-driven
   gap, while a poorest-segment-only subsidy backfires; and (c) the **grid-capacity
   threshold** that removes the scarcity-driven residual. Each maps to a lever a
   decision-maker already controls (Section 6, [Path to practice](#65-path-to-practice)).

2. **A problem formulation.** We formalize online EV charging under grid scarcity as
   *welfare-optimal control over an endogenous, policy-influenced beneficiary
   population* with departure-realized, non-additive utility. This conjunction —
   non-linear social welfare × stochastic arrivals **and departures** × capacity
   coupling — is not covered by prior work (Section 5).

3. **A released benchmark (facilitation).** [`chargax.equity`](chargax/equity) ships
   the fairness layer, rejection-aware arrivals, a **two-stage Pareto-efficient
   oracle** with revenue accounting, inequality metrics, baselines, affordability-
   grounded segments, the tariff analysis, and a state-augmented learning
   environment. The learned controller is a **benchmark baseline** (B2): the open
   problem it surfaces is that no online policy reaches positive worst-decile
   satisfaction and a trivial heuristic is competitive with learned welfare policies.

4. **A justice-faithful utility/welfare definition** that counts
   unserved-at-deadline, still-charging-at-horizon, **and rejected** drivers at zero
   utility over the *true arrival stream* (a *sample-then-reject* simulator change),
   robust to the reject-to-look-fair loophole we audit (Section 6.6).

5. **Method** enabling the above: a **bounded sufficient-statistic** augmentation
   (every welfare has generalized-mean form `W = Ψ((1/N)·Σ φ_α(u_i))`, so `(N_g,
   Σφ_α(u_i))` per segment is all the memory needed) and a **dense potential-shaped,
   undiscounted welfare-telescoping** reward exact over the daily horizon.

---

## 2. Problem formulation

**Environment.** Chargax simulates a charging station as a tree of power nodes
(splitters, EVSEs, on-site batteries) under a hard grid-connection limit. Each
5-minute step (one episode = one day = 288 steps) the agent sets a discrete
charging level per charger and per battery. Cars arrive by a time-of-day Poisson
process with heterogeneous attributes (battery capacity, arrival/desired charge,
dwell deadline, and a charge- vs. time-sensitive departure rule). When no charger
is free, an arrival is **rejected**.

**Per-customer utility.** For each customer *i* in the realized daily stream we
define satisfaction
```
u_i = clip( delivered_i / desired_i , 0, 1 ),
  desired_i = desired%·capacity − arrival_charge,   delivered_i = max(now − arrival_charge, 0),
```
realized at departure, at the horizon flush (still-charging cars are counted), or
at rejection (`u_i = 0`). This is defined over **all** arrivals, not only those
who left satisfied — closing the loophole that a charge-sensitive car leaves
*only* once satisfied (utility ≈ 1 by construction).

**Ability-to-pay segments (the source of the efficiency–equity tension).** Each
arriving customer is assigned, *independently of battery capacity*, an
ability-to-pay segment *g* with a price multiplier on the electricity tariff
(e.g. budget 0.6×, mid 1.0×, premium 1.8×). Revenue is therefore value-weighted,
so a profit maximizer is *incentivized* to direct scarce power to high-paying
customers. Crucially, the welfare is over *charging satisfaction* (delivered ÷
desired energy), which is **price-agnostic** — so there is a genuine conflict
between profit and equity. (Under flat pricing this conflict vanishes: max-charge
is then simultaneously the most profitable *and* the most equal policy, because
profit = total energy delivered, which also maximizes satisfaction. We verified
this empirically — hence the need to model ability-to-pay, the axis on which real
EV-charging inequity actually arises.) The price multiplier of each connected car
is included in the observation so the policy can act on it; segment membership is
exogenous, sidestepping demographic-proxy objections.

**Welfare objective.** Given the vector of realized utilities (grouped into the
ability-to-pay segments *g*), we optimize a social welfare function
```
W = Ψ_g( { EDE_α( (1/N_g) Σ_{i∈g} φ_α(u_i) ) } ),
```
the equally-distributed-equivalent (EDE) form of α-fairness, with an outer
operator Ψ that is either the count-weighted pool (individual α-fairness) or a
Rawlsian soft-min across segments. α=0 is utilitarian, α=1 is Nash, α→∞ is
leximin. The reward blends profit and welfare via an equity weight λ∈[0,1]:
```
r_t = (1−λ)·Δprofit_t / profit_scale  +  λ·( W(R_{t+1}) − W(R_t) ).
```

**MDP over an endogenous population.** Two properties make this *not* standard
fair RL: (i) the set of beneficiaries is dynamic and its composition is partly
*chosen* by the policy (admission, and dwell via charge completion / V2G); (ii)
W is a non-linear function of the whole realized-utility distribution, so it is
non-additive across steps.

---

## 3. Method

**Welfare telescoping (γ=1).** With the per-step reward `W(R_{t+1})−W(R_t)` and an
**undiscounted** finite-horizon return, the episode return telescopes to
`W(R_T) − W(R_0) = W(R_T)` (constant `R_0`), i.e. exactly the welfare of the final
allocation. *This identity holds only at γ=1*; with γ<1 it becomes potential-based
shaping of an approximate objective. Because a Chargax episode is a bounded
288-step day, we optimize the undiscounted return (γ=1) and report the empirical
γ-ablation gap. (This directly addresses the most common technical objection to
welfare-telescoping methods; cf. Siddique et al. 2020, who treat the discounted
case as an *approximation* of the average-reward fair objective.)

**Bounded sufficient statistic.** Every welfare above has the generalized-mean
form `Ψ((1/N)Σφ(u_i))`. Therefore the per-segment pair `(N_g, S_g = Σ φ_α(u_i))`
is a *bounded sufficient statistic* for W, independent of how many customers
stream through. We (a) accumulate it in the state, and (b) append a normalized
version of it — plus present per-segment unmet demand and day-progress — to the
observation (`fair_context`). This is the state augmentation that makes the
non-additive objective Markov, generalized from the *fixed*-dimension
accrued-reward augmentation of prior fair RL to a *streaming* population.

**Sample-then-reject.** We modify arrivals so that a rejected customer's
attributes are sampled *before* rejection, making its (zero) utility and its
segment observable. Without this the simulator cannot count rejected demand, and
the welfare could be gamed by rejecting needy customers.

**Dense welfare shaping.** The departure-only welfare signal is *sparse* (a
customer's utility lands only when it leaves), which is hard for PPO. We define a
**provisional welfare potential** Φ(s) — the welfare that *would* be realized if
every currently-connected car departed now at its current satisfaction (plus the
already-departed and rejected customers) — and use Φ(s') − Φ(s) as the welfare
reward. Because connected cars are flushed at the horizon, Φ(terminal) equals the
realized final welfare and Φ(reset)=0, so this is **potential-based reward shaping
(Ng et al. 1999) that leaves the episode return — and hence the optimized
objective — exactly equal to the final welfare**, while firing on most timesteps
(charging a behind-schedule car raises Φ immediately, weighted by the concave
kernel). This is the difference between a signal on ~5% of steps and on ~60%, and
is what makes the egalitarian objective learnable.

**Optimizer.** PPO (jaxnasium) with γ=1, observation normalization, on the
state-augmented MDP. The same algorithm optimizes every point on the SWF family
and the Pareto sweep — only the reward (α, λ, outer operator) changes.

---

## 4. Experiments

**Regime.** A *power-scarce* station: many AC chargers behind a constrained grid
connection, so the binding constraint is **grid power shared among
simultaneously-present cars** (not charger count), with an on-site battery for
time-shifting. We verify the regime is power-bound (low rejection, large
abundant-vs-scarce satisfaction gap) before training. An *abundant* station (large
grid) is the contrast.

**Policies.**
- *profit-PPO* (λ=0): the status-quo (value-weighted) profit maximizer — our
  injustice evidence; it learns to favour high-paying segments.
- *welfare-PPO (egalitarian)*: Rawlsian soft-min across the ability-to-pay
  segments, λ-swept to trace the Pareto front.
- *welfare-PPO (utilitarian)*: the segment-agnostic total-satisfaction welfare,
  λ-swept — the efficiency-leaning comparison (does optimizing *total* welfare
  alone close the segment gap, or is the egalitarian operator needed?).
- *myopic fair heuristics*: proportional-fair and least-laxity-first (fair among
  present cars, no anticipation), plus max-charge and random.
- *SAFFE-style planner*: model-based anticipatory allocation against expected
  future demand (the closest competitor; Hassanzadeh et al. 2023).
- *offline oracle*: a clairvoyant LP (grid + per-car power/energy caps) giving the
  utilitarian and maximin welfare **ceilings** — the online–offline gap.

**Metrics.** Profit (€/day); mean / worst-10% / min satisfaction; Gini and
Atkinson inequality; group disparity; rejection rate; the optimized welfare; and
the **price of fairness** PoF = (Π\* − Π_fair)/|Π\*|. Multiple seeds with
standard errors.

**Reproduce.**
```bash
python -m experiments.run_experiments --oracle-only   # RL-free core (all offline findings)
python -m experiments.run_experiments                 # full study (multi-seed benchmark)
python -m experiments.run_experiments --quick         # fast smoke run
python -m experiments.summarize_results               # -> results/RESULTS.md
python -m experiments.plot_results                     # figures -> results/
python experiments/make_paper_tables.py                # paper/tables/*.tex
```

---

## 5. Novelty and positioning (verified literature scan)

The contribution is the **conjunction** below; each individual axis has prior work,
but not their combination.

| Axis | Closest prior work | How we differ |
|---|---|---|
| Non-linear SWF (Nash/α/leximin) over per-user utility in RL | Siddique–Weng–Zimmer (ICML 2020, GGF + accrued-reward augmentation); Fan et al. (AAMAS 2023, NSW); Mandal & Gan (NeurIPS 2022, NSW axioms); Ju et al. (ICLR 2024, first α-fairness regret) | All assume a **fixed** set of *D* users present throughout; we have a **streaming, policy-influenced** population and a bounded-statistic augmentation. |
| Non-Markovian fairness needs memory | Alamdari et al. (ICML 2024); Kumar & Yeoh (2025) | We extend the memory argument to a **dynamic population** via a bounded sufficient statistic. |
| Online fair division under stochastic arrivals, non-linear welfare | SAFFE (Hassanzadeh et al., ICAIF 2023); Sinclair et al. (OR 2022); Barman et al. (AAAI 2022); Huang et al. (ICALP 2025); Banerjee–Gkatzelis (SODA 2022) | These are **planning / online-algorithm** methods (not RL), assume **no departures** (or a known demand model), and target items/one-shot demand; we **learn** a policy under **departures + capacity coupling** and complex EV dynamics. |
| Departures / finite presence windows | Liu–Hajiesmaili (SIGMETRICS 2025, reusable resources) | Theirs is time-averaged **max-min**, not RL; we optimize the full α/leximin family with RL. |
| Concave-utility / convex-MDP RL | Zhang (2020); Zahavy (2021); Geist (2022); Mutti (2022/23) | These optimize a functional of **one system's** occupancy; ours is a welfare across a **population of distinct, transient agents**. Mutti's finite-trials result is our hook: the welfare must be evaluated on the **realized** daily allocation, motivating the state augmentation. |
| Fair EV charging | RDDPG / federated-DRL / MARL with **Jain index** (2022–2025) | Prior EV-RL fairness is an *ex-post Jain metric or reward term* with no guarantee; we make a principled **non-linear social welfare** the objective. |

**Single sharpest framing.** *Not* "fair RL for EV charging," but **welfare-optimal
RL when the welfare operator acts on a population the policy itself creates and
dismisses** — with EV charging as the compelling, real, JAX-scaled instance.

**On "is RL necessary?"** We make this a *conditional empirical* claim, not a
theorem: model-free RL outperforms model-based fair planning (SAFFE) precisely
when the dynamics are too complex for a tractable demand model (nonlinear charge
curves, V2G arbitrage, time-of-use prices, charger heterogeneity), and is
competitive otherwise. We never assert necessity — online fair-division theory
shows planning also anticipates.

---

## 6. Results

> Produced by `experiments/run_experiments.py`; see
> [`experiments/results/RESULTS.md`](experiments/results/RESULTS.md) for the exact
> numbers (the paper's tables are auto-generated from the same `results.json`).

Configuration of the committed run: 16 charger connectors behind a 30 kW grid, three
**affordability-grounded** ability-to-pay segments (budget/mid/premium ≈ 0.6/1.0/1.5×,
derived from income terciles + energy burden, see
[`segments.py`](chargax/equity/segments.py)), γ=1, dense welfare potential, 16
realized days for the oracle, multi-seed learning benchmark. Data is **real** (Dutch
arrival/dwell/demand + 2023 NL prices); see [`docs/DATA.md`](docs/DATA.md).

**6.1 The price of fairness is a revenue cost, not a service cost (offline oracle).**
The clairvoyant LP is independent of any RL optimization. The **profit-optimal**
allocation holds the **budget tier near 0.40** while giving mid/premium ≈0.84/0.94
(tier gap **0.54**, mean sat 0.72, revenue 402). The **corrected two-stage tier maximin**
equalizes at ≈0.83 (gap **0.01**) while delivering the **same total energy** as the
profit optimum, so mean satisfaction *rises* to ≈0.84 and the worst tier rises to ≈0.83.
The price of fairness is therefore **≈9.5% of margin-weighted revenue**, not a cut in
service — though the premium tier does give up ~0.11 (see 6.1b). (`oracle_tradeoff.png`.)
A single-epigraph maximin instead *levels everyone down* to ≈0.36 — reported separately
as "strict equalization" — which is the artifact this correction removes.

**6.1c Robustness across sites — the disparate impact is SCARCITY-SPECIFIC (`audit_robustness.json`).**
Rerunning the oracle audit at 4 genuinely different stations (charger count, grid, arrival
rate, dwell profile, fleet): it is **NOT structural**. Budget-worst holds at the **scarce**
sites — reference (30 kW residential) **97%** of days (gap 0.56, PoF 8.8%), highway (45 kW,
high-rate) **78%** (gap 0.27, PoF 6.1%) — but **fades where power isn't binding** — workplace
(long spread dwells) 38% (gap 0.07, **PoF 0%**), shopping (low-traffic) 47% (gap 0.19, PoF 0%).
The **revenue PoF tracks the disparate impact** across sites: it appears iff rationing by tier
is profitable, which requires scarcity. So the honest, scoped claim is **"value-weighted
allocation disadvantages low-income drivers *at power-scarce sites*"** — not everywhere. This
*sharpens* the causal story (scarcity is the driver) and lets a regulator **target** the pricing
rule at sites with positive PoF rather than mandate it universally.

**6.1b Robustness — the aggregate gain is a redistribution premium drivers pay for.**
The mean-satisfaction rise is a consequence of *concave, saturating* charging utility,
not a free lunch. The premium tier drops ~0.94→0.83 (−0.11) while the budget tier gains
~+0.44; across a premium-margin sweep (1.2×–3.0×) these hold and the revenue PoF grows
8%→14%. So the contribution is the **magnitude + the price/scarcity decomposition, not
the sign** (which is the classical implication of concave utility), and "nearly free"
holds only under equal weighting of a satisfied kWh across tiers — a value choice we
state. (`robustness` in `RESULTS.md`.)

**6.2 Which tariff designs remove the disparate impact (rate design).** *Disparity* =
the gap between best- and worst-served tier in **tier-averaged** satisfaction (does a
tier *systematically* lose — a disparate impact), = max−min of the reported group means
by construction. Under the status-quo tiered tariff it is **0.54** (worst tier = budget).
**Income-neutral pricing** removes the price-driven, income-correlated part, cutting it
to **0.04** (**−93%**); the residual is not tier-correlated (day-to-day noise), so it is
not a disparate impact. A **budget-only subsidy does not close it**: the best it reaches
is 0.28, and past parity it **shifts the burden to the mid tier** (we report the starved
tier's identity, since the metric is blind to it). (`rate_design.png`, [`tariffs.py`](chargax/equity/tariffs.py).)

**6.3 The grid-capacity threshold (infrastructure).** A second, independent lever:
enough capacity removes the scarcity that forces rationing. The threshold is
**metric-dependent** and we give both — ≈**50 kW** by max-charge Gini, ≈**70 kW** by the
stricter oracle tier gap. The capacity sweep's 30 kW oracle disparity equals the Section-1
oracle exactly (same days/seed), so the analyses are consistent. (`capacity_threshold.png`.)

**6.4 Scarcity induces inequity.** The profit-blind max-charge policy has Gini ≈0.42
(mean sat 0.71) on the 30 kW grid vs Gini ≈0.31 (mean sat 0.87) at 600 kW.
(`scarcity_contrast.png`.)

**6.5 Second pillar — a learned controller beats the deployed heuristics online, on
efficiency and service (body).** Two RL claims were on the table; only one survived, and
it is an **efficiency/throughput** result, **not** a between-tier equity result.
- **Alive: beats the heuristics on inclusive terms.** The **survivorship check**
  (`check.json`, n=10, 3M steps) compares egal_l050 (λ=0.5) to the heuristics on the
  paper's **inclusive** metrics (rejected=0), *not* survivorship-biased served-sat, and
  *not* our own λ=0 run (confounded). Median vs `max_charge`: **profit 200 vs 186, incl.
  sat 0.595 vs 0.544, within-pop Gini 0.382 vs 0.416, incl. welfare 0.551 vs 0.499,
  rejection 0.213 vs 0.232.** Per-axis win rate: **profit 8/10, incl. sat 8/10, incl.
  welfare 8/10, Gini 7/10, rejection 9/10; 7/10 dominate all four.** It **rejects fewer**
  drivers, so the service win is real, not survivorship. (State fractions, not "dominates"
  — 3 seeds don't.) Worst seed (welfare 0.46) has *normal* rejection: failure mode is
  "didn't charge fully", not "games the metric".
- **The gain needs the demand-aware CONTENT — isolated by three conditions (`ablation.json`,
  `shape.json`).** (1) welfare (content+shape): profit 200 / incl-sat 0.595 / welfare 0.551.
  (2) **monotone** content-free (cumulative energy — no shape, no content): 164 / 0.511 /
  0.466, *below* max_charge. (3) **saturating** content-free (per-connector delivered/const,
  clipped — keeps saturation, drops demand content): **165 / 0.517 / 0.462 — indistinguishable
  from (2), still below max_charge.** So **the saturating shape is NOT sufficient**; the
  gain requires the **demand-aware content** — the satisfaction signal measuring delivered
  vs *each driver's own target* — which lets the controller complete drivers behind on their
  own demand. **Both my and the reviewer's "likely shape" hypothesis were wrong; the
  experiment says content.** Mechanism = a **demand-aware scheduling prior**. (The operative
  content is the demand-awareness, not the tier structure, consistent with the between-tier
  gap not responding to λ.)
- **Still a within-population, not between-tier, effect.** The driver a demand-aware
  potential prioritizes is whoever is behind on its own demand — as likely premium as budget
  — so it doesn't track ability-to-pay and the between-tier gap doesn't move.
- **Dead: the tunable between-tier equity dial (honest negative).** Across λ∈{0,0.5,1}
  (`gate.json`) between-tier disparity is at seed noise (~0.01–0.06), unordered by λ,
  worst-tier identity *roams*; best policy is the **interior** λ=0.5 (pure welfare λ=1 is
  worse + high-variance → needs the profit anchor). The weight is not a between-tier knob.
- **Two equity notions, separate boxes.** Controller → within-pop scheduling/service (real,
  welfare-content-driven). Between-tier disparate impact → offline profit-optimal only →
  the **tariff's** story.

### Path to practice

The route to impact does not require deploying a controller, because the actionable
objects are the **tariff** and the **connection**, not the policy. Decision-makers:
charging network operators, distribution utilities and regulators, site planners.
Decisions informed: (i) a regulator can read that requiring equal satisfaction costs
the operator ~10% revenue and *no* aggregate service, bounding any needed credit;
(ii) a rate designer can read that income-neutral / fully-equalizing pricing closes the
price-driven gap while a poorest-segment credit does not; (iii) a planner can read the
~50 kW capacity threshold. The released oracle + calibration hooks let any of these
actors re-run the analysis on their own arrival data — adoption of the *analysis*, not
a black box.

**6.6 The endogeneity payoff (reject-to-look-fair is closed).** Because admission is a
control decision, a policy could "look fair" by shedding drivers it serves poorly.
Defining utility over the true arrival stream (rejected = 0) closes this: ablating
rejection-counting lets the same policy inflate apparent welfare by ≈0.15, the concrete
payoff of modelling the population as endogenous rather than fixed.

---

## 7. Limitations & honest treatment

- **Oracle correction.** The reported maximin is the **Pareto-efficient two-stage**
  segment maximin; a naive single-epigraph LP levels down and misstates the price of
  fairness (we report the strict-equalization row too, for transparency).
- **The impact axis needs levers, not a controller.** Simulation alone cannot close
  the social-impact axis, which is why the findings are framed around the tariff and
  the connection — levers a decision-maker already controls (Section 6, Path to
  practice) — rather than around deploying the RL policy.
- **Ability-to-pay is a grounded model, not a measured demand curve.** Multipliers
  come from income terciles + the energy-burden gradient with an income elasticity; a
  real operator's own segmentation would refine them.
- **γ.** The telescoping identity is exact only at γ=1; we optimize the undiscounted
  finite-horizon return.
- **Oracle relaxation.** The LP drops the nonlinear charge curve, battery round-trip
  and per-subtree limits, so it is a valid but loose *upper bound* on the admitted
  stream.
- **Group attribution of rejections.** Exact for individual fairness; for grouped
  mode, rejected customers are attributed to segments by the arrival mix (expected
  attribution), stated explicitly.
- **RL is a benchmark baseline (B2), not a method claim.** Within our compute the
  learned policies do not beat max-charge; the online–offline gap is the open problem
  we release the benchmark to study, and multi-seed dispersion is reported.
- **SAFFE.** A faithful Chargax instantiation of "allocate against expected future
  demand," not the original code.

---

## 8. Selected references

**Energy justice, mobility equity, and rate design (outside CS).**
- Sovacool, Dworkin. *Global Energy Justice.* Cambridge University Press, 2016.
- Jenkins et al. *Energy Justice: A Conceptual Review.* Energy Research & Social
  Science, 2016.
- Drehobl, Ross, Ayala. *How High Are Household Energy Burdens?* ACEEE, 2020.
- Bednar, Reames. *Recognition of and Response to Energy Poverty in the US.* Nature
  Energy, 2020.
- Hsu, Fingerman. *Public EV Charger Access Disparities across Race and Income in
  California.* Transport Policy, 2021.
- Khan et al. *Inequitable Access to EV Charging Infrastructure.* The Electricity
  Journal, 2022.
- Borenstein. *The Redistributional Impact of Nonlinear Electricity Pricing.* AEJ:
  Economic Policy, 2012.
- Burger et al. *The Efficiency and Distributional Effects of Alternative Residential
  Electricity Rate Designs.* The Energy Journal, 2020.
- Atkinson. *On the Measurement of Inequality.* J. Economic Theory, 1970.
- Rawls. *A Theory of Justice.* Harvard University Press, 1971.
- Sen. *Development as Freedom.* Oxford University Press, 1999.
- Moulin. *Fair Division and Collective Welfare.* MIT Press, 2003.
- Lee, Li, Low. *ACN-Data: An Open EV Charging Dataset.* ACM e-Energy, 2019.

**Fair RL, online fair division, general-utility RL.**
- Siddique, Weng, Zimmer. *Learning Fair Policies in Multi-Objective RL with
  Average and Discounted Rewards.* ICML 2020. arXiv:2008.07773.
- Mandal, Gan. *Socially Fair Reinforcement Learning.* arXiv:2208.12584.
- Fan et al. *Welfare and Fairness in Multi-objective RL.* arXiv:2212.01382.
- Ju, Ghosh, Shroff. *Achieving Fairness in Multi-Agent MDPs Using RL.* ICLR 2024.
  arXiv:2306.00324.
- Alamdari et al. *Remembering to Be Fair.* ICML 2024.
- Hassanzadeh et al. *Sequential Fair Resource Allocation under an MDP Framework
  (SAFFE).* ICAIF 2023. arXiv:2301.03758.
- Sinclair, Jain, Banerjee, Yu. *Sequential Fair Allocation.* Operations Research
  2022. arXiv:2105.05308.
- Barman, Khan, Maiti. *Universal and Tight Online Algorithms for Generalized-Mean
  Welfare.* AAAI 2022. arXiv:2109.00874.
- Huang, Lee, Shu, Wang. *The Long Arm of Nashian Allocation in Online p-Mean
  Welfare Maximization.* ICALP 2025. arXiv:2504.13430.
- Banerjee, Gkatzelis, Gorokh, Jin. *Online Nash Social Welfare Maximization with
  Predictions.* SODA 2022. arXiv:2008.03564.
- Liu, Hajiesmaili. *Online Fair Allocation of Reusable Resources.* SIGMETRICS 2025.
- Zahavy et al. *Reward is Enough for Convex MDPs.* NeurIPS 2021. arXiv:2106.00661.
- Mutti et al. *Challenging Common Assumptions in Convex RL* (finite trials).
  NeurIPS 2022 / JMLR 2023. arXiv:2202.01511.
- Mo, Walrand. *Fair end-to-end window-based congestion control* (α-fairness).
  IEEE/ACM ToN 2000.
- Ponse et al. *Chargax: A JAX-Accelerated EV Charging Simulator.* arXiv:2507.01522.
