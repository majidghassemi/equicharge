# Chargax: A JAX Accelerated EV Charging Simulator

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)

> [!NOTE]
> Please refer to the submission branch to reproduce results as presented in the paper.
> The main branch ships the bare environment.

---

## 📦 Installation & Quick start

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), it is possible to run a standard PPO implementation with default settings by directly running `uv run main.py`.

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
uv run main.py
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
conda create -n chargax python=3.11
conda activate chargax
pip install -e .

python main.py
```

for CUDA support, additionally run `pip install jax[cuda]`.

---

> [!NOTE]
> The PPO implementation used in the example `train.py` file is different from the one
> used in the paper. As such, required hyperparameters may be different as well.
> Check out the submission branch to reproduce results.


### 🏗️ Customizing the Station Layout

The main function uses a default charging station initialized as follows:

```python
	import jax
	from chargax import Chargax, ChargingStation
	
	station = ChargingStation.init_default_station()
	env = Chargax(station=station)
	
	key = jax.random.PRNGKey(0)
	obs, state = env.reset_env(key)
```

The station layout can easily be changed. The station is a tree of nodes. You compose it from three building blocks:

| Node | Purpose |
|---|---|
| `StationSplitter` | Switchboards, cables, transformers — anything that splits or limits power |
| `EVSE` | A group of physical charger connectors |
| `StationBattery` | On-site battery storage |


#### Defining a custom station

```python
from chargax import ChargingStation, StationSplitter, EVSE, StationBattery

station = ChargingStation(
max_kw_throughput=150.0,  # Grid connection limit
efficiency=1.0,
connections=[
      StationSplitter(
            max_kw_throughput=150.0,
            efficiency=0.995,
            connections=[
            # 4 slow AC chargers (2 per EVSE)
            EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
            EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
            # 1 on-site battery
            StationBattery(
                  capacity_kw=500.0,
                  max_kw_throughput=100.0,
                  efficiency=0.99,
            ),
            ],
      ),
],
)

env = Chargax(station=station)
```

The tree can be nested arbitrarily deep — any StationSplitter can contain other splitters, EVSEs, or batteries.

## ⚙️ Configuring the Environment

#### Changing default data loaders

The default car arrivals, car profiles, and grid prices are configured through default_data_kwargs:

```python
env = Chargax(
    station=station,
    default_data_kwargs={
        "car_profile": "us",               # "eu" (default), "us", "world"
        "user_profile": "residential",      # "highway" (default), "residential", "workplace", "shopping"
        "average_cars_per_day": "low",      # "low", "medium", "high" (default) or int
        "grid_price_dataset": "2023_NL",    # Dataset identifier for price data
        "grid_sell_margin": -0.05,          # Sell price offset from buy price
    },
)
```

#### 🔌 Injecting Custom Callables

For full control, replace any of the data-generating functions directly. Each callable receives a PRNG key and the current environment state:

##### Custom grid pricing

```python
def my_buy_price(state):
    """Time-of-use pricing: expensive during the day, cheap at night."""
    hour = (state.timestep * env.minutes_per_timestep) / 60.0
    return jnp.where((hour >= 8) & (hour < 20), 0.30, 0.10)

def my_sell_price(state):
    return my_buy_price(state) - 0.05

env = Chargax(
    station=station,
    get_grid_buy_price=my_buy_price,
    get_grid_sell_price=my_sell_price,
)
```

##### Custom car arrivals

```python
def my_num_arriving(key, state):
    """Constant arrival rate of 2 cars per timestep."""
    return 2

env = Chargax(
    station=station,
    get_num_cars_arriving=my_num_arriving,
)
```

You can similarly override `get_new_cars_arriving` (generates EVSE entries for new cars) and `get_cars_departing` (determines which cars leave).


## ⚖️ Research extension: equitable charging under scarcity — `chargax.equity`

> **Target venue: ACM FAccT 2027** (socio-technical audit).
> Focus area: evaluations and evaluation practices (audits and evaluations for
> fairness and justice).

This repository includes a research extension on **equitable EV charging access
under grid scarcity**. Under a hard grid limit a revenue-maximizing controller
sends scarce power to whoever pays the most and leaves lower-income drivers
undercharged or rejected. The [`chargax.equity`](chargax/equity) package reframes
charging as **online sequential fair allocation over an endogenous, streaming
customer population**, and — using an exact offline oracle on a real-data-calibrated
environment — turns this into an *actionable characterization* for operators,
utilities and regulators.

Operators price by **tier** (memberships, dynamic pricing, fast-vs-slow); drivers sort
into tiers by budget in a way that **correlates with income**, so revenue-optimal
allocation under scarcity is a **disparate impact** on lower-income drivers, not
deliberate income discrimination.

**The headline findings (all reproducible, RL-free):**
1. **The price of fairness is a revenue cost, not a service cost** — but a
   redistribution the premium tier pays for. Equalizing tier satisfaction costs ~9.5%
   of margin-weighted revenue while it *raises* mean satisfaction and doubles the
   worst tier; the premium tier gives up ~0.11 (a consequence of concave saturating
   utility, robust across price gradients). The released oracle uses a Pareto-efficient
   two-stage maximin (the naive single-epigraph one "levels down" — a solver artifact).
   **Scarcity-specific:** a 4-site audit (`audit_robustness.json`) shows this holds at
   power-scarce sites (reference 97% budget-worst, highway 78%) and *fades* where power
   isn't binding (workplace/shopping, revenue PoF ≈0) — so the claim is scoped to
   power-scarce sites, and a regulator can target the fix there.
2. **Only income-neutral pricing removes the disparate impact** (tier gap 0.54→0.04,
   −93%). A subsidy to the poorest tier alone does *not* close it — it **shifts the
   burden to the middle tier**. See [`tariffs.py`](chargax/equity/tariffs.py).
3. **Grid capacity is a second, independent lever**, threshold metric-dependent
   (~50 kW by Gini, ~70 kW by the oracle gap).

> **On the RL (second pillar — an efficiency/service result, NOT equity).** A
> welfare-shaped learned controller (λ=0.5, 3M steps), judged on the paper's **inclusive**
> metrics (rejected=0) and against the deployed heuristics, beats the strongest heuristic
> (`max_charge`) at the median on **profit (200 vs 186), inclusive satisfaction (0.595 vs
> 0.544), within-pop Gini (0.382 vs 0.416), inclusive welfare (0.551 vs 0.499), rejection
> (0.213 vs 0.232)**. Per-axis win rate: **profit 8/10, incl. sat 8/10, incl. welfare
> 8/10, Gini 7/10, rejection 9/10; 7/10 dominate all four** (state fractions — 3 seeds
> don't). It rejects *fewer* drivers, so the win is not survivorship
> (`experiments/results/check.json`). **Mechanism — isolated by three conditions:** neither
> a **monotone** content-free potential (`ablation.json`) nor a **saturating** content-free
> one (`shape.json`) of matched magnitude reproduces the gain — both collapse below
> max_charge (164 and 165 vs welfare's 200). So it is **neither generic dense reward nor a
> mere saturating shape**: the gain needs the **demand-aware content** — the satisfaction
> signal measuring delivered vs *each driver's own target* — acting as a scheduling prior.
> The Gini gain (within-population, *not* tier-equalization) requires this content. Honest
> negative: the equity weight is **not** a tunable between-tier fairness dial (disparity
> ~0.01–0.06 for every λ, interior optimum). The between-tier
> disparate impact is the tariff's story (offline), not the controller's.

```python
from chargax import ChargingStation
from chargax.equity import EquiChargax, segments as SEG

station = ChargingStation.init_default_station()
# Egalitarian (Rawlsian) welfare across affordability-grounded ability-to-pay segments:
env = EquiChargax(station=station, welfare_alpha=0.0, welfare_outer="rawlsian",
                  lam=1.0, n_groups=3, group_probs=SEG.GROUP_PROBS,
                  price_by_group=SEG.PRICE_BY_GROUP)
# ... train any jaxnasium agent on `env` exactly as with Chargax.
```

* **What it adds:** social welfare functions + a *bounded sufficient-statistic* state
  augmentation ([`welfare.py`](chargax/equity/welfare.py)), the
  [`EquiChargax`](chargax/equity/fair_env.py) environment (rejection-aware arrivals,
  departure-realized utility, dense potential-shaped welfare reward at γ=1),
  inequality [`metrics`](chargax/equity/metrics.py), myopic/SAFFE-style
  [`baselines`](chargax/equity/baselines.py), a two-stage clairvoyant LP
  [`oracle`](chargax/equity/oracle.py) with revenue accounting, affordability-grounded
  [`segments`](chargax/equity/segments.py), the rate-design
  [`tariffs`](chargax/equity/tariffs.py) analysis, and real-data
  [`data_calibration`](chargax/equity/data_calibration.py) provenance + ACN-Data hook.
* **Data provenance** (what is real vs. modelled): see [`docs/DATA.md`](docs/DATA.md).
* **Paper** (the socio-technical audit): [`paper/facct27_equicharge.tex`](paper/facct27_equicharge.tex).
  Submission checklist and change log: [`docs/FACCT_SUBMISSION.md`](docs/FACCT_SUBMISSION.md).
* **Reproduce:**
  ```bash
  # RL-free core (every offline finding above); fast, CPU-only:
  python -m experiments.run_experiments --oracle-only
  # Full study incl. multi-seed learning benchmark (GPU-friendly):
  python -m experiments.run_experiments                 # 8 seeds (default)
  python -m experiments.run_experiments --quick         # fast smoke run
  # Regenerate the audit results and the paper figures:
  python -m experiments.audit_mechanism      # canonical box-1, mechanism, elasticity
  python -m experiments.audit_robustness     # generalization across sites
  python -m experiments.audit_grounding      # A1 grounding sweep
  python -m experiments.audit_levers         # lever cross-effects
  python -m experiments.plot_mechanism       # all FAccT figures -> results/figures/
  python -m pytest tests/test_equity.py -q   # tests
  ```


## 📑 Citing

```bibtex
@misc{ponse2025chargaxjaxacceleratedev,
      title={Chargax: A JAX Accelerated EV Charging Simulator}, 
      author={Koen Ponse, Jan Felix Kleuker, Aske Plaat, Thomas Moerland},
      year={2025},
      eprint={2507.01522},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.01522}, 
}
```
