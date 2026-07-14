r"""Differentiated-pricing tiers that correlate with income (the disparate-impact model).

**What the operator actually does.** Real charging networks do *not* infer a driver's
income and ration power by it -- that would be operationally strange and legally
fraught. What they *do* run is **differentiated pricing**: membership / subscription
tiers (e.g. pay-as-you-go vs. a monthly plan), dynamic and congestion pricing, and
product tiers (DC fast vs. Level 2) that carry different per-kWh margins. Drivers
sort into these tiers by budget, and tier membership is well documented to
**correlate with income** (lower-income drivers cluster in the lower-margin tiers and
are the most reliant on public rather than home charging). We model each segment as
such a **price tier**, characterized only by the per-kWh margin the operator earns
from it.

**Why this is an equity problem.** Given tiered margins, a revenue-maximizing
allocation under scarcity mechanically directs scarce power to the higher-margin
tiers. Because tier correlates with income, the *effect* is that lower-income drivers
are systematically under-served -- a **disparate impact**, not an act of deliberate
income discrimination by the operator. This is precisely the distributional-justice
concern (Sovacool & Dworkin; Jenkins et al.) that the released oracle makes
quantitative, and the policy levers we study (income-neutral pricing, a low-income
tariff, grid capacity) act on the *structure* that produces the disparate impact
rather than on any operator intent.

The per-segment margin multipliers are grounded in a real affordability structure so
the tiers are not arbitrary. We keep segment membership **exogenous and independent
of battery capacity**, so nothing rests on inferring a protected attribute at
allocation time.

Construction
------------
We segment arriving customers into household-income terciles and set each segment's
price weight to its *willingness to pay per kWh relative to the median household*.
Because charging is a quasi-necessity, willingness to pay grows sub-proportionally
with income: we apply an income elasticity ``eta`` (< 1) to the income ratio,

    weight_g = (income_g / income_median) ** eta.

The income terciles and the elasticity trace to primary sources (verified 2026):

* Household income terciles, US ACS 2024 1-year (Census tables B19013/B19080).
  The US median household income is US$81,604, and the income terciles are bounded
  at roughly US$54k and US$122k. We use representative within-tercile incomes of
  ~US$35k (lower), US$82k (~the median, middle), and US$161k (upper); the middle
  value matches the ACS-2024 median. State medians span ~US$54k (Mississippi) to
  ~US$108k (District of Columbia), a ~+-30% regional band the finding is robust to.
* An income elasticity of residential-energy / transport-fuel demand of ``eta ~ 0.6``.
  Residential energy is empirically a *necessity* (income elasticity strictly in
  (0,1)) across studies; the defensible interval is ~0.2-0.9 with a central cluster
  ~0.3-0.6 (Schulte & Heindl 2017 report ~0.40 for German households; Espey & Espey
  2004 meta-analysis ~0.28 short-run to ~0.97 long-run; Zhou & Teng 2013 ~0.14-0.33;
  Burke & Csereklyei 2016; Huntington et al. 2017/EIA ~0.39-0.50; the gasoline proxy
  Espey 1998 ~0.47-0.88, with Havranek & Kokes 2015 giving a bias-corrected lower
  ~0.10-0.23). No EV-charging-specific elasticity exists, so ``eta`` is proxied from
  these. The A1 grounding sweep (``experiments/audit_grounding.py``) shows the
  disparate impact is *invariant* across this entire range, so the exact value is not
  load-bearing.

With these, the derived tier margins are ``(0.6, 1.0, 1.5)`` -- the values used in the
experiments -- now *grounded* rather than assumed. The energy-burden gradient
corroborates the direction: low-income households (<=200% of the federal poverty
level) spend a median 8.1% of income on home energy versus 2.3% for higher-income
households, a ~3.5x gap (national median 3.1%; Black households 4.2% vs white 2.9%;
Drehobl, Ross & Ayala 2020, from the 2017 American Housing Survey; corroborated ~6%
vs ~2% by the US DOE LEAD tool). So lower-income drivers both sort into lower-margin
tiers and have materially lower marginal willingness to pay for discretionary charging.

Why tier is an *observable* proxy, not an abstract multiplier. The low-margin tier is
the public-charging-reliant tier, and reliance on public rather than cheap home
charging tracks housing tenure and dwelling type, which track income and race. Home
charging covers the large majority of EV charging today, but even at full
electrification ~25% of EVs are projected to lack it (Ge et al. 2021, NREL). Access splits by tenure
(81% of owner-occupied vs 39% of renter-occupied units have a garage/carport; DOE EERE
2018 from AHS 2017) and dwelling type (single-family ~93% vs apartments ~46% home-
charging in 2020, falling to ~80% vs ~30% by 2030; Bauer et al. 2021, ICCT), and by
income (lower-income communities 83%->59% vs 88%->70% general, 2020->2030; Bauer et al.
2021). It also splits by race: Black/Hispanic-majority California block groups have
~0.7x the odds of any public charger and ~0.5x of a publicly funded one (Hsu &
Fingerman 2021); nationally, disadvantaged communities have 64% fewer chargers per
capita, 73% fewer for multi-dwelling-unit renters (Yu et al. 2025), and 63% of Black
vs 30% of white households are renters (Lou et al. 2024). Those without home charging
pay a ~2-3x per-kWh premium (residential ~17 c/kWh, EIA 2024-25, vs public DC-fast
~40 c/kWh, Borlaug et al. 2026). So "who is in the budget tier" is an empirical
claim about real charging conditions, not a modeling parameter.

Segment membership is a **price tier**, exogenous and independent of battery capacity;
the operator never observes or acts on income. Income enters only through the
empirical correlation between tier and income, which is what makes the revenue-optimal
allocation a *disparate-impact* problem rather than deliberate discrimination.

References
----------
Bauer, Hsu, Nicholas & Lutsey (2021). Charging Up America. ICCT.
Borlaug et al. (2026). Economics of EV corridor fast charging in the US. Adv. Appl. Energy.
Drehobl, Ross & Ayala (2020). How High Are Household Energy Burdens? ACEEE (AHS 2017).
Espey & Espey (2004). Turning on the Lights: A Meta-Analysis of Residential Electricity
    Demand Elasticities. J. Agric. Appl. Econ.
Ge, Simeone, Duvall & Wood (2021). There's No Place Like Home. NREL/TP-5400-81065.
Hsu & Fingerman (2021). Public EV Charger Access Disparities across Race and Income in
    California. Transport Policy.
Lou, Shen, Niemeier & Hultman (2024). Income and racial disparity in household publicly
    available EV infrastructure accessibility. Nature Communications.
Schulte & Heindl (2017). Price and income elasticities of residential energy demand in
    Germany. Energy Policy.
US Census Bureau (2025). ACS 2024 1-year, tables B19013/B19080.
US DOE. Low-Income Energy Affordability Data (LEAD) Tool.
Yu et al. (2025). Equity and reliability of public EV charging stations in the US.
    Nature Communications.
Borenstein (2012). The Redistributional Impact of Nonlinear Electricity Pricing.
"""

from __future__ import annotations

# Public affordability inputs (see module docstring for sources; verified 2026).
# Representative within-tercile household incomes, ACS 2024 1-year (median US$81,604;
# tercile boundaries ~US$54k / US$122k). The middle value is the ACS-2024 median.
INCOME_TERCILES = (35_000.0, 82_000.0, 161_000.0)  # lower / middle / upper (US$)
INCOME_ELASTICITY = 0.6  # sub-proportional WTP; central of the ~0.2-0.9 empirical range

# Energy-burden gradient (home-energy cost / income): low-income (<=200% FPL) 8.1%,
# national median 3.1%, higher-income 2.3% (Drehobl, Ross & Ayala 2020, AHS 2017).
ENERGY_BURDEN = (0.081, 0.031, 0.023)

SEGMENT_NAMES = ("budget", "mid", "premium")


def derive_price_multipliers(
    income_terciles: tuple = INCOME_TERCILES,
    elasticity: float = INCOME_ELASTICITY,
    round_to: int = 1,
) -> tuple:
    """Willingness-to-pay-per-kWh multipliers from income terciles + elasticity.

    Normalized so the middle (median) segment is 1.0. See module docstring.
    """
    median = income_terciles[1]
    weights = tuple((y / median) ** elasticity for y in income_terciles)
    if round_to is not None:
        weights = tuple(round(w, round_to) for w in weights)
    return weights


#: Derived, affordability-grounded ability-to-pay multipliers (budget/mid/premium).
PRICE_BY_GROUP = derive_price_multipliers()

#: Equal-frequency arrival mix across the three income terciles (a modelling
#: choice made explicit; can be replaced with a region's actual income distribution).
GROUP_PROBS = (1 / 3, 1 / 3, 1 / 3)


def segment_summary() -> list:
    """Human-readable table of the grounded segments (for docs / logging)."""
    mults = PRICE_BY_GROUP
    return [
        {
            "segment": SEGMENT_NAMES[g],
            "income": INCOME_TERCILES[g],
            "energy_burden": ENERGY_BURDEN[g],
            "price_multiplier": mults[g],
        }
        for g in range(len(SEGMENT_NAMES))
    ]


if __name__ == "__main__":
    print("Affordability-grounded ability-to-pay segments:")
    print(f"  income elasticity eta = {INCOME_ELASTICITY}")
    for row in segment_summary():
        print(
            f"  {row['segment']:8s} income=${row['income']:>7,.0f} "
            f"burden={row['energy_burden']:.1%} weight={row['price_multiplier']:.2f}x"
        )
