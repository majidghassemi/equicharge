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

The income terciles and the elasticity are taken from public sources:

* Household income terciles (approximate medians, US ACS): lower ~US$30k,
  middle ~US$67k, upper ~US$130k.
* An income elasticity of residential electricity / transport-energy demand of
  ``eta ~ 0.6`` (a standard mid-range estimate; e.g. Schulte & Heindl 2017 for
  residential energy; Zhou & Teng 2013 review).

With these, the derived tier margins are approximately ``(0.6, 1.0, 1.5)`` -- the
values used in the experiments -- now *motivated* rather than assumed. The companion
energy-burden gradient corroborates the direction: low-income households spend
~8.6% of income on energy versus ~3% for higher-income households (ACEEE 2020
"How High Are Household Energy Burdens?"; US DOE LEAD tool), so lower-income drivers
both sort into lower-margin tiers and have materially lower marginal willingness to
pay for discretionary charging.

Segment membership is a **price tier**, exogenous and independent of battery capacity;
the operator never observes or acts on income. Income enters only through the
empirical correlation between tier and income, which is what makes the revenue-optimal
allocation a *disparate-impact* problem rather than deliberate discrimination.

References
----------
ACEEE (2020). How High Are Household Energy Burdens?
US DOE. Low-Income Energy Affordability Data (LEAD) Tool.
Schulte & Heindl (2017). Price and income elasticities of residential energy demand
    in Germany. Energy Policy.
Borenstein (2012). The Redistributional Impact of Nonlinear Electricity Pricing.
"""

from __future__ import annotations

# Public affordability inputs (see module docstring for sources).
INCOME_TERCILES = (30_000.0, 67_000.0, 130_000.0)  # lower / middle / upper (US$)
INCOME_ELASTICITY = 0.6  # sub-proportional WTP for a quasi-necessity

# Energy-burden gradient (energy cost / income), low vs. high income households.
ENERGY_BURDEN = (0.086, 0.045, 0.030)  # ACEEE 2020 / DOE LEAD (approximate)

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
