# ATTRIBUTION DECOMPOSITION ANALYSIS - EXPLANATION

## What Problem Are We Solving?

You want to understand **WHY** hourly rainfall extremes are changing in future climate projections.

Specifically: Is it just because there's more rain overall (daily totals increase), 
OR is the way rain falls within a day also changing (more concentrated bursts vs. steadier rain)?

## The Scientific Question

**Can changes in hourly rainfall extremes be explained by changes in daily totals alone, 
or is the temporal structure of rainfall within days also evolving?**

This matters because:
- If it's just daily totals → simple scaling relationships work
- If structure is changing → complex sub-daily dynamics are evolving
- Different implications for flash flooding, erosion, infrastructure design

---

## The Three Datasets You're Using

### 1. PRESENT CLIMATE (from observations)
**What it shows:** Hourly percentiles (P50, P90, P99, etc.) in current climate
**File:** `percentiles_and_significance_1H_mann_whitney_seqio.nc`

### 2. FUTURE CLIMATE (from climate model)
**What it shows:** Hourly percentiles in future climate
**File:** Same file as above - contains both present and future

### 3. SYNTHETIC FUTURE (the counterfactual)
**What it shows:** What hourly percentiles WOULD BE if:
- We take future daily totals (from climate model)
- Apply present-day structure (how rain distributes within days)
**File:** `synthetic_future_1H_from_D_confidence.nc`

**This is the key insight:** By creating a "fake" future where ONLY daily totals 
change but structure stays the same, we can isolate the effect of each.

---

## The Decomposition Math

For each percentile and grid point:

```
TOTAL CHANGE = Future - Present
             = What actually happens to hourly extremes

DAILY TOTAL EFFECT = Synthetic - Present  
                   = Change if ONLY daily totals changed
                   = What we CAN explain with structure unchanged

STRUCTURAL CHANGE EFFECT = Future - Synthetic
                         = The residual (what's left over)
                         = What we CANNOT explain without structure changing
```

### Key Property (Conservation):
```
TOTAL CHANGE = DAILY EFFECT + STRUCTURAL EFFECT
```

This should always be true (within numerical precision).

---

## What the Script Does - Step by Step

### STEP 1: Load Data
- Opens your percentiles file (present + future)
- Opens your synthetic future file
- Checks that dimensions match

### STEP 2: Align Datasets
- Matches percentiles between files (e.g., P50 in obs → quantile 0.5 in synthetic)
- Selects which bootstrap estimate to use from synthetic data (default: median)

### STEP 3: Calculate Decomposition
For each percentile:

```python
# Get the three values
present = observations[percentile]
future = observations[percentile] 
synthetic = synthetic_data[percentile]

# Decompose
change_total = future - present              # What happens
change_daily = synthetic - present           # Daily totals effect
change_structural = future - synthetic       # Structural effect

# Calculate fractions
fraction_daily = change_daily / change_total
fraction_structural = change_structural / change_total
```

### STEP 4: Classify Grid Points
Each grid point gets classified:

- **"daily_dominated"**: fraction_daily > 0.8
  → Changes mostly explained by daily totals
  
- **"structural_dominated"**: -0.2 < fraction_daily < 0.2
  → Changes mostly from structural evolution
  
- **"mixed"**: 0.2 < fraction_daily < 0.8
  → Both factors matter
  
- **"amplifying"**: fraction_daily > 1.2 or < -0.2
  → Daily and structural effects amplify each other

### STEP 5: Calculate Statistics
For each percentile, compute:
- Mean changes (spatial averages)
- Variance of each component
- Correlation between daily and structural effects
- Fraction of domain where each mechanism dominates

### STEP 6: Save Results
Creates NetCDF file with:
- All three change components (total, daily, structural)
- Fractions showing attribution
- Percentage changes
- Classification labels
- Summary statistics as global attributes

---

## How to Interpret the Results

### Example Output:

```
P90: Total change = +2.5 mm/h
     Daily totals contribute:   +60.0% (+1.5 mm/h)
     Structural changes add:    +40.0% (+1.0 mm/h)
     --> STRUCTURAL CHANGES ARE IMPORTANT!
```

**Interpretation:**
- Hourly P90 increases by 2.5 mm/h in future
- If only daily totals changed (structure constant): +1.5 mm/h
- But structure is ALSO changing, adding another +1.0 mm/h
- Conclusion: Can't ignore structural evolution!

### Another Example:

```
P99: Total change = +5.0 mm/h
     Daily totals contribute:   +90.0% (+4.5 mm/h)
     Structural changes add:    +10.0% (+0.5 mm/h)
     --> Changes well-explained by daily totals alone
```

**Interpretation:**
- Hourly P99 increases by 5.0 mm/h
- Almost all (90%) explained by more rain overall
- Structure barely changing (only +0.5 mm/h)
- Conclusion: Simple scaling from daily totals works here

---

## What Structural Changes Mean Physically

### Positive Structural Effect (Future > Synthetic)
**Physical meaning:** Rainfall becoming MORE concentrated within days
- Higher Gini coefficient
- More intense bursts, longer dry periods within wet days
- Example: Instead of 24mm spread over 6 hours → 24mm in 2 hours

### Negative Structural Effect (Future < Synthetic)
**Physical meaning:** Rainfall becoming LESS concentrated within days
- Lower Gini coefficient  
- More evenly distributed
- Example: Instead of 24mm in 2 hours → 24mm spread over 6 hours

### Near-zero Structural Effect
**Physical meaning:** Temporal structure not changing
- Daily totals increasing, but distribution within days stays similar
- Simple scaling relationship applies

---

## Key Variables in Output File

### Absolute Changes (mm/h or mm/day):
- `change_total`: What actually happens
- `change_daily_effect`: Effect of daily total changes only
- `change_structural_effect`: Effect of structural evolution only

### Attribution Fractions (0-1, can be >1 or negative):
- `fraction_daily`: Fraction explained by daily totals
  - If = 1.0 → all explained by daily totals
  - If = 0.5 → each contributes 50%
  - If = 0.0 → all from structural changes
  - If > 1.0 → effects amplify each other

- `fraction_structural`: Fraction from structural changes
  - Always equals (1 - fraction_daily) when no amplification

### Percentage Changes (%):
- `pct_change_total`: 100 * (Future - Present) / Present
- `pct_change_daily`: Contribution from daily totals
- `pct_change_structural`: Contribution from structure

### Classification:
- `classification`: String labels for each grid point
  - "daily_dominated", "structural_dominated", "mixed", "amplifying"

---

## Spatial Patterns to Look For

### Uniform Attribution:
If most of domain shows "daily_dominated" → simple scaling applies everywhere

### Spatial Variability:
If mountains show "structural_dominated" but plains show "daily_dominated" →  
Different physical processes at work in different regions

### Systematic Patterns:
- Coastal vs. inland
- Elevation-dependent
- Latitude gradients

### Amplification Zones:
Grid points with fraction > 1.2 or < -0.2 →  
Daily and structural effects working together (or against each other)

---

## How This Answers Your Question

**Before this analysis:**
"Hourly P99 increases by 5 mm/h in the future"
→ Descriptive, but why?

**After this analysis:**
"Hourly P99 increases by 5 mm/h. Of this:
 - 4 mm/h from increased daily totals (80%)
 - 1 mm/h from more concentrated rainfall within days (20%)
 - Structure is evolving, but daily totals dominate"
→ Mechanistic understanding, can test hypotheses

---

## Next Steps After Running

1. **Map the fractions:**
   - Plot `fraction_daily` spatially
   - Identify regions where structure matters most

2. **Compare across percentiles:**
   - Does P50 show different attribution than P99?
   - Are extremes more/less sensitive to structural changes?

3. **Link to physical processes:**
   - High structural effect in mountains → changing convection?
   - Coastal structural changes → sea breeze timing shifts?

4. **Test hypotheses:**
   - "I think structure changes because convection intensifies"
   - Look at Gini coefficient trends from original data
   - Check if high structural effect correlates with high Gini changes

5. **Validate:**
   - Bootstrap confidence intervals tell you uncertainty
   - Check if results robust to different percentile thresholds

---

## Common Pitfalls and What They Mean

### Problem: Fraction > 1.0 everywhere
**Meaning:** Daily and structural effects amplify each other
**Check:** Are both positive? This is physically reasonable (both increasing concentration)

### Problem: Fraction = NaN in many places
**Meaning:** Total change ≈ 0 (dividing by zero)
**Solution:** Add threshold: only analyze where |change_total| > 0.1 mm

### Problem: Negative fractions
**Meaning:** Daily and structural effects have opposite signs
**Example:** Daily totals increase (+) but structure spreads rain out (-)
**Interpretation:** Competing mechanisms

### Problem: Fractions don't add to 1.0
**Check your math!** By definition:
fraction_daily + fraction_structural = 1.0 (always)

---

## Summary - The Big Picture

This script answers: "Why are hourly rainfall extremes changing?"

By comparing:
1. What actually happens (Future)
2. What would happen with structure unchanged (Synthetic)
3. What we started with (Present)

We can say:
- "X% of the change comes from more rain overall"
- "Y% comes from how rain falls within days changing"

This is essential for:
- Understanding physical mechanisms
- Downscaling climate models
- Designing infrastructure
- Assessing flash flood risk

The key innovation is the SYNTHETIC dataset - it's your "control experiment"
that lets you isolate causal factors.
