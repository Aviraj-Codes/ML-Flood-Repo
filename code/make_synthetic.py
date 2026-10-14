import numpy as np, pandas as pd
rng = np.random.default_rng(0)
rows = []
for year, n in [(2015, 4000), (2005, 1500)]:
    x = rng.uniform(0, 30000, n); y = rng.uniform(0, 30000, n)
    elev = rng.gamma(2, 3, n); slope = rng.gamma(2, 1, n)
    twi = 10 - elev * 0.5 + rng.normal(0, 1, n)
    rain = rng.normal(900 if year == 2015 else 700, 100, n)
    lulc = rng.uniform(0, 1, n); soil = rng.integers(0, 4, n)
    dist = rng.uniform(0, 5000, n)
    logit = -2 + 0.8 * (elev < 4) + 0.3 * (twi > 6) + 0.002 * (rain - 800) - 0.0002 * dist + rng.normal(0, 0.5, n)
    flood = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    rows.append(pd.DataFrame(dict(x=x, y=y, rainfall=rain, elevation=elev, slope=slope, twi=twi,
        lulc_urban_frac=lulc, soil_class=soil, dist_river_m=dist, flood=flood, event_year=year)))
pd.concat(rows, ignore_index=True).to_csv("synthetic.csv", index=False)
print("ok", sum(len(r) for r in rows))
