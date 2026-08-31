# Using the quantile forecasts

The forecasts ship as two icechunk stores. This is what is in them, what the numbers mean, and
how to get answers out of them.

```
products/quantiles/hindcast_qf.icechunk    2000 -> 2005, 2010, 2015, 2020   (49 GB)
products/quantiles/forecast_qf.icechunk    2020 -> 2025, 2030, 2035, 2040   (52 GB)
```

---

## 1. The one idea: every pixel holds a curve, not a number

An ordinary forecast gives you one number per pixel. This one gives you **64 numbers**, and
together they describe the complete range of outcomes the model thinks possible.

The 64 numbers answer 64 versions of the same question:

> *"Give me a value that the true 2040 outcome will fall below with probability u."*

Ask with `u = 0.5` and you get the middle outcome. Ask with `u = 0.975` and you get a value the
truth should stay under 97.5% of the time. Ask with `u = 0.999` and you get the 1-in-1000 case.
Line all 64 answers up and you have the whole forecast.

That list is called a **quantile function**. The store's `quantile` coordinate holds the `u`
values; the data holds the answers.

**Why this matters.** A single number cannot say *"almost certainly nothing will happen here,
but there is a 1-in-200 chance of a road"*. A curve can, and in remote land that is the only
statement worth making.

---

## 2. The three questions you will actually ask

### "What is the best single guess?"

Read the curve at `u = 0.5`. That is the median. Note the published `central` COG is the
**mean** rather than the median — for skewed pixels they differ, and the mean can even fall
outside the 95% range. Use the median from this store when you want a percentile.

### "How uncertain is it?"

Read `u = 0.025` and `u = 0.975`. The gap between them is the **95% range**: the truth should
land inside it 19 times in 20. Those two levels are stored exactly, so they match the published
`lower` and `upper` COGs to the last bit.

A wide gap means the model sees several futures for that pixel. A narrow gap means it does not.
Mapping that width is the fastest way to see where the forecast is confident.

### "What is the chance of more than X?"

Run the curve backwards. Find where `X` sits on it, read off the `u` there, and the chance of
exceeding `X` is `1 − u`.

Example, for a pixel on the edge of Nairobi in 2020:

```
Q(0.025) = 0.828      Q(0.5) = 0.887      Q(0.975) = 0.970
P(HM > 0.60) = 0.9998                     almost certain to exceed 0.60
```

---

## 3. Reading uncertainty properly

**Uncertainty grows with lead time, always.** The model is built so a twenty-year range can
never be narrower than a five-year one. The same Nairobi pixel:

| forecast for | 95% range | width |
|---|---|---|
| 2005 | 0.836 – 0.893 | 0.057 |
| 2010 | 0.828 – 0.923 | 0.095 |
| 2015 | 0.827 – 0.948 | 0.121 |
| 2020 | 0.828 – 0.970 | 0.142 |

The bottom of the range barely moves and the top climbs. That is the shape almost everywhere:
land rarely becomes *less* modified, so the uncertainty is nearly all on the upside.

**The interesting pixels are lopsided.** In remote country a typical pixel looks like this:

```
Q(0.025) = 0.000     Q(0.5) = 0.000     Q(0.975) = 0.004     Q(0.999) = 0.065
```

The middle three say "nothing happens". The last one says there is a 1-in-1000 chance of real
development. **Do not summarise such a pixel by its median or its 95% range** — both are
essentially zero and both hide the only thing worth knowing. Ask about the tail directly.

**Width is not comparable across HM levels.** A pixel already at 0.9 cannot go far up; a pixel
at 0.1 can. Compare widths within similar starting values, or compare the probability of
crossing a threshold instead, which is on a common scale everywhere.

---

## 4. Things that will trip you up

**Values are stored as integers.** Multiply by `scale` (3.0518509e-05) to get Human
Modification. Anything equal to `sentinel` (−32768) is ocean or unpredicted — mask it before
scaling, not after.

**`ds.quantile` is not the coordinate.** It resolves to xarray's own `.quantile()` method. Use
`ds["quantile"]`. Selecting with `.sel(quantile=0.5)` works normally.

**Latitude descends.** The grid runs north to south, so a slice reads `slice(north, south)` —
`slice(-1.0, -1.6)`, not `slice(-1.6, -1.0)`.

**Pixels are 0.009°, about 1 km.** The grid is 17111 × 40000, EPSG:4326.

**`time` is the target year, and the hindcast has one origin.** `base_year` and `horizon` ride
along the `time` axis so you can always see what a forecast was made from and how far ahead it
looks.

**Chunks are (1 time, all 64 levels, 512, 512).** Reading one pixel's whole curve is a single
chunk. Reading one quantile level across a wide area still pulls all 64 levels off disk, so
prefer to select an area first and a level second.

---

## 5. Code

### Open a store

```python
import icechunk, xarray as xr

repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("products/quantiles/forecast_qf.icechunk"))
ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)

qf    = ds["quantile_forecast"]
SCALE = qf.attrs["scale"]        # 3.0518509e-05
SENT  = qf.attrs["sentinel"]     # -32768  (ocean / no prediction)
u     = ds["quantile"].values    # the 64 probability levels
```

### One pixel's whole forecast

```python
px = qf.sel(time=2040, latitude=-1.29, longitude=36.82, method="nearest").load()
q  = (px.where(px != SENT) * SCALE).values     # 64 outcomes, low to high
```

### Plot that forecast two ways

```python
import matplotlib.pyplot as plt, numpy as np

# as a curve: "chance the outcome is below this value"
plt.plot(q, u); plt.xlabel("Human Modification"); plt.ylabel("probability below")

# as bars: "chance of landing in each band"
edges = np.linspace(q.min(), q.max(), 25)
mass  = np.diff(np.interp(edges, q, u))
plt.bar(edges[:-1], mass, width=np.diff(edges), align="edge")
```

### Any quantile, and any threshold

```python
np.interp(0.90, u, q)          # the 90th-percentile outcome
1 - np.interp(0.30, q, u)      # the chance of exceeding HM = 0.30
```

### An uncertainty map

```python
box = qf.sel(time=2040, latitude=slice(-1.0, -1.6), longitude=slice(36.6, 37.2))
lo  = box.sel(quantile=0.025, method="nearest").where(lambda x: x != SENT) * SCALE
hi  = box.sel(quantile=0.975, method="nearest").where(lambda x: x != SENT) * SCALE

(hi - lo).plot()               # width of the 95% range
```

### How uncertainty grows with lead time

```python
col = (qf.sel(latitude=-1.29, longitude=36.82, method="nearest")
         .where(lambda x: x != SENT) * SCALE).load()

lo  = col.sel(quantile=0.025, method="nearest")
mid = col.sel(quantile=0.500, method="nearest")
hi  = col.sel(quantile=0.975, method="nearest")

plt.fill_between(ds["time"], lo, hi, alpha=0.3)
plt.plot(ds["time"], mid)
```

### Where might change appear?

```python
BASE = 0.05                    # today's HM for the area, or read the 2020 raster
box  = qf.sel(time=2040).sel(latitude=slice(-1.0, -1.6), longitude=slice(36.6, 37.2))
box  = box.where(box != SENT) * SCALE

# chance each pixel exceeds BASE + 0.05, from the two levels that bracket it
p_exceed = 1 - xr.apply_ufunc(
    np.interp, BASE + 0.05, box, xr.DataArray(u, dims="quantile"),
    input_core_dims=[[], ["quantile"], ["quantile"]], vectorize=True)
p_exceed.plot()
```

---

## 6. What the numbers are worth

From the global out-of-sample evaluation (`docs/dist_global_scorecard.md`):

- The stated probabilities hold up. When the model says 3%, it happens about 3% of the time,
  across two orders of magnitude and 17.7 million pixels.
- The 95% range is honest at +5 and +10 years (covers 96%) and **too narrow at +15 and +20**
  (covers 88% against a nominal 95%). Treat long-horizon ranges as a lower bound on uncertainty.
- The deep lower tail is thin: outcomes below `Q(0.001)` occur about 1.5% of the time rather
  than 0.1%. The upper tail is the reliable end, and is the one most questions are about.
