# Data sources — demo profile

Everything in `data/demo/` is **synthetic**. This file is the manifest for that
claim, and `scripts/audit_public.py` is the machine that enforces it.

## The rule

The demo profile carries no real property data of any kind: no real street
address, no real homeowners association, no real builder, no real contractor, no
real utility company, and no real statute or municipal ordinance.

The one deliberate exception is **city-level public geography**. "Dallas, TX" and
"Minneapolis, MN" are real cities, and the coordinates behind them are real, so
that the free weather APIs return genuine live forecasts for a plausible point.
Everything attached to those coordinates is invented.

## Saved homes

| home_id | Label | Address | Status |
|---|---|---|---|
| `demo-002` | Minneapolis, MN | 1412 Larkspur Lane, Minneapolis, MN 55409 | **primary** — invented street, HOA, builder, utilities |
| `demo-001` | Dallas, TX | 5000 Maple Street, Dallas, TX 75201 | secondary — invented street, HOA, builder, utilities |

Two homes, in deliberately distant climate zones (IECC 6A cold and IECC 3A
mixed-humid). That is not decoration: the jurisdiction-isolation control is only
testable with two homes whose documents disagree, and the freeze-risk pipeline is
only demonstrable year-round with a cold-climate home.

## Corpus

| Scope | Files | Kind |
|---|---|---|
| `common/` | appliance care, freeze prevention | synthetic, applies to every home |
| `demo-001/` | Maple Grove CC&Rs, city permit checklist, renter summary, short-term rental policy | synthetic |
| `demo-002/` | Lakeshore Commons CC&Rs, city permit checklist, renter summary, short-term rental policy | synthetic |

Every file carries a `> SYNTHETIC DOCUMENT` header stating that it is fictional
and cites no real statute. Section numbers, resolution numbers
(e.g. `Resolution LC-2019-14`) and dollar figures are all invented.

## Structured data

- `contractors.csv` — invented businesses, scoped per home. No real company.
- `utilities.csv` — invented providers, scoped per home. Municipal water and
  sanitation are labelled "synthetic stand-in" because a municipal utility is
  inherently named for its city; the rates are invented.
- `floorplans/<home_id>/` — hand-authored SVG, drawn for this project. Not a real
  builder plan, and each drawing says so on its face.

## Live services the demo does use

These are free, keyless, public-agency APIs. They return real *weather*, which is
not property data:

- US National Weather Service (`api.weather.gov`) — forecasts and active advisories
- Open-Meteo — forecast, elevation and geocoding fallback
- US Census Geocoder — address to coordinates
- USGS EPQS — elevation fallback
- US Energy Information Administration — state residential energy prices

## How this is enforced

```powershell
python scripts/audit_public.py .
```

Exits non-zero on any real-world identifier. The eval suite runs the same
tripwire table against the built vector index (`T18`) and asserts that the data
root, state root, home registry and every real-world provider stay inside the
demo boundary (`T17`). A test that cannot fail is worthless, so `T18` was
verified by planting a real address in the corpus and confirming it fails.

under the scanner's own rules.
