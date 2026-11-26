"""Fill the data cache before building the notebooks: ``python _builders/prefetch.py smap_l3e cci ...``.

Every task calls the same ``sm_data`` function a notebook calls, so the notebooks then read the cache.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sm_data as sd  # noqa: E402

TASKS = {
    "smap_l3e": lambda: sd.smap_l3e(),
    "smap_l3": lambda: sd.smap_l3(),
    "smap_l4": lambda: sd.smap_l4(),
    "smos": lambda: sd.smos(),
    "cci": lambda: [sd.cci(p) for p in ("COMBINED", "ACTIVE", "PASSIVE")],
    "cci_bangladesh": lambda: sd.cci("COMBINED", sd.BANGLADESH, ("1979-01-01", "2024-12-31"), "bangladesh"),
    "era5": lambda: (sd.era5_land(), sd.era5_land_points()),
    "era5_bangladesh": lambda: sd.era5_land(sd.BANGLADESH, ("2010-01-01", "2024-12-31"), "bangladesh", 1.0),
    "soilgrids_region": lambda: sd.soilgrids_grid(sd.IOWA_REGION, "iowa_region", overview_level=1),
    "soilgrids_sar": lambda: sd.soilgrids_grid(sd.IOWA_SAR, "iowa_sar", crs="EPSG:32615"),
    "spl2smap_s": lambda: sd.spl2smap_s(),
    "downscaling": lambda: sd.downscaling_stack(),
    "nisar": lambda: sd.nisar(),
    "s1_2026": lambda: sd.s1_cube(period=("2025-10-01", "2026-11-25"), name="iowa", relative_orbit=165),
    "smap_2026": lambda: sd.smap_l3e(region=sd.IOWA_REGION, period=("2026-01-01", "2026-11-24"), name="iowa"),
    "in_situ_2026": lambda: (sd.uscrn(years=range(2025, 2027)), sd.scan(start="2025-01-01", end="2026-11-24")),
    "s1": lambda: sd.s1_cube(),
    "soilgrids": lambda: ([sd.soilgrids_profile(st) for st in sd.STATIONS.index], sd.soilgrids_grid()),
    "in_situ": lambda: (sd.uscrn(), sd.scan(), sd.usgs_daily_flow(), sd.gridmet_point(sd.IOWA["lon"], sd.IOWA["lat"])),
}

for name in sys.argv[1:]:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
    try:
        TASKS[name]()
        print(f"[{time.strftime('%H:%M:%S')}] done {name} in {time.time() - t0:.0f} s", flush=True)
    except Exception as err:  # keep going with the other tasks
        import traceback

        print(f"[{time.strftime('%H:%M:%S')}] FAILED {name}: {err!r}", flush=True)
        traceback.print_exc()
