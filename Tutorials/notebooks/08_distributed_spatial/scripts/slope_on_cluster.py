"""Slope of the Great Plains DEM on a local Dask cluster, written to Zarr."""

import time
from pathlib import Path

import geolattice.distributed as pdist

ROOT = Path(__file__).resolve().parents[4]
DEM = ROOT / "Data" / "03_spatial_analysis" / "Great_plain" / "Raster" / "elev.tif"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "gp_slope.zarr"


def main():
    with pdist.start("local", n_workers=4, threads_per_worker=6) as session:
        print("workers:", session.status()["workers"])
        t = time.perf_counter()
        dem = pdist.read_raster(DEM, chunks={"y": 2048, "x": 2048})
        slope = pdist.raster.slope(dem, compute="cpu")
        pdist.write_raster(slope, OUT)  # each worker writes its own chunks
        print(f"wrote {OUT.name} in {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":  # worker processes import this file; only the parent runs main()
    main()
