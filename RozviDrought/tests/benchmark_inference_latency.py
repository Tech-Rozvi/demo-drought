import json
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon

BASE_DIR = Path(__file__).resolve().parents[1]          # .../RozviDrought
WORKSPACE_DIR = Path(__file__).resolve().parents[2]     # .../Infer RozviDrought
sys.path.insert(0, str(BASE_DIR))

from app.services.feature_service import FeatureService
from app.services.fusion_service import FusionService
from app.services.grid_locator import GridLocator
from app.services.polygon_inference_service import PolygonInferenceService
from app.services.subsystem_service import SubsystemService


MASTER_PATH = WORKSPACE_DIR / "data" / "master_inputs" / "master_inputs_long_198001_205012.parquet"
OUTPUT_JSON = BASE_DIR / "runs" / "exports" / "benchmark_latency_report.json"

SCENARIO = "ssp245"
RUN_YYYYMM = 202701
WARMUP_RUNS = 1
REPEAT_RUNS = 5

POINT_LON = 30.344238
POINT_LAT = -18.788573

POLYGON_GEOM = Polygon([
    (30.00, -18.00),
    (30.10, -18.00),
    (30.10, -18.10),
    (30.00, -18.10),
    (30.00, -18.00),
])

MULTIPOLYGON_GEOM = MultiPolygon([
    Polygon([
        (30.00, -18.00),
        (30.10, -18.00),
        (30.10, -18.10),
        (30.00, -18.10),
        (30.00, -18.00),
    ]),
    Polygon([
        (30.20, -18.20),
        (30.30, -18.20),
        (30.30, -18.30),
        (30.20, -18.30),
        (30.20, -18.20),
    ]),
])


def now_perf():
    return time.perf_counter()


def stats_dict(samples):
    xs = sorted(samples)
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0, "p95_s": 0.0, "worst_s": 0.0}
    p95_idx = min(n - 1, max(0, round(0.95 * (n - 1))))
    return {
        "n": n,
        "mean_s": float(statistics.mean(xs)),
        "median_s": float(statistics.median(xs)),
        "min_s": float(xs[0]),
        "max_s": float(xs[-1]),
        "p95_s": float(xs[p95_idx]),
        "worst_s": float(xs[-1]),
    }


def get_windows_cpu_name():
    try:
        result = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, check=False)
        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        return lines[1] if len(lines) >= 2 else None
    except Exception:
        return None


def get_specs():
    specs = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_name": get_windows_cpu_name(),
        "master_path": str(MASTER_PATH),
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        specs.update({
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        })
    except Exception as e:
        specs["psutil_error"] = str(e)

    try:
        import tensorflow as tf
        specs["tensorflow_version"] = tf.__version__
    except Exception as e:
        specs["tensorflow_error"] = str(e)

    try:
        specs["pandas_version"] = pd.__version__
    except Exception:
        pass

    return specs


def load_master_df():
    df = pd.read_parquet(MASTER_PATH)
    df["yyyymm"] = df["yyyymm"].astype(str)
    return df


def prepare_timeseries_for_pixel(master_df, pixel_id, scenario, run_yyyymm):
    run_yyyymm = str(run_yyyymm)
    base = master_df[master_df["pixel_id"] == pixel_id].copy()

    if scenario == "historical":
        ts = base[base["scenario"] == "historical"].copy()
        ts = ts[ts["yyyymm"] <= run_yyyymm]
    else:
        hist = base[(base["scenario"] == "historical") & (base["yyyymm"] <= "202512")].copy()
        fut = base[(base["scenario"] == scenario) & (base["yyyymm"] >= "202601") & (base["yyyymm"] <= run_yyyymm)].copy()
        ts = pd.concat([hist, fut], ignore_index=True)

    ts = (
        ts.sort_values(["yyyymm"])
        .drop_duplicates(subset=["pixel_id", "yyyymm"], keep="last")
        .reset_index(drop=True)
    )
    return ts


def geometry_pixel_ids(master_df, geometry):
    pts = master_df[["pixel_id", "lon", "lat"]].drop_duplicates().copy()
    pts["_inside"] = pts.apply(
        lambda r: geometry.contains(Point(r["lon"], r["lat"])) or geometry.touches(Point(r["lon"], r["lat"])),
        axis=1,
    )
    return sorted(pts.loc[pts["_inside"], "pixel_id"].unique().tolist())


def benchmark_point_models(master_df):
    grid_locator = GridLocator()
    feature_service = FeatureService()
    subsystem_service = SubsystemService()
    fusion_service = FusionService()

    pixel_id = grid_locator.locate_point(POINT_LON, POINT_LAT)
    ts = prepare_timeseries_for_pixel(master_df, pixel_id, SCENARIO, RUN_YYYYMM)

    prep_samples, subs_samples, fusion_samples, total_samples = [], [], [], []

    for _ in range(WARMUP_RUNS):
        prepared = feature_service.prepare_subsystem_inputs(ts, run_yyyymm=RUN_YYYYMM)
        subs_out = subsystem_service.run_subsystems(prepared)
        fusion_service.run(subs_out, model="hybrid")

    for _ in range(REPEAT_RUNS):
        t0 = now_perf()
        prepared = feature_service.prepare_subsystem_inputs(ts, run_yyyymm=RUN_YYYYMM)
        t1 = now_perf()
        subs_out = subsystem_service.run_subsystems(prepared)
        t2 = now_perf()
        fusion_service.run(subs_out, model="hybrid")
        t3 = now_perf()

        prep_samples.append(t1 - t0)
        subs_samples.append(t2 - t1)
        fusion_samples.append(t3 - t2)
        total_samples.append(t3 - t0)

    return {
        "pixel_id": int(pixel_id),
        "timeseries_rows": int(len(ts)),
        "prepare_inputs": stats_dict(prep_samples),
        "subsystems": stats_dict(subs_samples),
        "fusion": stats_dict(fusion_samples),
        "total_models_path": stats_dict(total_samples),
    }


def benchmark_point_full_chain(master_df):
    grid_locator = GridLocator()
    feature_service = FeatureService()
    subsystem_service = SubsystemService()
    fusion_service = FusionService()

    locate_samples, read_samples, prep_samples, subs_samples, fusion_samples, total_samples = [], [], [], [], [], []

    for _ in range(WARMUP_RUNS):
        pixel_id = grid_locator.locate_point(POINT_LON, POINT_LAT)
        ts = prepare_timeseries_for_pixel(master_df, pixel_id, SCENARIO, RUN_YYYYMM)
        prepared = feature_service.prepare_subsystem_inputs(ts, run_yyyymm=RUN_YYYYMM)
        subs_out = subsystem_service.run_subsystems(prepared)
        fusion_service.run(subs_out, model="hybrid")

    for _ in range(REPEAT_RUNS):
        t0 = now_perf()
        pixel_id = grid_locator.locate_point(POINT_LON, POINT_LAT)
        t1 = now_perf()
        ts = prepare_timeseries_for_pixel(master_df, pixel_id, SCENARIO, RUN_YYYYMM)
        t2 = now_perf()
        prepared = feature_service.prepare_subsystem_inputs(ts, run_yyyymm=RUN_YYYYMM)
        t3 = now_perf()
        subs_out = subsystem_service.run_subsystems(prepared)
        t4 = now_perf()
        fusion_service.run(subs_out, model="hybrid")
        t5 = now_perf()

        locate_samples.append(t1 - t0)
        read_samples.append(t2 - t1)
        prep_samples.append(t3 - t2)
        subs_samples.append(t4 - t3)
        fusion_samples.append(t5 - t4)
        total_samples.append(t5 - t0)

    return {
        "locate_point": stats_dict(locate_samples),
        "read_timeseries": stats_dict(read_samples),
        "prepare_inputs": stats_dict(prep_samples),
        "subsystems": stats_dict(subs_samples),
        "fusion": stats_dict(fusion_samples),
        "total_full_chain": stats_dict(total_samples),
    }


def benchmark_polygon_chain(master_df, geometry, label):
    service = PolygonInferenceService(master_df=master_df)
    pixel_ids = geometry_pixel_ids(master_df, geometry)

    for _ in range(WARMUP_RUNS):
        service.infer_polygon(geometry=geometry, scenario=SCENARIO, yyyymm=RUN_YYYYMM, model="hybrid")

    samples = []
    for _ in range(REPEAT_RUNS):
        t0 = now_perf()
        service.infer_polygon(geometry=geometry, scenario=SCENARIO, yyyymm=RUN_YYYYMM, model="hybrid")
        t1 = now_perf()
        samples.append(t1 - t0)

    s = stats_dict(samples)
    return {
        "geometry_type": label,
        "cells_selected": len(pixel_ids),
        "total": s,
        "mean_seconds_per_cell": (s["mean_s"] / len(pixel_ids)) if pixel_ids else None,
        "worst_seconds_per_cell": (s["worst_s"] / len(pixel_ids)) if pixel_ids else None,
    }


def main():
    print("\n" + "=" * 100)
    print("ROZVIDROUGHT PERFORMANCE BENCHMARK")
    print("=" * 100)

    specs = get_specs()
    print("\nPC SPECS")
    for k, v in specs.items():
        print(f"{k}: {v}")

    t0 = now_perf()
    master_df = load_master_df()
    t1 = now_perf()

    print("\nDATASET")
    print("rows:", f"{len(master_df):,}")
    print("load_time_s:", round(t1 - t0, 6))

    print("\nPOINT — MODELS PRIORITY")
    point_models = benchmark_point_models(master_df)
    print(json.dumps(point_models, indent=2))

    print("\nPOINT — FULL CHAIN")
    point_full = benchmark_point_full_chain(master_df)
    print(json.dumps(point_full, indent=2))

    print("\nPOLYGON — FULL CHAIN")
    polygon_perf = benchmark_polygon_chain(master_df, POLYGON_GEOM, "Polygon")
    print(json.dumps(polygon_perf, indent=2))

    print("\nMULTIPOLYGON — FULL CHAIN")
    multipolygon_perf = benchmark_polygon_chain(master_df, MULTIPOLYGON_GEOM, "MultiPolygon")
    print(json.dumps(multipolygon_perf, indent=2))

    report = {
        "selected_access_method": "Models accessed through installed packages and app services as the operational interface.",
        "pc_specs": specs,
        "dataset": {
            "master_path": str(MASTER_PATH),
            "rows": int(len(master_df)),
            "load_time_s": t1 - t0,
        },
        "benchmarks": {
            "point_models_priority": point_models,
            "point_full_chain": point_full,
            "polygon_full_chain": polygon_perf,
            "multipolygon_full_chain": multipolygon_perf,
        },
        "outlook": {
            "worst_case_point_models_s": point_models["total_models_path"]["worst_s"],
            "worst_case_point_full_chain_s": point_full["total_full_chain"]["worst_s"],
            "worst_case_polygon_full_chain_s": polygon_perf["total"]["worst_s"],
            "worst_case_multipolygon_full_chain_s": multipolygon_perf["total"]["worst_s"],
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nWORST-CASE OUTLOOK")
    print("point models only worst_s:", round(report["outlook"]["worst_case_point_models_s"], 6))
    print("point full chain worst_s:", round(report["outlook"]["worst_case_point_full_chain_s"], 6))
    print("polygon full chain worst_s:", round(report["outlook"]["worst_case_polygon_full_chain_s"], 6))
    print("multipolygon full chain worst_s:", round(report["outlook"]["worst_case_multipolygon_full_chain_s"], 6))
    print("\nSaved report to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()