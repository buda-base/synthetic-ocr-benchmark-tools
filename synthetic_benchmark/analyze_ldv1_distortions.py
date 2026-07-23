#!/usr/bin/env python3
"""Build a SQLite statistics database from sampled ldv1 Parquet results."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
import psycopg
import pyarrow.parquet as pq


QUANTILES = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
ROTATION_BINS = (
    -10,
    -5,
    -3,
    -2,
    -1.5,
    -1,
    -0.75,
    -0.5,
    -0.25,
    -0.1,
    0,
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    5,
    10,
)
ROTATION_ABS_BINS = (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)
TPS_PIXELS_BINS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
TPS_HEIGHT_BINS = (
    0,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.25,
)


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "ldv1_distortion_stats.sqlite",
    )
    parser.add_argument("--bucket", default=os.getenv("BEC_DEST_S3_BUCKET", "bec.bdrc.io"))
    parser.add_argument("--prefix", default="ldv1/")
    parser.add_argument("--sample-volumes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def sql_population() -> dict[str, Any]:
    required = ("BEC_SQL_HOST", "BEC_SQL_USER", "BEC_SQL_PASSWORD")
    if not all(os.getenv(name) for name in required):
        return {"available": False}
    with psycopg.connect(
        host=os.environ["BEC_SQL_HOST"],
        port=os.getenv("BEC_SQL_PORT", "5432"),
        dbname=os.getenv("BEC_SQL_DATABASE", "pipeline_v1"),
        user=os.environ["BEC_SQL_USER"],
        password=os.environ["BEC_SQL_PASSWORD"],
        connect_timeout=15,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT config FROM jobs WHERE name = %s", ("ldv1",))
            config_row = cursor.fetchone()
            cursor.execute(
                """
                SELECT count(*), COALESCE(sum(te.total_images), 0),
                       COALESCE(sum(te.nb_errors), 0), min(te.done_at),
                       max(te.done_at), count(DISTINCT te.volume_id)
                FROM task_executions te
                JOIN jobs j ON j.id = te.job_id
                WHERE j.name = %s AND te.status = %s
                """,
                ("ldv1", "done"),
            )
            row = cursor.fetchone()
    return {
        "available": True,
        "job_config": config_row[0] if config_row else None,
        "done_tasks": row[0],
        "total_images": row[1],
        "total_errors": row[2],
        "first_done_at": row[3].isoformat() if row[3] else None,
        "last_done_at": row[4].isoformat() if row[4] else None,
        "distinct_volumes": row[5],
    }


def reservoir_sample_objects(
    s3: Any,
    *,
    bucket: str,
    prefix: str,
    sample_size: int,
    seed: int,
) -> tuple[list[S3Object], dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[S3Object] = []
    object_count = 0
    total_bytes = 0
    first_modified: datetime | None = None
    last_modified: datetime | None = None
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if not item["Key"].endswith(".parquet"):
                continue
            object_count += 1
            total_bytes += int(item["Size"])
            modified = item["LastModified"]
            first_modified = modified if first_modified is None else min(first_modified, modified)
            last_modified = modified if last_modified is None else max(last_modified, modified)
            obj = S3Object(item["Key"], int(item["Size"]), modified.isoformat())
            if len(sample) < sample_size:
                sample.append(obj)
            else:
                replacement = rng.randrange(object_count)
                if replacement < sample_size:
                    sample[replacement] = obj
    sample.sort(key=lambda obj: obj.key)
    return sample, {
        "parquet_objects": object_count,
        "parquet_bytes": total_bytes,
        "first_modified": first_modified.isoformat() if first_modified else None,
        "last_modified": last_modified.isoformat() if last_modified else None,
    }


def strip_corner_anchors(
    points: list[list[float]] | None,
) -> tuple[list[list[float]], bool, float | None, float | None]:
    if not points or len(points) < 4:
        return points or [], False, None, None
    tail = np.asarray(points[-4:], dtype=np.float64)
    if tail.shape != (4, 4):
        return points, False, None, None
    identity = np.allclose(tail[:, :2], tail[:, 2:], atol=1e-3, rtol=0)
    max_y = float(tail[2, 0])
    max_x = float(tail[1, 1])
    expected = np.asarray(
        [
            [0, 0, 0, 0],
            [0, max_x, 0, max_x],
            [max_y, 0, max_y, 0],
            [max_y, max_x, max_y, max_x],
        ],
        dtype=np.float64,
    )
    if not identity or not np.allclose(tail, expected, atol=1e-3, rtol=0):
        return points, False, None, None
    return points[:-4], True, max_x + 1.0, max_y + 1.0


def parse_object(
    s3: Any,
    bucket: str,
    obj: S3Object,
) -> tuple[S3Object, list[dict[str, Any]], list[list[tuple[Any, ...]]]]:
    payload = s3.get_object(Bucket=bucket, Key=obj.key)["Body"].read()
    table = pq.read_table(
        io.BytesIO(payload),
        columns=["img_file_name", "rotation_angle", "tps_points", "tps_alpha", "ok"],
    )
    pages: list[dict[str, Any]] = []
    page_points: list[list[tuple[Any, ...]]] = []
    for row in table.to_pylist():
        rotation = row["rotation_angle"]
        raw_points = row["tps_points"] or []
        points, corners_removed, width, height = strip_corner_anchors(raw_points)
        control_rows: list[tuple[Any, ...]] = []
        observed_dy: list[float] = []
        observed_dx: list[float] = []
        for point_index, point in enumerate(points):
            in_y, in_x, out_y, out_x = (float(value) for value in point)
            dy = in_y - out_y
            dx = in_x - out_x
            observed_dy.append(dy)
            observed_dx.append(dx)
            control_rows.append(
                (
                    point_index,
                    in_x / width if width else None,
                    in_y / height if height else None,
                    out_x / width if width else None,
                    out_y / height if height else None,
                    dx,
                    dy,
                    dx / width if width else None,
                    dy / height if height else None,
                )
            )
        dy_array = np.asarray(observed_dy, dtype=np.float64)
        dx_array = np.asarray(observed_dx, dtype=np.float64)
        pages.append(
            {
                "img_file_name": row["img_file_name"],
                "ok": bool(row["ok"]),
                "rotation_angle_deg": float(rotation) if rotation is not None else None,
                "simulation_rotation_deg": -float(rotation) if rotation is not None else None,
                "tps_alpha": float(row["tps_alpha"]) if row["tps_alpha"] is not None else None,
                "tps_raw_point_count": len(raw_points),
                "tps_point_count": len(points),
                "corners_removed": corners_removed,
                "image_width_px": width,
                "image_height_px": height,
                "tps_mean_dy_px": float(dy_array.mean()) if dy_array.size else None,
                "tps_rms_dy_px": float(np.sqrt(np.mean(dy_array**2))) if dy_array.size else None,
                "tps_max_abs_dy_px": float(np.max(np.abs(dy_array))) if dy_array.size else None,
                "tps_peak_to_peak_dy_px": float(np.ptp(dy_array)) if dy_array.size else None,
                "tps_max_abs_dx_px": float(np.max(np.abs(dx_array))) if dx_array.size else None,
                "tps_rms_dy_height": (
                    float(np.sqrt(np.mean(dy_array**2)) / height)
                    if dy_array.size and height
                    else None
                ),
                "tps_max_abs_dy_height": (
                    float(np.max(np.abs(dy_array)) / height)
                    if dy_array.size and height
                    else None
                ),
                "tps_peak_to_peak_dy_height": (
                    float(np.ptp(dy_array) / height) if dy_array.size and height else None
                ),
            }
        )
        page_points.append(control_rows)
    return obj, pages, page_points


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS sampled_objects;
        DROP TABLE IF EXISTS pages;
        DROP TABLE IF EXISTS control_points;
        DROP TABLE IF EXISTS quantiles;
        DROP TABLE IF EXISTS histograms;
        DROP TABLE IF EXISTS recommendations;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE sampled_objects (
            object_key TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            last_modified TEXT NOT NULL,
            page_count INTEGER,
            download_ok INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE pages (
            page_id INTEGER PRIMARY KEY,
            object_key TEXT NOT NULL REFERENCES sampled_objects(object_key),
            img_file_name TEXT NOT NULL,
            ok INTEGER NOT NULL,
            rotation_angle_deg REAL,
            simulation_rotation_deg REAL,
            tps_alpha REAL,
            tps_raw_point_count INTEGER NOT NULL,
            tps_point_count INTEGER NOT NULL,
            corners_removed INTEGER NOT NULL,
            image_width_px REAL,
            image_height_px REAL,
            tps_mean_dy_px REAL,
            tps_rms_dy_px REAL,
            tps_max_abs_dy_px REAL,
            tps_peak_to_peak_dy_px REAL,
            tps_max_abs_dx_px REAL,
            tps_rms_dy_height REAL,
            tps_max_abs_dy_height REAL,
            tps_peak_to_peak_dy_height REAL
        );
        CREATE TABLE control_points (
            page_id INTEGER NOT NULL REFERENCES pages(page_id),
            point_index INTEGER NOT NULL,
            observed_x_norm REAL,
            observed_y_norm REAL,
            ideal_x_norm REAL,
            ideal_y_norm REAL,
            observed_dx_px REAL NOT NULL,
            observed_dy_px REAL NOT NULL,
            observed_dx_width REAL,
            observed_dy_height REAL,
            PRIMARY KEY (page_id, point_index)
        );
        CREATE TABLE quantiles (
            metric TEXT NOT NULL,
            subset TEXT NOT NULL,
            quantile REAL NOT NULL,
            value REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            PRIMARY KEY (metric, subset, quantile)
        );
        CREATE TABLE histograms (
            metric TEXT NOT NULL,
            subset TEXT NOT NULL,
            bin_lower REAL NOT NULL,
            bin_upper REAL NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (metric, subset, bin_lower, bin_upper)
        );
        CREATE TABLE recommendations (
            parameter TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            basis TEXT NOT NULL
        );
        """
    )


def set_metadata(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, sort_keys=True, default=str)) for key, value in values.items()],
    )


def insert_object_result(
    connection: sqlite3.Connection,
    obj: S3Object,
    pages: list[dict[str, Any]],
    page_points: list[list[tuple[Any, ...]]],
) -> None:
    connection.execute(
        """
        UPDATE sampled_objects
        SET page_count = ?, download_ok = 1
        WHERE object_key = ?
        """,
        (len(pages), obj.key),
    )
    page_columns = tuple(pages[0].keys()) if pages else ()
    for page, points in zip(pages, page_points):
        cursor = connection.execute(
            f"""
            INSERT INTO pages(object_key, {", ".join(page_columns)})
            VALUES (?, {", ".join("?" for _ in page_columns)})
            """,
            (obj.key, *(page[column] for column in page_columns)),
        )
        page_id = cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO control_points(
                page_id, point_index, observed_x_norm, observed_y_norm,
                ideal_x_norm, ideal_y_norm, observed_dx_px, observed_dy_px,
                observed_dx_width, observed_dy_height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(page_id, *point) for point in points],
        )


def finite_values(connection: sqlite3.Connection, column: str, where: str) -> np.ndarray:
    rows = connection.execute(
        f"SELECT {column} FROM pages WHERE {column} IS NOT NULL AND ({where})"
    ).fetchall()
    values = np.asarray([row[0] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def add_distribution(
    connection: sqlite3.Connection,
    *,
    metric: str,
    subset: str,
    values: np.ndarray,
    bins: Iterable[float],
) -> None:
    if not values.size:
        return
    quantile_values = np.quantile(values, QUANTILES)
    connection.executemany(
        "INSERT INTO quantiles(metric, subset, quantile, value, sample_count) VALUES (?, ?, ?, ?, ?)",
        [
            (metric, subset, quantile, float(value), int(values.size))
            for quantile, value in zip(QUANTILES, quantile_values)
        ],
    )
    bin_array = np.asarray(tuple(bins), dtype=np.float64)
    counts, edges = np.histogram(values, bins=bin_array)
    connection.executemany(
        "INSERT INTO histograms(metric, subset, bin_lower, bin_upper, count) VALUES (?, ?, ?, ?, ?)",
        [
            (metric, subset, float(lower), float(upper), int(count))
            for lower, upper, count in zip(edges[:-1], edges[1:], counts)
        ],
    )


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def finalize_statistics(connection: sqlite3.Connection, *, seed: int) -> dict[str, Any]:
    ok_pages = connection.execute("SELECT count(*) FROM pages WHERE ok = 1").fetchone()[0]
    rotation_pages = connection.execute(
        "SELECT count(*) FROM pages WHERE ok = 1 AND rotation_angle_deg IS NOT NULL"
    ).fetchone()[0]
    rotation_nonzero_pages = connection.execute(
        """
        SELECT count(*) FROM pages
        WHERE ok = 1 AND rotation_angle_deg IS NOT NULL AND rotation_angle_deg != 0
        """
    ).fetchone()[0]
    tps_pages = connection.execute(
        "SELECT count(*) FROM pages WHERE ok = 1 AND tps_point_count > 0"
    ).fetchone()[0]
    corner_pages = connection.execute(
        "SELECT count(*) FROM pages WHERE tps_raw_point_count > 0 AND corners_removed = 1"
    ).fetchone()[0]
    raw_tps_pages = connection.execute(
        "SELECT count(*) FROM pages WHERE tps_raw_point_count > 0"
    ).fetchone()[0]

    rotation = finite_values(connection, "simulation_rotation_deg", "ok = 1")
    rotation_abs = np.abs(rotation)
    tps_max_px = finite_values(connection, "tps_max_abs_dy_px", "ok = 1")
    tps_rms_px = finite_values(connection, "tps_rms_dy_px", "ok = 1")
    tps_range_px = finite_values(connection, "tps_peak_to_peak_dy_px", "ok = 1")
    tps_max_height = finite_values(connection, "tps_max_abs_dy_height", "ok = 1")
    tps_rms_height = finite_values(connection, "tps_rms_dy_height", "ok = 1")
    tps_range_height = finite_values(connection, "tps_peak_to_peak_dy_height", "ok = 1")
    object_rates = np.asarray(
        connection.execute(
            """
            SELECT count(*),
                   sum(rotation_angle_deg IS NOT NULL AND rotation_angle_deg != 0),
                   sum(tps_point_count > 0)
            FROM pages WHERE ok = 1 GROUP BY object_key
            """
        ).fetchall(),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_rates = []
    for _ in range(2000):
        sample = object_rates[rng.integers(0, len(object_rates), len(object_rates))].sum(axis=0)
        bootstrap_rates.append((sample[1] / sample[0], sample[2] / sample[0]))
    bootstrap_rates_array = np.asarray(bootstrap_rates)
    rotation_rate_ci = [
        float(value) for value in np.quantile(bootstrap_rates_array[:, 0], [0.025, 0.975])
    ]
    tps_rate_ci = [
        float(value) for value in np.quantile(bootstrap_rates_array[:, 1], [0.025, 0.975])
    ]

    control_x_medians = []
    for point_index in range(5):
        values = np.asarray(
            [
                row[0]
                for row in connection.execute(
                    "SELECT ideal_x_norm FROM control_points WHERE point_index = ?",
                    (point_index,),
                )
            ],
            dtype=np.float64,
        )
        control_x_medians.append(float(np.median(values)))
    point_counts = connection.execute(
        "SELECT min(tps_point_count), max(tps_point_count) FROM pages WHERE tps_point_count > 0"
    ).fetchone()
    alpha_values = connection.execute(
        "SELECT DISTINCT tps_alpha FROM pages WHERE tps_point_count > 0 ORDER BY tps_alpha"
    ).fetchall()
    max_horizontal_displacement = connection.execute(
        "SELECT max(tps_max_abs_dx_px) FROM pages WHERE tps_point_count > 0"
    ).fetchone()[0]

    distributions = (
        ("simulation_rotation_deg", "rotation_present", rotation, ROTATION_BINS),
        ("simulation_rotation_abs_deg", "rotation_present", rotation_abs, ROTATION_ABS_BINS),
        ("tps_max_abs_dy_px", "tps_present", tps_max_px, TPS_PIXELS_BINS),
        ("tps_rms_dy_px", "tps_present", tps_rms_px, TPS_PIXELS_BINS),
        ("tps_peak_to_peak_dy_px", "tps_present", tps_range_px, TPS_PIXELS_BINS),
        ("tps_max_abs_dy_height", "tps_present", tps_max_height, TPS_HEIGHT_BINS),
        ("tps_rms_dy_height", "tps_present", tps_rms_height, TPS_HEIGHT_BINS),
        ("tps_peak_to_peak_dy_height", "tps_present", tps_range_height, TPS_HEIGHT_BINS),
    )
    for metric, subset, values, bins in distributions:
        add_distribution(
            connection,
            metric=metric,
            subset=subset,
            values=values,
            bins=bins,
        )

    recommendations: dict[str, tuple[Any, str]] = {
        "rotation_probability": (
            rotation_nonzero_pages / ok_pages if ok_pages else 0,
            "Fraction of successful sampled pages with a nonzero stored correction angle.",
        ),
        "rotation_probability_cluster_bootstrap_95_ci": (
            rotation_rate_ci,
            "95% volume-cluster bootstrap interval for nonzero rotation prevalence.",
        ),
        "rotation_recorded_probability": (
            rotation_pages / ok_pages if ok_pages else 0,
            "Fraction of successful sampled pages with a stored angle, including exact zero.",
        ),
        "tps_probability": (
            tps_pages / ok_pages if ok_pages else 0,
            "Fraction of successful sampled pages with non-corner TPS control points.",
        ),
        "tps_probability_cluster_bootstrap_95_ci": (
            tps_rate_ci,
            "95% volume-cluster bootstrap interval for TPS prevalence.",
        ),
        "tps_given_rotation_probability": (
            tps_pages / rotation_pages if rotation_pages else 0,
            "All sampled TPS pages had a stored rotation; this is P(TPS | rotation recorded).",
        ),
        "tps_control_point_count": (
            point_counts[0] if point_counts[0] == point_counts[1] else list(point_counts),
            "Non-corner control points per sampled TPS page.",
        ),
        "tps_control_point_x_norm_medians": (
            control_x_medians,
            "Median ideal x positions of the five non-corner points, normalized by image width.",
        ),
        "tps_alpha_values": (
            [row[0] for row in alpha_values],
            "Distinct TPS regularization values in the sample.",
        ),
        "tps_max_horizontal_displacement_px": (
            max_horizontal_displacement,
            "Maximum |observed_x - ideal_x|; ldv1 currently records vertical-only TPS warps.",
        ),
        "corner_anchor_recognition_rate": (
            corner_pages / raw_tps_pages if raw_tps_pages else 0,
            "Fraction of raw TPS pages whose final four points matched identity image corners.",
        ),
    }
    if rotation.size:
        recommendations.update(
            {
                "rotation_typical_range_deg": (
                    [quantile(rotation, 0.10), quantile(rotation, 0.90)],
                    "Central 80% of inverse stored correction angles; apply to clean pages.",
                ),
                "rotation_broad_range_deg": (
                    [quantile(rotation, 0.025), quantile(rotation, 0.975)],
                    "Central 95% of inverse stored correction angles.",
                ),
                "rotation_abs_p95_deg": (
                    quantile(rotation_abs, 0.95),
                    "95th percentile absolute inverse correction angle.",
                ),
                "rotation_abs_p99_deg": (
                    quantile(rotation_abs, 0.99),
                    "99th percentile absolute inverse correction angle; useful as a cap.",
                ),
            }
        )
    if tps_max_height.size:
        recommendations.update(
            {
                "tps_max_displacement_typical_height_fraction": (
                    quantile(tps_max_height, 0.90),
                    "90th percentile page maximum |observed_y - ideal_y| / image height.",
                ),
                "tps_max_displacement_broad_height_fraction": (
                    quantile(tps_max_height, 0.975),
                    "97.5th percentile page maximum vertical displacement / image height.",
                ),
                "tps_max_displacement_cap_height_fraction": (
                    quantile(tps_max_height, 0.99),
                    "99th percentile page maximum vertical displacement / image height.",
                ),
                "tps_peak_to_peak_typical_height_fraction": (
                    quantile(tps_range_height, 0.90),
                    "90th percentile within-page peak-to-peak vertical displacement / image height.",
                ),
            }
        )
    connection.executemany(
        "INSERT INTO recommendations(parameter, value, basis) VALUES (?, ?, ?)",
        [
            (parameter, json.dumps(value, sort_keys=True), basis)
            for parameter, (value, basis) in recommendations.items()
        ],
    )
    summary = {
        "sampled_pages": connection.execute("SELECT count(*) FROM pages").fetchone()[0],
        "successful_pages": ok_pages,
        "rotation_pages": rotation_pages,
        "rotation_nonzero_pages": rotation_nonzero_pages,
        "tps_pages": tps_pages,
        "raw_tps_pages": raw_tps_pages,
        "corner_pages": corner_pages,
        "recommendations": {key: value for key, (value, _) in recommendations.items()},
    }
    set_metadata(connection, {"analysis_summary": summary})
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS pages_rotation_idx ON pages(rotation_angle_deg);
        CREATE INDEX IF NOT EXISTS pages_tps_idx ON pages(tps_point_count);
        CREATE INDEX IF NOT EXISTS pages_object_idx ON pages(object_key);
        CREATE INDEX IF NOT EXISTS points_dy_idx ON control_points(observed_dy_height);
        ANALYZE;
        """
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.sample_volumes <= 0:
        raise ValueError("--sample-volumes must be positive")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=os.getenv("BEC_REGION"))
    sql_stats = sql_population()
    print("Listing ldv1 Parquet objects and selecting a deterministic reservoir sample...")
    objects, s3_stats = reservoir_sample_objects(
        s3,
        bucket=args.bucket,
        prefix=args.prefix,
        sample_size=args.sample_volumes,
        seed=args.seed,
    )
    if not objects:
        raise RuntimeError(f"No Parquet objects found under s3://{args.bucket}/{args.prefix}")

    with sqlite3.connect(args.database) as connection:
        create_schema(connection)
        set_metadata(
            connection,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": f"s3://{args.bucket}/{args.prefix}",
                "sample_method": "deterministic reservoir sample over Parquet objects",
                "sample_seed": args.seed,
                "requested_sample_volumes": args.sample_volumes,
                "sql_population": sql_stats,
                "s3_population": s3_stats,
                "corner_policy": (
                    "Remove only a trailing TL,TR,BL,BR identity quartet in [in_y,in_x,out_y,out_x]."
                ),
                "simulation_policy": (
                    "Stored transforms correct observed pages; simulation_rotation_deg and "
                    "control_points.observed_dy_* invert them to recreate observed distortion."
                ),
            },
        )
        connection.executemany(
            "INSERT INTO sampled_objects(object_key, size_bytes, last_modified) VALUES (?, ?, ?)",
            [(obj.key, obj.size, obj.last_modified) for obj in objects],
        )
        connection.commit()

        processed = 0
        print(f"Reading {len(objects)} sampled Parquet objects with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda obj: parse_object(s3, args.bucket, obj),
                objects,
            )
            for obj, pages, page_points in results:
                insert_object_result(connection, obj, pages, page_points)
                processed += 1
                if processed % 32 == 0 or processed == len(objects):
                    connection.commit()
                    print(f"Processed {processed}/{len(objects)} objects")

        summary = finalize_statistics(connection, seed=args.seed)
        connection.commit()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.database}")


if __name__ == "__main__":
    main()
