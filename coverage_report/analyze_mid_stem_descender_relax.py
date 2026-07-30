"""Post-hoc analysis: relax the mid_stem geometry rule for descender-less fonts.

The mid_stem check fires when the lowest body layer hangs below the base by
>12% of base height. Cursive/khyug styles legitimately draw such descending
tails, so mid_stem systematically misfires on them. This script estimates the
coverage impact of only applying mid_stem to fonts whose catalog descender
class is Medium/Tall/Extra (i.e. relaxing it for Short/Legless fonts) WITHOUT
re-shaping: dropping mid_stem from a fail's reason set can only turn fails into
passes, never the reverse.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "synthetic_benchmark"))

from synthetic_common import (  # noqa: E402
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_FONTS_CSV,
    DEFAULT_SCRIPTS_CSV,
    load_font_catalog,
    load_scripts,
)

# Descender classes for which the mid_stem rule is KEPT (real vertical stems).
APPLY_PREFIXES = ("Medium", "Tall", "Extra")
# Descender classes for which the mid_stem rule is RELAXED (little/no descender).
RELAX_PREFIXES = ("Short", "Legless")


def descender_class(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "unknown"
    for p in APPLY_PREFIXES:
        if raw.startswith(p):
            return "apply"
    for p in RELAX_PREFIXES:
        if raw.startswith(p):
            return "relax"
    return "unknown"


def build_descender_map() -> dict[str, tuple[str, str]]:
    """basename -> (raw descender ratio, apply/relax/unknown)."""
    scripts = load_scripts(DEFAULT_SCRIPTS_CSV)
    fonts = load_font_catalog(DEFAULT_BENCHMARK_CSV, DEFAULT_SCRIPTS_CSV, DEFAULT_FONTS_CSV)
    out: dict[str, tuple[str, str]] = {}
    for f in fonts:
        raw = (scripts.get(f.script_id, {}).get("descenders length ratio") or "").strip()
        out[f.basename] = (raw, descender_class(raw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("support_parquet", type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "v3_density018" / "mid_stem_relax")
    ap.add_argument("--kyuma-basenames", type=Path, default=None,
                    help="Optional newline file of Kyuma basenames for focused reporting.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    desc = build_descender_map()
    kyuma = set()
    if args.kyuma_basenames and args.kyuma_basenames.exists():
        kyuma = {b.strip() for b in args.kyuma_basenames.read_text().splitlines() if b.strip()}

    df = pd.read_parquet(
        args.support_parquet,
        columns=["test_kind", "basename", "stack", "is_standard_tibetan",
                 "stack_nb_occurrences", "ok", "reason"],
    )
    df = df[df["test_kind"] == "stack"].copy()
    df["is_standard_tibetan"] = df["is_standard_tibetan"].astype(int)

    df["desc_class"] = df["basename"].map(lambda b: desc.get(b, ("", "unknown"))[1])

    # A failing row flips to ok under the relaxed rule iff the font is in the
    # relax class AND mid_stem is the ONLY reason for the fail.
    reasons = df["reason"].fillna("")
    only_mid_stem = reasons == "mid_stem"
    df["flip"] = (~df["ok"]) & (df["desc_class"] == "relax") & only_mid_stem
    df["ok_relaxed"] = df["ok"] | df["flip"]

    print("== descender class distribution (fonts) ==")
    dist: dict[str, int] = {}
    for _b, (_raw, c) in desc.items():
        dist[c] = dist.get(c, 0) + 1
    for c, n in sorted(dist.items()):
        print(f"  {c:8s} {n}")

    # --- Global coverage over ALL corpus stacks ---
    covered_before = set(df.loc[df["ok"], "stack"].unique())
    covered_after = set(df.loc[df["ok_relaxed"], "stack"].unique())
    newly_covered = covered_after - covered_before
    print("\n== global coverage over ALL corpus stacks (union over ALL fonts) ==")
    print(f"  unique corpus stacks total : {df['stack'].nunique()}")
    print(f"  covered before             : {len(covered_before)}")
    print(f"  covered after relax        : {len(covered_after)}")
    print(f"  newly covered stacks       : {len(newly_covered)}")
    print(f"  (font,stack) fails flipped : {int(df['flip'].sum())}")

    # --- Per-Kyuma-font supported stacks before/after (all + occ-weighted) ---
    # Occurrence weight is per distinct stack (max occ across rows).
    occ = df.groupby("stack")["stack_nb_occurrences"].max()
    total_occ = float(occ.sum()) or 1.0

    rows_out = []
    focus = sorted(kyuma) if kyuma else sorted(df["basename"].unique())
    print("\n== per-font supported stacks (before -> after) and occ-weighted coverage ==")
    for b in focus:
        sub = df[df["basename"] == b]
        if sub.empty:
            continue
        before = int(sub["ok"].sum())
        after = int(sub["ok_relaxed"].sum())
        ok_before_stacks = set(sub.loc[sub["ok"], "stack"])
        ok_after_stacks = set(sub.loc[sub["ok_relaxed"], "stack"])
        w_before = float(occ.reindex(list(ok_before_stacks)).fillna(0).sum()) / total_occ
        w_after = float(occ.reindex(list(ok_after_stacks)).fillna(0).sum()) / total_occ
        cls = desc.get(b, ("", "unknown"))
        rows_out.append({
            "basename": b,
            "descender_raw": cls[0],
            "desc_class": cls[1],
            "ok_before": before,
            "ok_after": after,
            "flips": after - before,
            "occ_cov_before": round(w_before, 4),
            "occ_cov_after": round(w_after, 4),
        })
        if kyuma or after != before:
            print(f"  {b:52s} {cls[1]:7s} {before:6d} -> {after:6d}  "
                  f"occ {w_before*100:5.1f}% -> {w_after*100:5.1f}%")

    out_csv = args.out_dir / "per_font_mid_stem_relax.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_csv}")

    # --- Residual most-common corpus stacks still failing per Kyuma font ---
    if kyuma:
        resid = df[(df["basename"].isin(kyuma)) & (~df["ok_relaxed"])].copy()
        top_resid = (
            resid.groupby(["stack"]).agg(
                nb_occurrences=("stack_nb_occurrences", "max"),
                n_kyuma_fonts_failing=("basename", "nunique"),
            ).sort_values("nb_occurrences", ascending=False).reset_index()
        )
        resid_csv = args.out_dir / "kyuma_residual_failing_stacks.csv"
        top_resid.to_csv(resid_csv, index=False)
        print(f"Wrote {resid_csv} ({len(top_resid)} stacks)")
        print("\n== top-15 most-common corpus stacks STILL failing across Kyuma fonts ==")
        # Map residual fail reasons for context.
        reason_lookup = (
            resid.groupby("stack")["reason"].agg(lambda s: s.value_counts().index[0])
        )
        for _i, r in top_resid.head(15).iterrows():
            rs = reason_lookup.get(r["stack"], "")
            print(f"  {r['stack']:8s} occ={int(r['nb_occurrences']):>9d} "
                  f"fails {int(r['n_kyuma_fonts_failing'])}/{len(kyuma)} kyuma  top_reason={rs}")


if __name__ == "__main__":
    main()
