"""Analyze the top_mark_horizontal_misalignment rule by script style.

Goals:
1. For Khyuma (cursive) fonts, list the stacks that would flip to OK if we relax
   the top-mark horizontal-alignment rule (together with mid_stem, already
   relaxed for short/legless). These are the render candidates for a
   false-positive visual review.
2. Report which OTHER script categories have very common stacks excluded purely
   because of top_mark_horizontal_misalignment, so we can decide whether the
   relaxation should extend beyond Khyuma.
"""
from __future__ import annotations

import argparse
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

TMHM = "top_mark_horizontal_misalignment"
# Relaxed-together for Khyuma fonts (mid_stem already relaxed for short/legless).
KHYUMA_RELAXED = {"mid_stem", TMHM}


def build_font_meta():
    scripts = load_scripts(DEFAULT_SCRIPTS_CSV)
    fonts = load_font_catalog(DEFAULT_BENCHMARK_CSV, DEFAULT_SCRIPTS_CSV, DEFAULT_FONTS_CSV)
    cat = {}
    desc = {}
    for f in fonts:
        cat[f.basename] = f.script_category
        desc[f.basename] = (scripts.get(f.script_id, {}).get("descenders length ratio") or "").strip()
    return cat, desc


def reason_set(reason: str) -> set[str]:
    return {r for r in (reason or "").split(";") if r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("support_parquet", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "reports" / "v3_density018" / "topmark_relax")
    ap.add_argument("--khyuma-category-prefix", default="Kyuma")
    ap.add_argument("--common-occ", type=int, default=100000,
                    help="Occurrence threshold for 'very common' stack reporting.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cat, desc = build_font_meta()
    df = pd.read_parquet(
        args.support_parquet,
        columns=["test_kind", "basename", "stack", "ok", "reason", "stack_nb_occurrences"],
    )
    df = df[df["test_kind"] == "stack"].copy()
    df["ok"] = df["ok"].astype(bool)
    df["category"] = df["basename"].map(lambda b: cat.get(b, "?"))
    df["reasons"] = df["reason"].fillna("").map(reason_set)
    df["has_tmhm"] = df["reasons"].map(lambda s: TMHM in s)
    # Per-row occurrence is often 0; use the global max per stack as the weight.
    occ_map = df.groupby("stack")["stack_nb_occurrences"].max()
    df["stack_nb_occurrences"] = df["stack"].map(occ_map)

    # ---------- Report 1: other styles excluded by TMHM ----------
    fails = df[(~df["ok"]) & df["has_tmhm"]].copy()
    fails["pure_tmhm"] = fails["reasons"].map(lambda s: s == {TMHM})
    print("== top_mark_horizontal_misalignment failures by script category ==")
    print(f"{'category':40s} {'fonts':>5s} {'fail_rows':>9s} {'pure_rows':>9s} "
          f"{'common_pure_stacks':>18s}")
    cat_rows = []
    for c, sub in fails.groupby("category"):
        n_fonts = sub["basename"].nunique()
        n_rows = len(sub)
        pure = sub[sub["pure_tmhm"]]
        n_pure = len(pure)
        common_pure = pure[pure["stack_nb_occurrences"] >= args.common_occ]["stack"].nunique()
        cat_rows.append((c, n_fonts, n_rows, n_pure, common_pure))
    for c, nf, nr, npu, ncp in sorted(cat_rows, key=lambda r: -r[4]):
        print(f"{c:40s} {nf:5d} {nr:9d} {npu:9d} {ncp:18d}")

    # Per-category: most common stacks failing PURELY on tmhm.
    lines = []
    for c, sub in fails[fails["pure_tmhm"]].groupby("category"):
        top = (sub.groupby("stack")
               .agg(occ=("stack_nb_occurrences", "max"),
                    nfonts=("basename", "nunique"))
               .sort_values("occ", ascending=False).head(10))
        lines.append(f"\n[{c}] most common stacks failing purely on {TMHM}:")
        for stack, r in top.iterrows():
            lines.append(f"   {stack:10s} occ={int(r['occ']):>9d}  fails_on {int(r['nfonts'])} {c} fonts")
    (args.out_dir / "tmhm_by_category.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.out_dir/'tmhm_by_category.txt'}")

    # ---------- Report 2: Khyuma flip candidates ----------
    khy = df[df["category"].str.startswith(args.khyuma_category_prefix)].copy()
    khy_basenames = sorted(khy["basename"].unique())
    # Flip if currently failing and every reason is in the relaxed set (for
    # Khyuma we relax both mid_stem and tmhm), AND tmhm is actually one of them
    # (so this specifically exercises the tmhm relaxation).
    def flips(row):
        return (not row["ok"]) and row["has_tmhm"] and row["reasons"].issubset(KHYUMA_RELAXED)
    khy["flip_tmhm"] = khy.apply(flips, axis=1)
    cand = khy[khy["flip_tmhm"]].copy()
    cand_out = (cand[["basename", "stack", "stack_nb_occurrences", "reason"]]
                .sort_values(["basename", "stack_nb_occurrences"], ascending=[True, False]))
    cand_csv = args.out_dir / "khyuma_tmhm_flip_candidates.csv"
    cand_out.to_csv(cand_csv, index=False)
    print(f"\n== Khyuma stacks that flip to OK when relaxing {TMHM} (render candidates) ==")
    print(f"  Khyuma fonts: {len(khy_basenames)}")
    print(f"  total flip candidate rows: {len(cand)}")
    per_font = cand.groupby("basename").size().sort_values(ascending=False)
    for b, n in per_font.items():
        print(f"    {b:52s} {n}")
    print(f"Wrote {cand_csv}")


if __name__ == "__main__":
    main()
