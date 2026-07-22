# Tibetan shorthand lexicon

Unified expansion → abbreviation pairs used for linguistic augmentation in the synthetic OCR benchmark.

## Sources

| File | Source | Notes |
|------|--------|-------|
| `rkts_abb.xml` | [brunogml/rKTs](https://github.com/brunogml/rKTs) `zzz in progress/abb.xml` | Dzongkha handbook, Babelstone contractions, Bacot 1912 |
| `tibschol_abbr.csv` | [ERC-TibSchol/abbreviations](https://github.com/ERC-TibSchol/abbreviations) | Corpus-attested abbreviations |
| `shorthands.csv` | Built locally | Columns: `long_form`, `shorthand`, `source` |
| `denylist.csv` | Manual review | Optional rejects: `basename`, `shorthand`, `stack`, `reason` |

Pagan Tibet / MonlamAI dictionaries are intentionally not included in this pass.

## Rebuild

```bash
/home/eroux/pvenvs/1/bin/python synthetic_benchmark/build_shorthand_lexicon.py
```

Then export coverage probes:

```bash
/home/eroux/pvenvs/1/bin/python coverage_report/export_shorthand_stacks.py
```
