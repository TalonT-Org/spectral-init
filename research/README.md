# Research artifacts

Each dated directory holds one experiment: its plan, scripts, data and report.

## Reproducing an experiment

The scripts under `research/**/scripts` hard-code the Cargo features, binary
names and test names the repository had before the 0.1 release work. They
reproduce at the git tag `research-baseline-pre-0.1`:

```sh
git checkout research-baseline-pre-0.1
```

Run them from that checkout, not from `main`.

## Data files

Some experiments read arrays produced by another experiment or by the
visual-evaluation tooling in `tests/visual_eval/`. Where that is the case, the
experiment's data directory has a `DATA.md` naming each source file and the
script that regenerates it.
