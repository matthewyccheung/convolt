# Contributing

Thanks for considering a contribution.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Sanity checks

There is no unit test suite yet. Before opening a PR:

```bash
python -m compileall -q reg scripts
```

Optionally run the Learn2Reg smoke script (requires data on disk):
```bash
python scripts/smoke_learn2reg.py -h
```

## What to include

- Keep changes focused and minimal.
- If you add a CLI flag or output file, update the relevant docs under `docs/`.

