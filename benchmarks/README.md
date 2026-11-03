# MPQP performance benchmarks

This suite measures the performance of key MPQP operations without making network
requests or running jobs on a quantum provider.

The current benchmarks cover:

- circuit construction;
- circuit depth calculation;
- circuit conversion to a matrix;
- QASM 2 import and export;
- Pauli-string grouping.

All locally generated results are stored in `.benchmarks/`. This directory is
ignored by Git to keep generated files away from the repository root and prevent
them from being committed accidentally.

## Installation

From the root of the MPQP repository:

```bash
python -m pip install -r requirements-dev.txt
```

## Running benchmarks locally

Run the benchmarks and display the results in the terminal:

```bash
python -m pytest benchmarks --benchmark-only
```

Export the results as JSON:

```bash
python -m pytest benchmarks --benchmark-only --benchmark-json=.benchmarks/results/benchmark-results.json
```

Save a baseline:

```bash
python -m pytest benchmarks --benchmark-only --benchmark-save=baseline
```

`pytest-benchmark` saves this baseline in a subdirectory of `.benchmarks/`
corresponding to the operating system and Python version.

Compare a new run against this baseline and fail if the median execution time
increases by more than 15%. Use the numeric prefix printed when saving the
baseline: for example, `0001` for `0001_baseline.json`.

```bash
python -m pytest benchmarks --benchmark-only --benchmark-compare=0001 --benchmark-compare-fail=median:15%
```

Generate SVG charts from saved measurements:

```bash
pytest-benchmark compare --histogram=.benchmarks/reports/benchmark-histogram
```

The standard `python -m pytest` command continues to run only the functional tests
in `tests/`.

## GitHub Actions workflow

The `.github/workflows/benchmarks.yml` workflow can run:

- automatically after a push or merge to `main`;
- automatically on `perf-benchmark`, also saving the measurement;
- manually from **Actions → Benchmarks dashboard → Run workflow**.

The following inputs are available for manual runs:

| Input             | Description                                                      |
| ----------------- | ---------------------------------------------------------------- |
| `runner`          | Machine used to measure performance.                             |
| `python_version`  | Python version to use.                                           |
| `commit_ref`      | Branch, tag, or SHA to benchmark. Empty uses the current commit. |
| `save`            | Whether to add the result to the permanent history.              |
| `gh_pages_branch` | Target branch in `MPQP-PrivateBenchmark`.                        |

The benchmark suite always comes from the revision that triggers the workflow.
`commit_ref` selects only the MPQP revision to install and measure, so commits
without a `benchmarks/` directory can also be benchmarked. The selected revision
must be installable with the chosen Python version and support the APIs used by
the suite. Benchmarks import the installed package, and the JSON commit metadata
identifies the selected revision.

The workflow first creates its JSON files in a directory identifying the runner
and Python version:

```text
.benchmarks/<system>-CPython-<version>/log/<run>_<commit>.json
```

For example:

```text
.benchmarks/Linux-CPython-3.12/log/42_a1b2c3d4.json
```

The files are available as GitHub Actions artifacts for 90 days. When a measurement
is saved, the same JSON file is also archived alongside the web page in
`MPQP-PrivateBenchmark`.

## How `save` works

With `save=false`, the workflow:

1. runs the benchmarks;
2. produces the JSON file under `.benchmarks/<environment>/log/`;
3. compares the results with the existing history;
4. adds a summary in GitHub Actions;
5. does not push any changes to `MPQP-PrivateBenchmark`.

With `save=true`, it performs the same operations, then adds the measurement and
updates the web page in:

```text
ColibrITD-SAS/MPQP-PrivateBenchmark
```

Automatic runs on `main` and `perf-benchmark` are always saved, even though the
`save` input is not present for push events.

If publishing fails at the `Check benchmark publishing token` step, the secret is
missing. If it fails later during `github-action-benchmark`, check that the token
has not expired, that its organization access is approved, and that it has
**Contents: Read and write** permission on `MPQP-PrivateBenchmark`.

## MPQP-PrivateBenchmark repository layout

Results are separated by operating system and Python version so that comparisons
use measurements from equivalent environments:

```text
MPQP-PrivateBenchmark/
└── dev/
    ├── Linux-CPython-3.12/
    │   ├── index.html
    │   ├── data.js
    │   └── log/
    │       ├── 41_a1b2c3d4.json
    │       └── 42_e5f6a7b8.json
    └── Windows-CPython-3.12/
        ├── index.html
        ├── data.js
        └── log/
            └── 43_c9d0e1f2.json
```

The Linux/Python 3.12 page is available at the following URL when GitHub Pages is
enabled:

```text
https://colibritd-sas.github.io/MPQP-PrivateBenchmark/dev/Linux-CPython-3.12/
```

Each page shows performance over time for each benchmark, along with the
corresponding commit information. Each chart displays the latest 100 measurements.
