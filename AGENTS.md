# MPQP pull request review guide

These instructions apply to the whole repository. They are intended for reviews,
not for implementing a change. Do not modify the pull request unless the user asks
for fixes explicitly.

## Review objective

Review MPQP as a user-facing scientific library and as a provider-independent
abstraction over several quantum SDKs. Prioritize observable correctness,
scientific validity, API stability, and regressions over subjective style.

Start by understanding the pull request's stated goal and its complete diff against
the merge base. Review changed behavior in context: inspect callers, sibling
implementations, tests, public exports, and documentation. Do not report unrelated
pre-existing problems unless the pull request makes them worse or relies on them.
Call out unexplained or accidental changes outside the stated scope.

## MPQP design principles

- Preserve provider independence. A user should express a quantum computation in
  MPQP concepts; provider-specific quirks belong in translation, transpilation, or
  provider layers and must not leak into the common API without a strong reason.
- Favor a clear, compact, user-friendly public API over exposing implementation
  details. Use names that describe the quantum concept, not the current internal
  representation or provider implementation.
- Put behavior on the lowest meaningful domain abstraction and keep one source of
  truth. Reuse existing conversion, validation, cloning, and formatting paths
  instead of duplicating their logic across providers or call sites.
- Treat mathematical conventions as API. Qubit order, target order, global phase,
  basis changes, observable layout, measurement semantics, and numeric precision
  must remain coherent through every conversion.
- Make ownership and mutation explicit. Reusing a gate, instruction, circuit,
  observable, or result must not silently mutate another object. Copying APIs must
  honor their documented shallow/deep-copy contract.
- Documentation and examples are part of the product. Public behavior should be
  understandable without reading the implementation, and examples should be
  accurate, pedagogical, and executable.
- Optimize measured bottlenecks without weakening semantics. Avoid unnecessary
  copies, imports, matrix construction, transpilation, or list allocation, but ask
  for evidence when an optimization makes the design harder to reason about.
- Keep changes focused. A worthwhile refactor that is unrelated to the PR should
  normally become a separate issue or PR.

## What to inspect

### 1. Quantum and numerical correctness

Trace affected paths end to end, typically from `QCircuit` or another public object
through job generation, translation/transpilation, provider execution, and
`Result` construction.

Pay particular attention to:

- MPQP/provider endianness and qubit ordering, including Qiskit's reversed Pauli
  convention;
- `targets` and `c_targets`, non-contiguous or reordered targets, partial
  measurements, basis changes, and zero-shot/state-vector jobs;
- preservation of global phase and equivalence up to global phase where applicable;
- gate, circuit, observable, and noise translation in both directions;
- matrix dimensions, power-of-two constraints, unitarity, Hermiticity,
  normalization, symbolic values, and complex precision (normally
  `numpy.complex128` when precision matters);
- floating-point comparisons and doctest output: use tolerant comparisons or
  `ELLIPSIS` where exact decimal representation is not stable;
- noise application order, gate eligibility, device connectivity, and behavior on
  simulated versus real devices;
- device/job compatibility and unsupported features. Never imply real-hardware
  support that has neither authoritative support nor a safe test strategy.

A translation that produces an equivalent but unnecessarily decomposed circuit can
still be a regression when MPQP has a native representation for the original
operation, because it harms readability and hardware optimization.

### 2. State, ownership, and side effects

Check constructors, properties, cloning helpers, caches, and conversion methods for:

- mutation of the input circuit, measurement, observable, instruction, or provider
  object;
- shared mutable defaults such as `[]` or `{}`;
- shallow copies presented as deep copies, or redundant `deepcopy` calls;
- one instruction reused in multiple positions or circuits;
- stale cached environment/provider state;
- public and private attributes representing the same state and drifting apart;
- partially initialized objects, especially code using `__new__` or bypassing the
  normal constructor;
- returned internal arrays or lists that users can mutate unexpectedly.

### 3. Public API and compatibility

Treat signatures, import paths, names, defaults, exceptions, warnings, `repr`, and
`str` as compatibility-sensitive.

- Prefer MPQP-level types and concepts in the public API.
- Verify that new public objects are exported from the appropriate facade module
  when they are meant to be discoverable.
- Preserve established defaults and behavior unless the change is intentional and
  documented.
- Validate inputs at the boundary and raise an informative, appropriate exception.
- Use warnings for supported-but-lossy conversions; do not silently discard quantum
  information.
- Keep `repr` faithful to the object and consistent with displayed names. For
  reconstructible MPQP objects, preserve the repository convention
  `eval(repr(value)) == value`.
- Avoid exposing details such as a concrete simulator class, static/dynamic
  implementation strategy, or provider cache layout when users do not need them.
- Ensure compatibility with Python 3.10 through 3.13. Syntax and annotations must
  work on Python 3.10; use `from __future__ import annotations` when needed.

### 4. Providers, optional dependencies, and remote execution

MPQP can be installed with individual provider extras. A change for one provider
must not require every provider SDK merely to import `mpqp`.

- Keep heavy or optional SDK imports delayed until the relevant provider or
  conversion is used.
- Test the feature with the smallest relevant extra, not only an `all` environment.
- Keep translation that depends only on a language separate from transpilation that
  depends on a concrete device or backend.
- Key transpilation or observable caches with every device/backend property that
  affects the output; never reuse an artifact prepared for another device.
- Preserve parity across providers when the feature is common, while respecting
  explicitly documented provider limitations.
- Ordinary tests and doctests must not authenticate, submit remote jobs, consume
  credits, open UI windows, or rely on network access. Remote/costly tests belong
  behind the repository's explicit long/remote/costly controls.

### 5. Tests

Require a regression test for each fixed bug and focused coverage for new behavior.
Tests under `tests/` broadly mirror the source tree.

Good MPQP tests should cover, as relevant:

- the common MPQP behavior and each affected provider;
- exact/sample/observable job types and meaningful shot values, including zero;
- empty, single-qubit, multi-qubit, reordered, non-contiguous, and partial targets;
- invalid dimensions, duplicate targets, unsupported operations, and other boundary
  failures;
- symbolic and numeric parameters;
- round trips and semantic equivalence across translations;
- non-mutation and reuse of the same object in more than one circuit;
- deterministic random behavior with an explicit seed;
- stable public representation and imports.

Prefer an independent expected value or small hand-computed example. A test that
computes its expected result through the same helper as the implementation can
reproduce the same bug and is weak evidence. Use `pytest.mark.parametrize` for
meaningful families of cases and the appropriate `pytest.mark.provider(...)` for
provider-specific tests.

Docstring examples are executed by `tests/test_doc.py`. Provider-dependent examples
need the matching doctest flag (`QISKIT`, `CIRQ`, `BRAKET`, `MYQLM`, and any newer
repository-supported equivalent). Use `SKIP` only when execution is inherently
remote, interactive, costly, or otherwise unsafe—not to conceal a broken example.

### 6. Documentation

All user-facing documentation is in English. Public classes, functions, methods,
arguments, exceptions, warnings, and changed semantics need documentation. Check
the generated Sphinx structure, not only the source text.

Docstrings broadly use Google style, omit types already present in annotations, and
keep optional sections in this order:

1. `Args`
2. `Returns`
3. `Warns`
4. `Raises`
5. `Example` or `Examples`
6. `Note`
7. `Warning`

Argument descriptions start with an uppercase letter and end with a period. Prefer
links to documented MPQP objects over vague references. Explain scientific
conventions, target ordering, units, and lossy behavior explicitly. Examples and
notebooks should teach the concept, use established quantum terminology, and match
the current API.

### 7. Maintainability and repository hygiene

- Search for an existing helper or equivalent provider implementation before
  accepting duplicated logic.
- Prefer straightforward control flow, early validation, standard Python protocols,
  and domain classes over stringly typed dispatch.
- Check annotations with the repository's strict `pyrightconfig.json`; do not add
  an ignore merely to silence a real mismatch.
- Formatting follows Black with line length 88 and single-quote preservation, plus
  isort's Black profile.
- Reject debugging prints, accidental databases/build artifacts, dead imports,
  completed TODO comments, unexplained dependency changes, and unrelated generated
  files.
- Security, credential handling, local storage, and remote submission paths require
  explicit failure handling and must not expose tokens or user data.

## Review procedure

1. Read the PR description, linked issue, commits, and full merge-base diff. State
   any missing context that limits the review.
2. Map changed public behavior and affected providers. Inspect surrounding code,
   callers, sibling provider implementations, exports, tests, and docs.
3. Exercise at least one representative user path mentally or locally from public
   API to result. For mathematical changes, verify a minimal case by hand or with an
   independent oracle.
4. Run the narrowest relevant tests first. Expand to provider tests, doctests,
   Pyright, and the wider suite in proportion to the risk. Never claim a command
   passed unless it was actually run.
5. Check packaging/import behavior when dependencies or providers change. Check the
   documentation build when public docs, docstrings, notebooks, or exports change.
6. Re-read the diff for accidental edits, missing tests/docs, debug output, and
   scope creep before writing findings.

Useful commands include:

```bash
python -m pytest <relevant-test-path> --providers <provider>
python -m pytest --providers all
python -m pytest --long-local --providers all
pyright
black --check mpqp tests
isort --check-only mpqp tests
sphinx-build -b html docs build
```

Adapt commands to the installed extras. Do not run remote or credit-consuming tests
without explicit authorization.

## How to report findings

Lead with findings, ordered by severity. Report only issues introduced or exposed by
the PR that the author can act on. Use these levels:

- **P0**: catastrophic or irreversible impact, such as credential exposure, data
  loss, or broadly incorrect paid/remote execution.
- **P1**: incorrect scientific result, broken core workflow/import/installation,
  unexpected paid execution, or major public API regression.
- **P2**: functional bug in a supported case, provider inconsistency, meaningful
  side effect, missing boundary validation, or regression with a practical impact.
- **P3**: localized low-impact correctness, documentation, testing, or
  maintainability issue worth fixing before merge.

Each finding must:

- have a concise imperative title with its priority;
- point to the smallest relevant changed line range;
- describe the concrete input, provider, or execution path that triggers it;
- explain the observable consequence, not merely a preference;
- suggest the direction of a fix when it is not obvious;
- distinguish verified behavior from an inference or an open question.

Do not inflate style preferences into blockers. Avoid vague comments such as
"why?", "this looks wrong", or "add tests" without naming the risk and missing
case. Put optional design ideas and questions in a separate non-blocking section.
If no actionable finding remains, say so explicitly and mention any tests or areas
that could not be verified.

Request changes for unresolved P0/P1 findings and normally for P2 findings that can
produce wrong results or unsafe side effects. Approval is appropriate only when the
tested behavior, public contract, provider boundaries, and documentation are
coherent.
