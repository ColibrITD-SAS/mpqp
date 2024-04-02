# Contribution Guidelines

When contributing to `MPQP`, whether on GitHub or in other community spaces:

- Be respectful, civil, and open-minded.
- If you want to make code changes based on your personal opinion(s), make sure
  you open an issue first describing the changes you want to make, and open a
  pull request only when your suggestions get approved by maintainers.

## How to Contribute

### Prerequisites

In order to not waste your time implementing a change that has already been
declined, or is generally not needed, start by [opening an
issue](https://github.com/ColibrITD-SAS/mpqp/issues/new/choose) describing the
problem you would like to solve.

### Setup your environment locally

_Some commands will assume you have the Github CLI installed, if you haven't,
consider [installing it](https://github.com/cli/cli#installation), but you can
always use the Web UI if you prefer that instead._

In order to contribute to this project, you will need to fork the repository:

```bash
gh repo fork ColibrITD-SAS/mpqp
```

then, clone it to your local machine:

```bash
gh repo clone <your-github-name>/mpqp
```

To install all the dependencies needed to contribute (documentation, tests,
etc... included), use pip:

```bash
pip install -r requirements-dev.txt
```

### Our tech stack

Here are the pieces of software useful to know to work on our library:

- All the code of the Library is in [Python](https://www.python.org).
- We run our tests using [pytest](https://docs.pytest.org).
- We generate our documentation using [Sphinx](https://www.sphinx-doc.org).
- We format the code using [black](https://black.readthedocs.io), but this is not
  configured yet.
- We check our types using [Pyright](https://microsoft.github.io/pyright), but
  this is not configured yet.
- The documentation is automatically deployed on new versions with
  [GitHub Actions](https://docs.github.com/en/actions). (As well as a few other
  bits and bobs)

### Implement your changes

This project is organized as such:

- `mpqp/` contains the source code of the library;
- `docs/` contains the source code of the documentation, but working on the
  documentation will require you to also get comfortable with the source code,
  since the biggest part of the documentation is as docstrings in the library
  source code (using the `autodoc` Sphinx extension);
- all the source files requiring testing are mirrored from `mpqp\` to `tests\`.
  We do not consider all code need to be tested, but the "tricky" (error prone)
  code should be covered a minimum amount (we are actively trying to improve the
  state of testing in our library currently).

Strong of this knowledge, you should be able to go through the files of this
repository, and find the one you need to modify to achieve your goal.

Here are some useful scripts for when you are developing:

| Command                           | Description              |
| --------------------------------- | ------------------------ |
| `sphinx-build -b html docs build` | Builds the documentation |
| `python -m pytest`                | Runs the test suite      |
| `python -m pytest -l`             | Runs the long tests too  |

When making commits, make sure to follow the
[conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)
guidelines, i.e. prepending the message with `feat:`, `fix:`, `doc:`, etc...
You can use `git status` to double check which files have not yet been staged
for commit:

```bash
git add <file> && git commit -m "feat/fix/doc: commit message"
```

### When you're done

We would like you to format your code using `black`, and check that your type
hints are coherent using `Pyright`, but there are not configured yet. This
should be dealt with shortly.

Please make sure your changes are working as expected (and that you didn't break
any previous feature) by making manual, running the automated tests and adding
now ones corresponding to your feature.

When all that's done, it's time to file a pull request to upstream:

```bash
gh pr create --web
```

and fill out the title and body appropriately.

## Translations

For now, we only support the english language. If you would like to start a
translation of the documentation, get in touch with us so we set it up together!

## Credits

This documented was inspired by the contributing guidelines for
[t3-oss/create-t3-app](https://github.com/t3-oss/create-t3-app/blob/main/CONTRIBUTING.md).
