# Contribution Guidelines

When contributing to `MPQP`, whether on GitHub or in other community spaces:

- Be respectful, civil, and open-minded.
- If you want to make code changes based on your personal opinion(s), make sure
  you open an issue first describing the changes you want to make, and open a
  pull request only when your suggestions get approved by maintainers.

## How to Contribute

### Prerequisites

In order to not waste your time implementing a change that has already been
declined, or is generally not needed, start by
[opening an issue](https://github.com/ColibrITD-SAS/mpqp/issues/new/choose)
describing the problem you would like to solve.

### Setup your environment locally

\_Some commands will assume you have the GitHub CLI installed, if you haven't,
consider [installing it](https://github.com/cli/cli#installation), but you can
always use the Web UI if you prefer that instead.

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

In the documentation, some notebooks are rendered to HTML, this process is done
using [Pandoc](https://pandoc.org/). Unfortunately, this software cannot be
installed from the pip repository, so you need to install it separately. You can
check [their documentation](https://pandoc.org/installing.html) to see how to
install it on your OS (you can find it on most package manager: `apt`,
`yum`, `pacman`, `choco`, `winget`, `brew` and more... )

The last (optinal) step is to setup a GitHub personal access tokens to enable
the sphinx automatic changelog generation. This step is only important if you
want to preview this changelog generation on your personal computer. Not being
able to generate it will not affect the rest of the documentation, and even less
the rest of the library.

In order to generate the token, you can read about it on
[this page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).
The only scope you need for this token is the `public_repo` one. Once you
generated it, duplicate the `.env.example` file at the root of this repository,
and rename it to `.env`, the replace `...` by your token and your all set!

### Our tech stack

Here are the pieces of software useful to know to work on our library:

- All the code of the Library is in [Python](https://www.python.org).
- We run our tests using [pytest](https://docs.pytest.org).
- We generate our documentation using [Sphinx](https://www.sphinx-doc.org).
- We format the code using [black](https://black.readthedocs.io) and
  [isort](https://pycqa.github.io/isort).
- We check our types using [Pyright](https://microsoft.github.io/pyright), but
  this is not configured yet.
- The documentation is automatically deployed on new versions with
  [GitHub Actions](https://docs.github.com/en/actions) (as well as a few other
  bits and bobs).

### Implement your changes

This project is organized as such:

- `mpqp/` contains the source code of the library;
- `docs/` contains the source code of the documentation, but working on the
  documentation will require you to also get comfortable with the source code,
  since the biggest part of the documentation is as docstrings in the library
  source code (using the `autodoc` Sphinx extension);
- all the source files requiring testing are mirrored from `mpqp/` to `tests/`.
  We do not consider all code need to be tested, but the "tricky" (error prone)
  code should be covered a minimum amount (we are actively trying to improve the
  state of testing in our library currently).

Strong of this knowledge, you should be able to go through the files of this
repository, and find the one you need to modify to achieve your goal.

Here are some useful scripts for when you are developing:

| Command                               | Description                               |
| ------------------------------------- | ----------------------------------------- |
| `sphinx-build -b html docs build`     | Builds the documentation                  |
| `python -m pytest`                    | Runs the test suite                       |
| `python -m pytest --long`             | Runs the long tests too                   |
| `python -m pytest --long-local`       | Runs the local long tests                 |
| `python -m pytest --seed=<your_seed>` | Runs the test suite with a specified seed |

When making commits, make sure to follow the
[conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)
guidelines, i.e. prepending the message with `feat:`, `fix:`, `doc:`, etc... You
can use `git status` to double check which files have not yet been staged for
commit:

```bash
git add <file> && git commit -m "feat/fix/doc: commit message"
```

Note that our docstrings follow broadly the
[Google format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
One example is worth one thousand words so you can also have a look of
[the Sphinx examples](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for guidance. Types are not needed in the documentation as we use type hints
(and so should be included to avoid data duplication). In addition, because of
our automatic web documentation generation, the order of sections has to be the
following (even though each of these sections is optional):

1. `Args`
2. `Returns`
3. `Warns`
4. `Raises`
5. `Example(s)`
6. `Note`
7. `Warning`

### When you're done

We would like you to format your code using `black` and `isort`, and check that
your type hints are coherent using `Pyright`, but there are not configured yet.
This should be dealt with shortly.

Please make sure your changes are working as expected (and that you didn't break
any previous feature) by making manual, running the automated tests and adding
now ones corresponding to your feature.

When all that's done, it's time to file a pull request to upstream:

```bash
gh pr create --web
```

and fill out the title and body appropriately.

## Translations

For now, we only support the English language. If you would like to start a
translation of the documentation, get in touch with us so we set it up together!

## Credits

This documented was inspired by the contributing guidelines for
[t3-oss/create-t3-app](https://github.com/t3-oss/create-t3-app/blob/main/CONTRIBUTING.md).

## A note about `qiskit`

When `qiksit` went from version `0.x` to version `1.x`, the migration caused
problems. In order to facilitate the migration, we provide a shorthand to
uninstall all `qiskit` packages: you can simply run the two following commands

```sh
pip uninstall -y -r all-qiskit.txt
pip install -r requirements.txt
```
