from pathlib import Path

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

long_description = long_description.replace(
    "resources/dark-logo.svg",
    "https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/resources/dark-logo.svg",
).replace(
    "resources/mpqp-usage.gif",
    "https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/resources/mpqp-usage.gif",
)

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

providers_dir = Path("requirements_providers")
extras = {}
all_extras = []
for f in providers_dir.glob("*.txt"):
    provider_name = f.stem
    with open(f, "r", encoding="utf-8") as fh:
        extra = [line.strip() for line in fh if line.strip()]
        extras[provider_name] = extra
        all_extras.extend(extra)
extras["all"] = sorted(set(all_extras))

setup(
    name="mpqp",
    use_scm_version=True,
    description="Facilitate quantum algorithm development and execution, regardless of the hardware, with MPQP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    author="MPQP Development Team",
    author_email="quantum@colibritd.com",
    keywords=[
        "mpqp",
        "quantum programming language",
        "sdk",
    ],
    url="https://colibritd.com",
    python_requires=">=3.10,<=3.13",
    install_requires=["wheel"] + requirements,
    extras_require=extras,
    packages=find_packages(include=["mpqp*"]),
    entry_points={
        "console_scripts": [
            "setup_connections = mpqp_scripts.setup_connections:main_setup",
            "update_qiskit = mpqp_scripts.update_qiskit:update_packages",
        ]
    },
    project_urls={
        "Repository": "https://github.com/ColibrITD-SAS/mpqp",
        "Documentation": "https://mpqpdoc.colibri-quantum.com/",
        "License": "https://github.com/ColibrITD-SAS/mpqp/blob/main/LICENSE",
        "Issues": "https://github.com/ColibrITD-SAS/mpqp/issues",
        "Changelog": "https://mpqpdoc.colibri-quantum.com/changelog",
    },
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Quantum Computing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    package_data={"mpqp.qasm.header_codes": ["*.qasm", "*.inc"]},
)
