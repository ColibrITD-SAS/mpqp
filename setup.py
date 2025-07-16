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


setup(
    name="mpqp",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Facilitate quantum algorithm development and execution, regardless of the hardware, with MPQP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    author="ColibriTD",
    author_email="quantum@colibritd.com",
    url="https://colibritd.com",
    install_requires=["wheel"] + requirements,
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
        "Company": "https://colibritd.com",
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    package_data={"mpqp.qasm.header_codes": ["*.qasm", "*.inc"]},
)
