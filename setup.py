from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("LICENSE", "r") as f:
    license = f.readline()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

# # TODO: get version from git
# import git_sdk


# def matches_version(tag):
#     return ...


# def resolve_version():
#     for tag in git_sdk.get_tag_ordered():
#         if matches_version(tag):
#             return tag
#     raise ValueError("NoTagMatched")


# version = resolve_version()
version = "0.1"

setup(
    name="mpqp",
    version=version,
    description="Facilitate quantum algorithm development and execution, regardless of the hardware, with MPQP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=license,
    author="ColibriTD",
    author_email="quantum@colibritd.com",
    install_requires=["wheel"] + requirements,
    packages=find_packages(include=["mpqp*"]),
    entry_points={
        "console_scripts": [
            "setup_connections = mpqp.execution.connection.setup_connections:main_setup",
        ]
    },
    package_data={"mpqp.qasm.header_codes": ["*.qasm", "*.inc"]},
)
