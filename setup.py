import os

from setuptools import setup

FAIR = "oxonfair"

version = "0.2.1.8"

PYTHON_REQUIRES = ">=3.8"


def create_version_file(*, version):
    print("-- Building version " + version)
    version_path = os.path.join("src", FAIR, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {FAIR} version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


def update_version(version, use_file_if_exists=True, create_file=False):
    """
    To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish
    a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml. **Currently disabled**.
    You need to increase the version number after stable release, so that the nightly pypi can work properly.
    """
    try:
        if not os.getenv("RELEASE"):
            from datetime import date

            minor_version_file_path = "VERSION.minor"
            if use_file_if_exists and os.path.isfile(minor_version_file_path):
                with open(minor_version_file_path) as f:
                    day = f.read().strip()
            else:
                today = date.today()
                day = today.strftime("b%Y%m%d")
            version += day
    except Exception:
        pass
    if create_file and not os.getenv("RELEASE"):
        with open("VERSION.minor", "w") as f:
            f.write(day)
    return version


def default_setup_args(*, version):
    from setuptools import find_packages

    long_description = open("README.md").read()
    name = f"{FAIR}"
    setup_args = dict(
        name=name,
        version=version,
        author="Governance of Emerging Technologies Programme (Oxford Internet Insitute)",
        url="https://github.com/oxfordinternetinstitute/oxonfair",
        description="Toolkit for evaluating and enforcing ML model fairness",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        license_files=("LICENSE", "NOTICE"),
        # Package info
        package_dir={"": "src"},
        packages=find_packages("src"),
        namespace_packages=[],
        zip_safe=True,
        include_package_data=True,
        python_requires=PYTHON_REQUIRES,
        package_data={
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Customer Service",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Healthcare Industry",
            "Intended Audience :: Telecommunications Industry",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
        project_urls={
            "Documentation": "https://github.com/oxfordinternetinstitute/oxonfair",
            "Bug Reports": "https://github.com/oxfordinternetinstitute/oxonfair/issues",
            "Source": "https://github.com/oxfordinternetinstitute/oxonfair",
            "Contribute!": "https://github.com/oxfordinternetinstitute/oxonfair/blob/master/CONTRIBUTING.md",
        },
    )
    return setup_args


install_requires = [
    "numpy>=1.21.4",
    "pandas>=1.2.5",
    "scikit-learn",
    'ucimlrepo'
]

extras_require = dict()
full_requirements = ['matplotlib', 'autogluon.tabular', 'torch', 'xgboost', 'jupyterlab']
notebook_requirements = full_requirements + ['fairlearn', 'fairret']
test_requirements = notebook_requirements + ["tox", "pytest", "pytest-cov", 'flake8', 'tabulate',
                                             'linkcheckmd', 'ipynbcompress', 'nbmake']

full_requirements = list(set(full_requirements))
notebook_requirements = list(set(notebook_requirements))
test_requirements = list(set(test_requirements))
extras_require['full'] = full_requirements
extras_require['notebooks'] = notebook_requirements
extras_require["tests"] = test_requirements


if __name__ == "__main__":
    create_version_file(version=version)
    setup_args = default_setup_args(version=version)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
