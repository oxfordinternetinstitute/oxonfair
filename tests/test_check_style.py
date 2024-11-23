import logging
import warnings
from subprocess import Popen, PIPE, check_call
import linkcheckmd as lc
import glob
# Owing to dependency issues with github actions this file is excluded from the tests.
# run pip install nbtest before running test_run_notebooks_without_errors


def test_check_style_codebase():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(["flake8",   'src/oxonfair/',
                         "--count",
                         "--max-line-length", "150",],
                        stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    print(lines)
    count = int(lines[-1].decode())
    if count > 0:
        warnings.warn(f"{count} PEP8 warnings remaining")
    assert count < 10, "Too many PEP8 warnings found, improve code quality to pass test."


def test_check_style_tests():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(["flake8",   'test',
                         "--count",
                         "--max-line-length", "150",],
                        stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    count = int(lines[-1].decode())
    if count > 0:
        warnings.warn(f"{count} PEP8 warnings remaining")
    assert count < 10, "Too many PEP8 warnings found, improve code quality to pass test."


def test_check_style_examples():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(["flake8",   'examples',
                         "--count",
                         "--max-line-length", "150",],
                        stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    count = int(lines[-1].decode())
    if count > 0:
        warnings.warn(f"{count} PEP8 warnings remaining")
    assert count < 10, "Too many PEP8 warnings found, improve code quality to pass test."


def test_md_links():
    missing_links = lc.check_links('./', ext='.md', recurse=True, use_async=False)
    missing_links_eg = lc.check_links('./examples/', ext='.md', recurse=True, use_async=False)

    for link in missing_links:
        warnings.warn(link)

    for link in missing_links_eg:
        warnings.warn(link)

    assert missing_links_eg == []
    assert missing_links == []


def test_run_notebooks_without_errors():
    "run pip install nbtest before running this test"
    for file in glob.glob('./examples/*.ipynb'):
        check_call(['pytest', '--nbmake',  '--nbmake-timeout=500', file])


def test_compress_notebooks():
    # Now compress notebooks because running test makes them too large
    # This is not really a test, hijacking the test suite to build.
    from ipynbcompress import compress
    for file in glob.glob('./examples/*.ipynb'):
        compress(file)
