import logging
import warnings
from subprocess import Popen, PIPE
import linkcheckmd as lc


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
    missing_links = lc.check_links('./', ext='.md')
    for link in missing_links:
        warnings.warn(link)
    assert missing_links == []


def test_run_notebooks_without_errors():
    Popen('pytest' '--nbmake', '--overwrite', '-n=auto', 'examples')
