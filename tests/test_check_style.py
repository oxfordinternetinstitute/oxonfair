import logging
import warnings
from subprocess import Popen, PIPE


def test_check_style_codebase():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(["flake8",   'src/oxonfair/',
                         "--count",
                         "--max-line-length", "150",],
                        stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
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
