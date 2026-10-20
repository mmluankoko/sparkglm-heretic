#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Include Heretic regressions in SparkGLM's existing launch-contract CI step."""
from pathlib import Path
import unittest


def load_tests(loader, tests, pattern):
    return unittest.TestLoader().discover(str(Path(__file__).parent / 'heretic/tests'))
