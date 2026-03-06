# Copyright 2024, All rights reserved.
#
# Licensed under the MIT License.

from ament_pep257.main import main
import pytest


@pytest.mark.linter
def test_pep257():
    rc = main(argv=['--add-ignore', 'D100,D101,D102,D103,D104,D105'])
    assert rc == 0, 'Found code style errors / warnings'
