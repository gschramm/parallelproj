from __future__ import annotations

import parallelproj
from types import ModuleType

# import the global pytestmark variable containing the xp/dev matrix we
# want to test
from config import pytestmark


def test_calc_chunks(xp: ModuleType, dev: str) -> None:
    assert parallelproj.backend.calc_chunks(10, 3) == [0, 4, 7, 10]
    assert parallelproj.backend.calc_chunks(10, 2) == [0, 5, 10]
    assert parallelproj.backend.calc_chunks(10, 1) == [0, 10]

    assert parallelproj.backend.calc_chunks(1, 1) == [0, 1]


def test_event_multiplicity(xp: ModuleType, dev: str) -> None:

    events = xp.asarray(
        [
            [2, 1, 1, 1, 1],
            [1, -1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    mu = parallelproj.count_event_multiplicity(events)

    assert xp.all(mu == xp.asarray([2, 1, 4, 4, 4, 2, 4], device=dev))
