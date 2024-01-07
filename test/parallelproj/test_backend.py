import parallelproj


def test_calc_chunks() -> None:

    assert (parallelproj.backend.calc_chunks(10, 3) == [0, 4, 7, 10])
    assert (parallelproj.backend.calc_chunks(10, 2) == [0, 5, 10])
    assert (parallelproj.backend.calc_chunks(10, 1) == [0, 10])

    assert (parallelproj.backend.calc_chunks(1, 1) == [0, 1])
