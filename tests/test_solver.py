from __future__ import annotations
from typing import Any, List, Tuple

from numpy.typing import NDArray
import numpy
import numpy as np

import wfc.solver


def test_makeWave() -> None:
    wave = wfc.solver.makeWave(3, 10, 20, ground=[-1])
    assert wave.sum() == (2 * 10 * 19) + (1 * 10 * 1)
    assert wave[2, 5, 19] == True
    assert wave[1, 5, 19] == False


def test_entropyLocationHeuristic() -> None:
    wave = numpy.ones((5, 3, 4), dtype=bool)  # everthing is possible
    wave[1:, 0, 0] = False  # first cell is fully observed
    wave[4, :, 2] = False
    preferences: NDArray[np.float_] = numpy.ones((3, 4), dtype=np.float_) * 0.5
    preferences[1, 2] = 0.3
    preferences[1, 1] = 0.1
    heu = wfc.solver.makeEntropyLocationHeuristic(preferences)
    result = heu(wave)
    assert (1, 2) == result


def test_observe() -> None:

    my_wave = numpy.ones((5, 3, 4), dtype=np.bool_)
    my_wave[0, 1, 2] = False

    def locHeu(wave: NDArray[np.bool_]) -> Tuple[int, int]:
        assert numpy.array_equal(wave, my_wave)
        return 1, 2

    def patHeu(weights: NDArray[np.bool_], wave: NDArray[np.bool_]) -> int:
        assert numpy.array_equal(weights, my_wave[:, 1, 2])
        return 3

    assert wfc.solver.observe(my_wave, locationHeuristic=locHeu, patternHeuristic=patHeu) == (
        3,
        1,
        2,
    )


def test_propagate() -> None:
    wave = numpy.ones((3, 3, 4), dtype=bool)
    adjLists = {}
    # checkerboard #0/#1 or solid fill #2
    adjLists[(+1, 0)] = adjLists[(-1, 0)] = adjLists[(0, +1)] = adjLists[(0, -1)] = [
        [1],
        [0],
        [2],
    ]
    wave[:, 0, 0] = False
    wave[0, 0, 0] = True
    adj = wfc.solver.makeAdj(adjLists)
    wfc.solver.propagate(wave, adj, periodic=False)
    expected_result = numpy.array(
        [
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
            ],
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        ]
    )
    assert numpy.array_equal(wave, expected_result)


def test_run() -> None:
    wave = wfc.solver.makeWave(3, 3, 4)
    adjLists = {}
    adjLists[(+1, 0)] = adjLists[(-1, 0)] = adjLists[(0, +1)] = adjLists[(0, -1)] = [
        [1],
        [0],
        [2],
    ]
    adj = wfc.solver.makeAdj(adjLists)

    first_result = wfc.solver.run(
        wave.copy(),
        adj,
        locationHeuristic=wfc.solver.lexicalLocationHeuristic,
        patternHeuristic=wfc.solver.lexicalPatternHeuristic,
        periodic=False,
    )

    expected_first_result = numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    assert numpy.array_equal(first_result, expected_first_result)

    event_log: List[Any] = []

    def onChoice(pattern: int, i: int, j: int) -> None:
        event_log.append((pattern, i, j))

    def onBacktrack() -> None:
        event_log.append("backtrack")

    second_result = wfc.solver.run(
        wave.copy(),
        adj,
        locationHeuristic=wfc.solver.lexicalLocationHeuristic,
        patternHeuristic=wfc.solver.lexicalPatternHeuristic,
        periodic=True,
        backtracking=True,
        onChoice=onChoice,
        onBacktrack=onBacktrack,
    )

    expected_second_result = numpy.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])

    assert numpy.array_equal(second_result, expected_second_result)
    print(event_log)
    assert event_log == [(0, 0, 0), "backtrack", (2, 0, 0)]

    class Infeasible(Exception):
        pass

    def explode(wave: NDArray[np.bool_]) -> bool:
        if wave.sum() < 20:
            raise Infeasible
        return False

    try:
        result = wfc.solver.run(
            wave.copy(),
            adj,
            locationHeuristic=wfc.solver.lexicalLocationHeuristic,
            patternHeuristic=wfc.solver.lexicalPatternHeuristic,
            periodic=True,
            backtracking=True,
            checkFeasible=explode,
        )
        print(result)
        happy = False
    except wfc.solver.Contradiction:
        happy = True

    assert happy
