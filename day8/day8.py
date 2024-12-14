from pathlib import Path
from itertools import combinations
import numpy as np

from numpy.typing import NDArray


path = Path("day7/input.txt")


def read_input(path: Path) -> list[str]:
    return path.open().readlines()

type Position = NDArray

def symbol_to_number(symbol: str) -> int:
    if symbol == ".":
        return 0
    elif symbol == "#":
        return 1
    raise ValueError

def check_antinode(antinode, shape):
    return (
        antinode[0] >= 0
        and antinode[0] < shape[0]
        and antinode[1] < shape[1]
        and antinode[1] >= 0
    )

def yield_antinodes_in_direction(point, direction, shape):
    next_antinode = point.copy()
    while True:
        next_antinode += direction
        if not check_antinode(next_antinode, shape):
            break
        yield next_antinode

def get_position_of_antinodes(
    ant1: Position, ant2: Position, shape: tuple[int, int]
) -> list[Position, Position]:
    diff = ant2 - ant1
    antinodes = [ant1, ant2]
    for antinode in yield_antinodes_in_direction(ant1.copy(), -diff, shape):
      antinodes.append(antinode.copy())
    for antinode in yield_antinodes_in_direction(ant2.copy(), diff, shape):
        antinodes.append(antinode.copy())  

    return antinodes


input = read_input(path)
field_matrix = np.array([[symbol for symbol in line.strip()] for line in input])
shape = field_matrix.shape
antinode_matrix = np.zeros(field_matrix.shape)

unique_values = np.unique(field_matrix)
unique_antennas = np.delete(unique_values, np.where(unique_values == "."))
print(unique_antennas)

for antenna in unique_antennas:
    antenna_positions = np.where(field_matrix == antenna)
    positions = [
        np.array([antenna_positions[0][i], antenna_positions[1][i]])
        for i in range(len(antenna_positions[0]))
    ]
    for i, j in combinations(positions, 2):
        antinodes = get_position_of_antinodes(i, j, shape)
        for antinode in antinodes:
            assert(check_antinode(antinode, shape))
            antinode_matrix[antinode[0], antinode[1]] = 1
sum = antinode_matrix.sum()
print(antinode_matrix)
print(sum)
