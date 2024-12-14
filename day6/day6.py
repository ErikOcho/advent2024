from pathlib import Path
from numpy.typing import NDArray
from typing import Iterable

import numpy as np

path = Path("day6/input.txt")


def read_input(path: Path) -> list[str]:
    return path.open().readlines()


type Position = tuple[int, int]

def symbol_to_number(symbol: str) -> int:
    if symbol == ".":
        return 0
    elif symbol == "#":
        return 1
    raise ValueError


def get_starting_position(matrix: NDArray):
    position = np.where(matrix == "^")
    assert len(position[0]) == 1
    assert len(position[1]) == 1
    return (position[0][0], position[1][0])


def is_in_field(position: Position, shape: tuple[int, int]) -> bool:
    return (
        position[0] < shape[0]
        and np.abs(position[1]) < shape[1]
        and position[0] >= 0
        and position[1] >= 0
    )


def get_next_position(position: Position, direction: int):
    if direction == 0:
        return (position[0] - 1, position[1])
    elif direction == 1:
        return (position[0], position[1] + 1)
    elif direction == 2:
        return (position[0] + 1, position[1])
    elif direction == 3:
        return (position[0], position[1] - 1)
    else:
        raise ValueError


def mark_field(field, position, direction):
    if field[position[0]][position[1]] == direction:
        return False
    field[position[0]][position[1]] = direction
    return True


def get_field_value(field, position):
    return field[position[0]][position[1]]


def count_values(matrix):
    return np.sum(matrix)


def has_loop(matrix) -> bool:
    shape = matrix.shape
    check_matrix = np.zeros(shape) + 10
    actual_position: Position = get_starting_position(matrix)
    direction = 0

    while is_in_field(actual_position, shape):
        if not mark_field(check_matrix, actual_position, direction):
            return True
        if not is_in_field(get_next_position(actual_position, direction), shape):
            break
        while (
            get_field_value(matrix, get_next_position(actual_position, direction))
            == "#"
        ):
            # rotate direction
            direction = (direction + 1) % 4
        # move
        actual_position = get_next_position(actual_position, direction)
    return False


def add_hash(matrix) -> Iterable[NDArray]:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if not (matrix[row][col] == "#" or matrix[row][col] == "^"):
                matrix_copy = matrix.copy()
                matrix_copy[row][col] = "#"
                yield matrix_copy


if __name__ == "__main__":
    input = read_input(path)
    field_matrix = np.array([[symbol for symbol in line.strip()] for line in input])
    counter = 0
    for matrix in add_hash(field_matrix):
        if has_loop(matrix):
            counter += 1
    print(counter)
