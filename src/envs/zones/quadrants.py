import enum


class Quadrant(enum.Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

# dict[str, list[Quadrant]]