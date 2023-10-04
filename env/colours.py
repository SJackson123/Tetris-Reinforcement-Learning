"""
Colours class - define piece colours.
Code available from: https://github.com/educ8s/Python-Tetris-Game-Pygame
"""

class Colours():
    # black = (0,0,0)
    dark_grey = (40, 31, 26)
    green = (23, 230, 47)
    red = (18, 18, 232)
    orange = (17, 116, 226)
    yellow = (4, 234, 237)
    purple = (246, 0, 166)
    cyan = (209, 204, 21)
    blue = (216, 64, 13)

    @classmethod
    # cls is a reference to the colours class
    def get_cell_colours(cls):
        return [cls.dark_grey, cls.green, cls.red, cls.orange, 
                cls.yellow, cls.purple, cls.cyan, cls.blue]