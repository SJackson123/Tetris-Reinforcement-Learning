"""
Grid object class.
Code available from: https://github.com/educ8s/Python-Tetris-Game-Pygame
"""

import cv2 as cv
import copy

from .colours import Colours

class Grid():
    def __init__(self, num_rows=20, num_cols=10):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = 30
        self.grid = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        self.colours = Colours.get_cell_colours()


    def is_empty(self, row, column):
        """Return true if a grid cell is empty."""
        if self.grid[row][column] == 0:
            return True
        return False
    
    def is_inside(self, row, column):
        """Return true if a the position of a tile is inside the board boundaries."""
        if row >= 0 and row < self.num_rows and \
            column >= 0 and column < self.num_cols:
            return True
        return False
    
    def is_row_full(self, row, grid):
        for column in range(self.num_cols):
            if grid[row][column] == 0:
                return False

        return True

    def clear_row(self, row, grid):
        for column in range(self.num_cols):
             grid[row][column] = 0

        return grid

    def move_row_down(self, row, num_rows, grid):
        for column in range(self.num_cols):
            grid[row+num_rows][column] = grid[row][column]
            grid[row][column] = 0

        return grid

    def clear_full_rows(self, grid):
        """clear full rows does not change the grid passed to it!"""
        copy_grid = copy.deepcopy(grid)
        completed = 0
        for row in range(self.num_rows-1, -1, -1):
            if self.is_row_full(row, copy_grid):
                self.clear_row(row, copy_grid)
                # print(f'grid is: {grid}')
                completed += 1
            elif completed > 0:
                grid = self.move_row_down(row, completed, copy_grid)
        
        return completed, copy_grid
    
    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)

        if len(to_delete) > 0:
            for index in to_delete[::-1]:  # Reverse the order to prevent index errors
                del board[index]
            # Add new rows at the top with all zeroes
            for _ in range(len(to_delete)):
                board.insert(0, [0 for _ in range(self.num_cols)])

        return len(to_delete), board


    def print_grid(self):
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                print(self.grid[row][column], end=" ")
            print()

    def print_current_grid(self, copy_grid):
        """Print out the grid for debugging purposes."""
        for row in copy_grid:
            for i in row:
                print(i, end=" ")
            print()

    def reset(self):
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                self.grid[row][column] = 0
    
    def draw_grid(self, img):
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                # assign a cell value to keep trakc of whats currently in cell
                cell_value = self.grid[row][column]
                # (img, (x1,y1), (x2, y2), colour)
                cell_rect = cv.rectangle(img, (column * self.cell_size+1, row * self.cell_size+1),
                                         (column * self.cell_size + self.cell_size-1, row * self.cell_size + self.cell_size-1),
                                          self.colours[cell_value], thickness=cv.FILLED)
        # cv.imshow('Grid', cell_rect)    
        # cv.waitKey(0)
        return cell_rect

    