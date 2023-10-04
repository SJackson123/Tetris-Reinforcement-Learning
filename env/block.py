"""
Block object class.
Code available from: https://github.com/educ8s/Python-Tetris-Game-Pygame
"""

import cv2 as cv
from .colours import Colours

class Block():
    def __init__(self, id):
            # distinguish between different blocks
            self.id = id
            # attribute to store occupied cells for each rotation state of block
            self.cells = {}
            self.cell_size = 30
            self.row_offset = 0
            self.column_offset = 0
            self.rotation_state = 0
            self.colours = Colours.get_cell_colours()
      
    def move(self, rows, columns):
         self.row_offset += rows
         self.column_offset += columns

    def get_cell_positions(self):
        """Return list of positions of occupied cells with the offset applied."""
        # print(f'rotation state: {self.rotation_state}')
        tiles = self.cells[self.rotation_state]
        moved_tiles = []
        # row, column
        for position in tiles:
            position = (position[0] + self.row_offset, position[1] + self.column_offset)
            moved_tiles.append(position)
        
        return moved_tiles

    def draw_block(self, img):
        # tile is list of positions for current tetrimino with given rotation
        tiles = self.get_cell_positions()
        for tile in tiles:
            #tile[0] - row
            tile_rect = cv.rectangle(img, (tile[1] * self.cell_size+1, tile[0] * self.cell_size+1),
                                            (tile[1] * self.cell_size + self.cell_size-1, tile[0] * self.cell_size + self.cell_size-1),
                                            self.colours[self.id], thickness=cv.FILLED)
            
        # cv.imshow('Draw block', tile_rect)    
        # cv.waitKey(0)
        return tile_rect
    
    def get_rotated_piece(self, rotation_state):
         return rotation_state