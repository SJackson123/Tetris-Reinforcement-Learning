"""
Tetris environment.
Code available from: https://github.com/educ8s/Python-Tetris-Game-Pygame
"""


import numpy as np
import cv2 as cv
import random 
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import *
from .blocks import *


class Tetris():

    def __init__(self, num_rows=20, num_cols=10):
        self.game_grid = Grid(num_rows, num_cols)
        self.reset()

    def reset(self):
        """Return the empty grid as a parameterised vector."""
        self.game_grid.reset()
        self.game_over = False
        self.lines_cleared = 0
        self.tetriminoes = [IBlock(),
                                JBlock(),
                                LBlock(),
                                OBlock(),
                                SBlock(), 
                                TBlock(), 
                                ZBlock()]
        self.previous_piece = None
        self.current_piece = self.get_random_piece()
        self.next_piece = self.get_random_piece()
    
        return self.get_state_properties(self.current_piece, current_grid=self.game_grid, intermediate_grid=self.game_grid)
    

    def get_next_states(self, piece: Block, state: Grid):
        """Return a dictionary of valid action keys and the parameterised next state."""
        states = {}
        
        if piece.id == 3 or piece.id == 5 or piece.id == 7:
            n_rotations = 2
        elif piece.id == 4:
            n_rotations = 1
        else:
            n_rotations = 4
        
        for i in range(n_rotations):
            simulation_piece = copy.deepcopy(piece)
            simulation_piece.rotation_state = i

            coords = simulation_piece.get_cell_positions()
            # max_col = [(y, x), (y, x). ..] find max x
            max_col = max(coord[1] for coord in coords)
            # valid columns given the rotation
            valid_cols = self.game_grid.num_cols - max_col

            for column in range(valid_cols):
                copy_grid = copy.deepcopy(state)

                if column > 0:
                    self.move_right_one(simulation_piece)

                if self.piece_fits(simulation_piece,grid_object=copy_grid):
                    # drop piece
                    while self.piece_inside_grid(simulation_piece) and self.piece_fits(simulation_piece, grid_object=copy_grid):
                        self.move_down(simulation_piece)
                    # move back up
                    simulation_piece.move(-1, 0)
                    copy_grid.grid, rows_cleared, intermediate_grid = self.lock_block(simulation_piece, copy_grid)
                    states[(column, simulation_piece.rotation_state)] = self.get_state_properties(simulation_piece, copy_grid, intermediate_grid)
        return states


    def get_state_properties(self, piece: Block, current_grid: Grid, intermediate_grid: Grid) -> np.array:
        """Return a parameterised vector for a state."""
        
        if piece is None:
            piece = self.current_piece
        
        holes = self.get_holes(current_grid)
        bumpiness, height = self.get_bumpiness_and_height(current_grid.grid)
        landing_height = self.get_landing_height(piece)
        eroded_piece_cells = self.get_eroded_piece_cells(piece, intermediate_grid)
        row_transitions = self.get_row_transitions(current_grid)
        column_transitions = self.get_column_transitions(current_grid)
        cumulative_wells = self.get_well_sums(current_grid)

        return np.array([self.lines_cleared, holes, bumpiness, height, landing_height, eroded_piece_cells,
                         row_transitions, column_transitions, cumulative_wells])


    def get_bumpiness_and_height(self, grid):
        """Return the bumpiness and height properties of a state."""
        board = np.array(grid)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.game_grid.num_rows)
        heights = self.game_grid.num_rows - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)

        return total_bumpiness, total_height
    
    def get_holes(self, grid: Grid):
        """Return the number of holes in a state."""
        number_of_holes = 0
        for col in zip(*grid.grid):
            row = 0
            while row < grid.num_rows and col[row] == 0:
                row += 1
            number_of_holes += len([x for x in col[row + 1:] if x == 0])
        return number_of_holes

    def get_landing_height(self, piece: Block):
        """Return the landing height of a piece."""
        highest_column = max(piece.get_cell_positions(), key=lambda cell: cell[0])[0]

        return self.game_grid.num_cols - highest_column

    
    def get_eroded_piece_cells(self, piece: Block, intermediate_grid: Grid):
        """Return a score based on how well the current piece removes full rows."""
        score = 0
        cleared = np.all(intermediate_grid.grid, axis=1)
        cleared_num = sum(cleared)
        for i, is_full in enumerate(cleared):
            if is_full:
                # Count the number of cells in the piece that are in the full row
                piece_cells_in_row = sum(1 for row, _ in piece.get_cell_positions() if row == i)
                score += piece_cells_in_row

        return score * cleared_num

    
    def get_row_transitions(self, grid: Grid):
        """Return row transitions for adjacent pieces."""
        score = 0
        for row in range(self.game_grid.num_rows):
            for col in range(self.game_grid.num_cols - 1):  # Note the -1 here
                if grid.grid[row][col] != 0 and grid.grid[row][col + 1] != 0:
                    if grid.grid[row][col + 1] != grid.grid[row][col]:
                        score += 1
        return score
    

    def get_column_transitions(self, grid: Grid):
        score = 0
        for col in range(self.game_grid.num_cols):
            for row in range(self.game_grid.num_rows - 1):  # Note the -1 here
                if grid.grid[row][col] != 0 and grid.grid[row + 1][col] != 0:
                    if grid.grid[row + 1][col] != grid.grid[row][col]:
                        score += 1
        return score
    

    def get_well_sums(self, grid: Grid):
        """Return the number of wells. A well is a sequence of empty cells surrounded by occupied cells."""
        well_sum = 0
        
        for col in range(self.game_grid.num_cols):
            for row in range(1, self.game_grid.num_rows):
                if grid.grid[row][col] == 0:
                    left_occupied = (col == 0) or (grid.grid[row][col - 1] != 0)
                    right_occupied = (col == self.game_grid.num_cols - 1) or (grid.grid[row][col + 1] != 0)
                    if left_occupied and right_occupied:
                        well_sum += 1
        
        return well_sum



    def get_random_piece(self) -> Block(id):
        """Return a random block object."""
        if (len(self.tetriminoes) == 0):
            self.tetriminoes = [IBlock(),
                                JBlock(),
                                LBlock(),
                                OBlock(),
                                SBlock(), 
                                TBlock(), 
                                ZBlock()]
        piece = random.choice(self.tetriminoes)
        self.tetriminoes.remove(piece)

        return piece
    
    def piece_inside_grid(self, piece):
        """Return a boolean to test whether a piece is inside the grid."""
        tiles = piece.get_cell_positions()
        for tile in tiles:
            if self.game_grid.is_inside(tile[0], tile[1]) == False:
                return False
            
        return True
    
    def piece_fits(self, piece: Block, grid_object: Grid):
        """
        Return true if a piece can be placed in the current grid sqaures and not overlap
        with an existing piece.
        """
        tiles = piece.get_cell_positions()
        for row, column in tiles:
            # if there is a block already there, return false
            if grid_object.is_empty(row, column) == False:
                return False
        return True
    
    def lock_block(self, piece: Block, grid: Grid) -> list:
        """
        Return grid after a piece has been locked in place and full rows have been
        removed.
        """
        copy_grid = copy.deepcopy(grid)
        tiles = piece.get_cell_positions()
        for position in tiles:
            copy_grid.grid[position[0]][position[1]] = piece.id
        
        # remove full rows and update grid
        rows_cleared, new_grid = grid.clear_full_rows(copy_grid.grid)

        return new_grid, rows_cleared, copy_grid

    def new_round(self):
        """Change the current piece and get the next piece."""
        self.previous_piece = self.current_piece
        self.current_piece = self.next_piece
        self.next_piece = self.get_random_piece()
    
    def update_rows_cleared(self, rows_cleared):
        """Return the total number of rows cleared."""
        self.lines_cleared += rows_cleared

    def move_down(self, piece):
        piece.move(1, 0)

    def move_right(self, piece: Block,  x: int):
        piece.move(0, x)

    def move_right_one(self, piece):
        piece.move(0, 1)

    def print_grid_arg(self, copy_grid):
        """Print out the grid for debugging purposes."""
        for row in copy_grid:
            for i in row:
                print(i, end=" ")
            print()

    def step(self, action:tuple, render=False, render_delay=None):
        """Return the updated grid, rewards and done condition when interacting with the environment."""
        column, rotation = action
        copy_piece = copy.deepcopy(self.current_piece)
        copy_grid = copy.deepcopy(self.game_grid)
        
        # rotate and move to column
        copy_piece.rotation_state = rotation
        self.move_right(copy_piece, column)

        while self.piece_inside_grid(piece=copy_piece) and self.piece_fits(piece=copy_piece, grid_object=self.game_grid):
            if render:
                self.render(piece=copy_piece)
            self.move_down(piece=copy_piece)
        copy_piece.move(-1, 0)
        
        # intermediate_grid - grid with piece placed but no rows removed
        self.game_grid.grid, rows_cleared, intermediate_grid = self.lock_block(copy_piece, copy_grid)
        self.update_rows_cleared(rows_cleared)

        self.new_round()

        next_states = self.get_next_states(piece = self.current_piece, state=self.game_grid)
        valid_actions = set(next_states.keys())
        
        # give reward of 1 for placing a piece
        # reward = 1
        
        # game over condition
        if len(valid_actions) == 0:
            # reward -= 5
            self.game_over = True

        new_state = self.get_state_properties(piece=self.previous_piece, current_grid=self.game_grid, intermediate_grid=intermediate_grid)

        # if rows_cleared == 1:
        #     reward += 40
        # elif rows_cleared == 2:
        #     reward += 100
        # elif rows_cleared == 3:
        #     reward += 300
        # elif rows_cleared == 4:
        #     reward += 1200

        reward = self.lines_cleared
    
        return new_state, reward, self.game_over


    def render(self, piece: Block, save_path=None):
        """Display the grid with tetriminoes."""
        img = np.zeros((self.game_grid.num_rows * self.game_grid.cell_size+1,
                          self.game_grid.num_cols * self.game_grid.cell_size+1, 3), dtype='uint8')

        r = self.game_grid.draw_grid(img)
        # draw the block on the game grid
        r = self.current_piece.draw_block(img)
        # if save_path:
        #     cv.imwrite(save_path, r)
        r = piece.draw_block(img)

        cv.imshow("Tetris", r)
        cv.waitKey(1)

    def evaluate(self, W: list, state: list) -> int:
        """Return the best action for a state using a weight vector."""
        s0,s1,s2,s3,s4,s5,s6,s7=0,0,0,0,0,0,0,0
        
        s0 += W[0] * state[0]
        s1 += W[1] * state[1]
        s2 += W[2] * state[2]
        s3 += W[3] * state[3]
        s4 += W[4] * state[4]
        s5 += W[5] * state[5]
        s6 += W[6] * state[6]
        s7 += W[7] * state[7]

        return s0+s1+s2+s3+s4+s5+s6+s7

    def get_best_move(self, W, next_states : dict):
        actions, states = zip(*next_states.items())
        
        scores = []
        for s in states:
            scores.append(self.evaluate(W, s))
        # reutnr index of highest score
        index = np.argmax(scores)

        return actions[index]


    def simulation(self, W, render=False):
        """One episode of tetris. Select actions according to weight vector."""
        rewards_sum = 0
        state = self.reset()
        done = False
        
        max_steps = 500
        score_tetris = 0
        # while True:
        for i in range(max_steps):
            next_states = self.get_next_states(piece=self.current_piece, state=self.game_grid)
            
            action = self.get_best_move(W, next_states)
            if render:
                next_state, reward, done = self.step(action, render)
            else:
                next_state, reward, done = self.step(action)
            rewards_sum += reward
            if reward == 4:
                score_tetris += 1 
            state = next_state

            if done or i > 500:
                break

        return rewards_sum, score_tetris