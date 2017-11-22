'''
Created on 7 mai 2017

@author: EM
'''

import numpy as np
import math
import pandas as pd

import colorama
import copy

class Sudoku(object):
    '''
    Class for Sudoku grid resolution
    '''
    n = 3

    def __init__(self, grid):
        '''
        grid should be a 9 x 9 numpy array with integers from 0 to 9 
        '''
        grid=grid.astype(np.uint8)
        self.grid = copy.deepcopy(grid)
        self.initial_grid = copy.deepcopy(grid)
        N = Sudoku.n ** 2 
        
        # initialize boolean values
        self.is_in_row = np.zeros(shape = (N, N), dtype = bool) # 1st index corresponds to row in grid, 2nd index to value
        self.is_in_col = np.zeros(shape = (N, N), dtype = bool) # 1st index corresponds to col in grid, 2nd index to value
        self.is_in_square = np.zeros(shape = (Sudoku.n, Sudoku.n, N), dtype = bool)
        
        for i in range(0, 9):
            for j in range(0,9):
                k = self.grid[i,j]
                if (k != 0):
                    self.is_in_row[i, k-1] = True
                    self.is_in_col[j, k-1] = True
                    
                    self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), int(k-1)] = True
    
    
    def create_position_list(self):
        # This function create an ordered list of positions
        # positions are from 0 to N ** 2
        # positions are ordered by increasing number of posibilities
        
        N = Sudoku.n ** 2
        # create a vector with the number of possibilities for each position
        possibility_nb = np.zeros(shape = (N**2))
        for idx in range(0, N**2):
            i = int(math.floor(idx/N))
            j = idx % N
            if self.grid[i,j] != 0:
                # A number is already written in the grid
                possibility_nb[idx] = -1
            else:
                possible_nb = np.ones(shape = (N))
                for k in range(0, N):
                    if self.is_in_row[i, k] :
                        possible_nb[k] = 0
                    if self.is_in_col[j, k] :
                        possible_nb[k] = 0
                    if self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k]:
                        possible_nb[k] = 0
                possibility_nb[idx] = np.sum(possible_nb)
                
        # order the positions according to possibility_nb
        position_array = np.argsort(possibility_nb)     
        # remove indices corresponding to -1
        deleted_nb = np.sum(possibility_nb == -1)          
        position_array = position_array[deleted_nb:]        
        
        return position_array
        
    def is_valid(self, position_list):    
        # return true if the grid is valid
        # also update grid with the solution
        N = Sudoku.n**2
        
        if len(position_list) == 0 :
            # perhaps check if grid is indeed correct... to be added?
            return True
        
        i = int(math.floor(position_list[0]/N))
        j = position_list[0] % N
        # remove 1st element from position_list
        position_list = position_list[1:]
        
        for k in range(0, N):
            if (not self.is_in_row[i,k] 
                and not self.is_in_col[j,k] 
                and not self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k]):
                # update booleans with k
                self.is_in_row[i,k] = True
                self.is_in_col[j,k] = True
                self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k] = True
                if (self.is_valid(position_list)):
                    # update grid value
                    self.grid[i,j] = k+1
                    return True
                else:
                    self.is_in_row[i,k] = False
                    self.is_in_col[j,k] = False
                    self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k] = False
        return False
    
    def get_all_solutions_inner(self, position_list):
        # return (solution_nb, grid_list)
        N = Sudoku.n**2
        grid_list = []
        solution_nb = 0
        
        if len(position_list) == 0 :
            grid_list = [copy.deepcopy(self.grid)]
            return (1, grid_list)
        
        i = int(math.floor(position_list[0]/N))
        j = position_list[0] % N
        # remove 1st element from position_list
        position_list = position_list[1:]
        
        for k in range(0, N):
            if (not self.is_in_row[i,k] 
                and not self.is_in_col[j,k] 
                and not self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k]):
                # update booleans with k
                self.is_in_row[i,k] = True
                self.is_in_col[j,k] = True
                self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k] = True
                self.grid[i,j] = k+1
                temp = self.get_all_solutions_inner(position_list)
                solution_nb += temp[0]
                grid_list += temp[1]
                
                self.is_in_row[i,k] = False
                self.is_in_col[j,k] = False
                self.is_in_square[int(math.floor(i/3)), int(math.floor(j/3)), k] = False
        return (solution_nb, grid_list)
    
    def solve(self):
        # Solve the sudoku (returns true if at least a solution exists and update self.grid with that solution
        
        # put grid back to its initial value
        self.grid = copy.deepcopy(self.initial_grid)
        # compute ordered positions (solver will iterate on these positions)
        position_list = self.create_position_list()
        return self.is_valid(position_list)
    
    def get_all_solutions(self):
        # Returns (solution_nb, grid_list)
            
        # put grid to its initial value
        self.grid = copy.deepcopy(self.initial_grid)
        # compute ordered positions (solver will iterate on these positions)
        position_list = self.create_position_list()
        return self.get_all_solutions_inner(position_list)
    
    
    def print_grid(self, colored = False):
        N = Sudoku.n ** 2
        
        if colored:
            colorama.init()
        
        line_sep = " +-------+-------+-------+"
        for i in range(0, N):
            row_str = ""
            if i % Sudoku.n == 0:
                print(line_sep)
            for j in range(0, N):
                if j % Sudoku.n == 0:
                    row_str += " |"
                
                if (colored and self.initial_grid[i,j] == 0):
                    row_str += colorama.Fore.RED + " " + str(int(self.grid[i,j])) + colorama.Fore.WHITE
                else:
                    row_str += " " + str(int(self.grid[i,j]))
            row_str += " |"
            print(row_str)
        print(line_sep)
    
    
            
    def grid_to_csv(self, file):
        df = pd.DataFrame(self.grid)
        df.to_csv(file, sep = ";", header = False)

def solve_sudoku(grid):
	mySudoku = Sudoku(grid)
	mySudoku.solve()
	return(mySudoku.grid)