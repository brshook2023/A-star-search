'''
Homework 1
Machine Reasoning 
Professor: Dr. Raghu Ramanujan
Fall 2020
Davidson College

This program uses the A* search algorithm to solve
the 8-puzzle. Three heuristic funtions are available 
to be used in the A* search algorithm.

@authors: Brad Shook
          Wilbert Garcia
'''


from random import shuffle, randint
from queue import PriorityQueue
from sys import getsizeof
from collections import deque
import csv
import time

'''
Puzzle class creates a sliding puzzle to be solved.
'''
class Puzzle:

    # Initialize a Puzzle object with the heuristic to be used.
    def __init__(self, heuristic):
        # initialize the goal state
        self.goal_state = [1,2,3,
                          4,5,6,
                          7,8,0]

        # the heuristic function to be used in A*
        self.heuristic = heuristic

        # initialize the initial state with a random solvable board
        self.initial_state = self.generate_solvable_board()

    '''
    Sets the initial state of the board to a given board
    
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            None
    '''
    def set_initial_state(self, board):
        self.initial_state = board


    '''
    Creates a sliding puzzle which is solvable.
    
        Parameters: 
            self
        
        Returns: 
            board (list): the sliding puzzle
    '''
    def generate_solvable_board(self):
        solvable_bool = False
        # generate random boards until a solvable one is made
        while solvable_bool != True:
            # generate random board
            board = self.generate_random_board()

            # check if board is solvable
            solvable_bool = self.is_solvable(board)

        return board

    '''
    Creates a random sliding puzzle
    
        Parameters: 
            self
        
        Returns: 
            board (list): the sliding puzzle
    '''
    def generate_random_board(self):
        board = [i for i in self.goal_state]

        # shuffle values
        shuffle(board)

        return board

    '''
    Checks if a given slidng puzzle is solvable. 
                      
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            boolean: True if the inversion count is even, false otherwise
    '''
    def is_solvable(self, board): 

        inversion_count = 0
        # iterate through each index on the board except the last one
        for i in range(len(board) - 1):
            for j in range(i + 1, len(board)):
                # count pairs(i, j) such that i appears 
                # before j, but i > j.  [4, 3, 2, 1] has 6 inversions. The 4 has 3 inv. The 3 has two. The 2 has 1.
                if (board[j] and board[i] and board[i] > board[j]):
                    inversion_count += 1

        # return true if inv count is even, false if odd
        return (inversion_count % 2 == 0)

    '''
    Prints a properly formatted version of a given sliding puzzle.
    
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            None
    '''
    def print_board(self, board):
        print(board[0:3])
        print(board[3:6])
        print(board[6:9])

    '''
    Checks if a given state is the goal state.
                   
        Parameters: 
            self
            board (list): the sliding puzzle
    
        Returns: 
            boolean: True if the given state is the goal state, False otherwise.
    '''
    def goal_test(self, state):

        for i in range(len(state)):
            if state[i] != self.goal_state[i]:
                return False

        return True

'''
Node class that contains functions and fields relevant
to individual nodes in the A* search algorithm.
'''
class Node:
    '''
    Initializes a node object.
               
        Parameters: 
            self
            heuristic (str): the heuristic function to be used
            current_state (list): the current state of the sliding puzzle
            path_cost (int): the total path cost to the current node
            f (int): the f cost of the current node
            path (list): the solution path to the current node
        
        Returns: 
            None
    '''
    def __init__(self, heuristic, current_state, path_cost, f=0, path=[]):
        self.current_state = current_state
        self.vacant_sq_index = self.current_state.index(0)
        self.heuristic_function = heuristic
        self.children = []
        self.goal_state = [1,2,3,
                          4,5,6,
                          7,8,0]

        # array of nodes leading to current node
        self.path = path

        # edge cost to get from start node to current node
        self.path_cost = path_cost
        self.heuristic_cost = 0
        self.f = f

        # update f based on heuristic given
        if self.heuristic_function == "h1":
            self.count_misplaced_tiles(self.current_state)
            self.increment_path_cost()
            self.calculate_f()
        elif self.heuristic_function == "h2":
            self.calculate_manhattan_distance_sum(self.current_state)
            self.increment_path_cost()
            self.calculate_f()
        elif self.heuristic_function == "h3":
            self.find_linear_conflict(self.current_state)
            self.increment_path_cost()
            self.calculate_f()
        elif self.heuristic_function == "none":
            self.increment_path_cost()

    '''
    Defines how the less than operator works.
    It had to be redefined to fix the error 
    of a node having two children with the same
    f cost.   
    '''
    def __lt__(self, other):
        return self.f < other.f

    '''
    Calculates the f cost of the node.
                   
        Parameters: 
            self
        
        Returns: 
            None
    '''
    def calculate_f(self):
        self.f = self.path_cost + self.heuristic_cost

    '''
    Adds 1 to the path cost of the node. Used to keep track 
    of the path cost of each node.
                   
        Parameters: 
            self
        
        Returns: 
            None
    '''
    def increment_path_cost(self):
        self.path_cost += 1

    '''
    Counts how many tiles are misplaced in relation to the goal 
    state and adds the count to the node's heuristic_cost field. 
    This is the first heuristic function, 'h1'.
                   
        Parameters: 
            self
            board (list): the current state of the puzzle
        
        Returns: 
            None
    '''
    def count_misplaced_tiles(self, board):
        count = 0
        # iterate through each index on the board
        for i in range(len(self.goal_state)):
            # check if the current tile matches its position in the goal state
            if board[i] != self.goal_state[i]:
                count += 1

        self.heuristic_cost += count

    '''
    Calculates the manhattan distance from each tile to that tile's
    goal position and sums these distances. The sum is added to the
    node's heuristic cost field. This is the second heuristic function, 'h2'.
                   
        Parameters: 
            self
            board (list): the current state of the puzzle.
        
        Returns: 
            None
    '''
    def calculate_manhattan_distance_sum(self, board):
        total = 0

        for i in range(len(board)):

            start_index = i 
            final_index = board[i] - 1 
            if final_index == -1:
                final_index = 8

            start_index_row_col = self.find_row_and_col(start_index)
            final_index_row_col = self.find_row_and_col(final_index)

            row_diff = abs(start_index_row_col[0] - final_index_row_col[0])
            col_diff = abs(start_index_row_col[1] - final_index_row_col[1])

            total += row_diff + col_diff

        self.heuristic_cost += total
        return total

    '''
    Finds what row and column a given index belongs to in the puzzle.
                   
        Parameters: 
            self
            index (int): the position of a tile
        
        Returns: 
            list: contains the row and column that the index belongs to
    '''
    def find_row_and_col(self, index):
        row = 0
        col = 0
        if index <= 2:
            row = 1
        elif index <= 5:
            row = 2
        else:
            row = 3

        if index in [0, 3, 6]:
            col = 1
        elif index in [1, 4, 7]:
            col = 2
        else:
            col = 3

        return [row, col]

    '''
    Updates the heuristic cost according to the linear 
    conflict heuristic.
                   
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            None
    '''
    def find_linear_conflict(self, board):
        lc = 0

        row_conflict_count_dict, row_conflict_tile_dict = self.find_conflicts_in_rows(board)
        col_conflict_count_dict, col_conflict_tile_dict = self.find_conflicts_in_cols(board)

        max_conflict_in_rows = max(row_conflict_count_dict.values())
        while max_conflict_in_rows > 0:

            # get tile with most conflicts
            max_row_tile = max(row_conflict_count_dict, key=row_conflict_count_dict.get)

            # set max tile number of conflicts to 0
            row_conflict_count_dict[max_row_tile] = 0
            # get the tiles that are conflicting with the max tile
            tile_k_conflicting_tiles = row_conflict_tile_dict[max_row_tile]

            # iterate through the conflicting tiles, subtracting 1 from
            # their conflict counts
            for tile in tile_k_conflicting_tiles: 
                row_conflict_count_dict[tile] -= 1
                row_conflict_tile_dict[tile].remove(max_row_tile)
                row_conflict_tile_dict[max_row_tile] = []

            # add 1 to the total linear conflict
            lc += 1
            # find the new max conflict
            max_conflict_in_rows = max(row_conflict_count_dict.values())

        max_conflict_in_cols = max(col_conflict_count_dict.values())
        while max_conflict_in_cols > 0:
            # get tile with most conflicts
            max_col_tile = max(col_conflict_count_dict, key=col_conflict_count_dict.get)

            # set max tile number of conflicts to 0
            col_conflict_count_dict[max_col_tile] = 0
            # get the tiles that are conflicting with the max tile
            tile_k_conflicting_tiles = col_conflict_tile_dict[max_col_tile]

            for tile in tile_k_conflicting_tiles: 
                col_conflict_count_dict[tile] -= 1
                col_conflict_tile_dict[tile].remove(max_col_tile)
                col_conflict_tile_dict[max_col_tile] = []

            # add 1 to the total linear conflict
            lc += 1
            # find the new max conflict
            max_conflict_in_cols = max(col_conflict_count_dict.values())

        final_lc = 2 * lc

        self.heuristic_cost += final_lc + self.calculate_manhattan_distance_sum(board)

    '''
    Finds the row conflicts of a board.
    
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            tuple of two dictionaries
    '''
    def find_conflicts_in_rows(self, board):

        board_with_rows = []
        board_with_rows.append(board[0:3])
        board_with_rows.append(board[3:6])
        board_with_rows.append(board[6:9])
        offset = 0

        row_conflict_count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 0: 0}
        row_conflict_tile_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 0: []}
        # iterate through each row on board
        for row_i in board_with_rows:
            for tile_j in range(len(row_i)):
                for tile_k in range(tile_j + 1, len(row_i)):
                    tile_j_value = row_i[tile_j]
                    tile_k_value = row_i[tile_k]
                    goal_conflict = False

                    tile_j_correct_index = self.find_correct_tile_index(tile_j_value)
                    tile_k_correct_index = self.find_correct_tile_index(tile_k_value)

                    current_row_indexes = [0 + offset, 1 + offset, 2 + offset]

                    if tile_j_correct_index in current_row_indexes and tile_k_correct_index in current_row_indexes:
                        # tile j and tile k aren't in correct spots
                        if (((tile_j_correct_index > board.index(tile_j_value)) and (tile_k_correct_index < board.index(tile_k_value))) 
                            # tile J is in correct spot but tile K needs to move to the left of tile J
                            or ((tile_j_correct_index == board.index(tile_j_value)) and (tile_k_correct_index < tile_j_correct_index)) 
                                # tile K is in correct spot but tile J needs to move to the right of tile K 
                                or ((tile_k_correct_index == board.index(tile_k_value)) and (tile_j_correct_index > tile_k_correct_index))):

                            goal_conflict = True

                        if goal_conflict:
                            row_conflict_count_dict[tile_j_value] += 1
                            row_conflict_count_dict[tile_k_value] += 1
                            row_conflict_tile_dict[tile_j_value].append(tile_k_value)
                            row_conflict_tile_dict[tile_k_value].append(tile_j_value)

            offset += 3

        return (row_conflict_count_dict, row_conflict_tile_dict)
    
    '''
    Finds the column conflicts of a board.
    
        Parameters: 
            self
            board (list): the sliding puzzle
        
        Returns: 
            tuple of two dictionaries
    '''
    def find_conflicts_in_cols(self, board):

        board_with_cols = []
        first_col = []
        second_col = []
        third_col = []
        offset = 0

        col_conflict_count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 0: 0}
        col_conflict_tile_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 0: []}

        for i in range(len(board)):
            if i in [0,3,6]:
                first_col.append(board[i])
            elif i in [1,4,7]:
                second_col.append(board[i])
            else:
                third_col.append(board[i])

        board_with_cols = [first_col, second_col, third_col]

        for col_i in board_with_cols:
            for tile_j in range(len(col_i)):
                for tile_k in range(tile_j + 1, len(col_i)):

                    tile_j_value = col_i[tile_j]
                    tile_k_value = col_i[tile_k]
                    goal_conflict = False

                    tile_j_correct_index = self.find_correct_tile_index(tile_j_value)
                    tile_k_correct_index = self.find_correct_tile_index(tile_k_value)

                    current_col_indexes = [0 + offset, 3 + offset, 6 + offset]

                    if tile_j_correct_index in current_col_indexes and tile_k_correct_index in current_col_indexes:

                        # tile j and tile k aren't in correct spots
                        if (((tile_j_correct_index > board.index(tile_j_value)) and (tile_k_correct_index < board.index(tile_k_value))) 
                            # tile J is in correct spot but tile K needs to move to the left of tile J
                            or ((tile_j_correct_index == board.index(tile_j_value)) and (tile_k_correct_index < tile_j_correct_index)) 
                                # tile K is in correct spot but tile J needs to move to the right of tile K 
                                or ((tile_k_correct_index == board.index(tile_k_value)) and (tile_j_correct_index > tile_k_correct_index))):

                            goal_conflict = True

                        if goal_conflict:
                            col_conflict_count_dict[tile_j_value] += 1
                            col_conflict_count_dict[tile_k_value] += 1
                            col_conflict_tile_dict[tile_j_value].append(tile_k_value)
                            col_conflict_tile_dict[tile_k_value].append(tile_j_value)

            offset += 1

        return (col_conflict_count_dict, col_conflict_tile_dict)


    def find_correct_tile_index(self, tileValue):
        return self.goal_state.index(tileValue)

    '''
    Finds the possible successors of the the current state and 
    appends them to the children field array.
                   
        Parameters: 
            self
        
        Returns: 
            None
    '''
    def find_next_states(self):
        # move left
        if self.vacant_sq_index not in [0, 3, 6]:

            self.children.append(Node(self.heuristic_function, self.move_vacant_left(), 
                                        self.path_cost, self.f, self.path + [self.current_state]))

        # move right
        if self.vacant_sq_index not in [2, 5, 8]:

            self.children.append(Node(self.heuristic_function, self.move_vacant_right(), 
                                        self.path_cost, self.f, self.path + [self.current_state]))

        # move down
        if self.vacant_sq_index < 6:

            self.children.append(Node(self.heuristic_function, self.move_vacant_down(), 
                                        self.path_cost, self.f, self.path + [self.current_state]))

        # move up
        if self.vacant_sq_index > 2:

            self.children.append(Node(self.heuristic_function, self.move_vacant_up(), 
                                        self.path_cost, self.f, self.path + [self.current_state]))

    '''
    Creates a new state list with the vacant tile shifted left.   
     
        Parameters: 
            self
        
        Returns: 
            state (list): the updated state of the board.
    '''
    def move_vacant_left(self):

        state = [i for i in self.current_state]
        left_sq_val = state[self.vacant_sq_index - 1]
        state[self.vacant_sq_index] = left_sq_val
        state[self.vacant_sq_index - 1] = 0

        return state
    '''
    Creates a new state list with the vacant tile shifted right.   
     
        Parameters: 
            self
        
        Returns:
            state (list): the updated state of the board.
    '''
    def move_vacant_right(self):
        state = [i for i in self.current_state]
        right_sq_val = state[self.vacant_sq_index + 1]
        state[self.vacant_sq_index] = right_sq_val
        state[self.vacant_sq_index + 1] = 0

        return state

    '''
    Creates a new state list with the vacant tile shifted down.   
     
        Parameters: 
            self
    
        Returns: 
            state (list): the updated state of the board.
    '''
    def move_vacant_down(self):
        state = [i for i in self.current_state]
        below_sq_val = state[self.vacant_sq_index + 3]
        state[self.vacant_sq_index] = below_sq_val
        state[self.vacant_sq_index + 3] = 0

        return state

    '''
    Creates a new state list with the vacant tile shifted up.     
     
        Parameters: 
            self
    
        Returns:
            state (list): the updated state of the board.
    '''
    def move_vacant_up(self):
        state = [i for i in self.current_state]
        above_sq_val = state[self.vacant_sq_index - 3]
        state[self.vacant_sq_index] = above_sq_val
        state[self.vacant_sq_index - 3] = 0

        return state

    '''
    Sets a board that takes a specific number of moves to solve.
     
        Parameters: 
            self
            num_moves (int): the number of moves to be made
    
        Returns:
            None
    '''
    def set_board_with_num_of_moves(self, num_moves):
        moves_made = 0
        last_move = 0
        while moves_made < num_moves:

            move = randint(1,4)
            # move left
            if self.vacant_sq_index not in [0, 3, 6] and move == 1 and last_move != 2:

                self.current_state = self.move_vacant_left()
                self.vacant_sq_index = self.current_state.index(0)

                last_move = 1
                moves_made += 1

            # move right
            if self.vacant_sq_index not in [2, 5, 8] and move == 2 and last_move != 1:

                self.current_state = self.move_vacant_right()
                self.vacant_sq_index = self.current_state.index(0)

                last_move = 2
                moves_made += 1

            # move down
            if self.vacant_sq_index < 6 and move == 3 and last_move != 4:

                self.current_state = self.move_vacant_down()
                self.vacant_sq_index = self.current_state.index(0)

                last_move = 3
                moves_made += 1

            # move up
            if self.vacant_sq_index > 2 and move == 4 and last_move != 3:

                self.current_state = self.move_vacant_up()
                self.vacant_sq_index = self.current_state.index(0)

                last_move = 4
                moves_made += 1

'''
Executes the A* search algorithm.  
    
    Parameters: 
        self
        depth_num (int): the number of moves used to create the board
        iteration (int): a counter keeping track of the number of boards solved

    Returns:
        state (list): the updated state of the board.
'''
def a_star_search(puzzle, depth_num, iteration):
        # create initial node
        start_time = time.time()
        node = Node(puzzle.heuristic, puzzle.initial_state, 0)

        # frontier is priority queue of nodes with f values as priority
        frontier = PriorityQueue()

        # put initial node in frontier
        frontier.put((0, node))

        explored = set()
        counter = 0
        bf_sum = 0
        node_size = getsizeof(node.current_state)
        mem_size = 0

        while 1:
            # print("\nNodes Generated: ")
            # print(len(explored))

            # if len(explored) > 100000:
            #     print("Memory Error: Too many nodes have been generated.")
            #     exit()

            # print("\nExplored Size: ")
            # print(mem_size)
            if frontier.empty():
                return "Failure: Solution not found."

            node = frontier.get()
            # print(puzzle.initial_state)
            # print("\nPath Cost/Depth: ")
            # print(node[1].path_cost)
            # puzzle.print_board(node[1].state)
            mem_size += node_size

            # check if puzzle was solved
            if puzzle.goal_test(node[1].current_state) == True:
                # write to csv - start state, path length, EBF, nodes generated
                # print("Initial State: ")
                # puzzle.print_board(puzzle.initial_state)
                # print()
                # print("Path Length: ")
                # print(len(node[1].path))
                # print("\nSolution Path: ")
                # for i in node[1].path:
                #     puzzle.print_board(i)
                #     print()
                # puzzle.print_board(puzzle.goal_state)
                # print("\nPath Length: ")
                # print(len(node[1].path))
                # ebf = bf_sum / len(explored)

                # print("\nEBF: " + str(ebf))

                end_time = time.time()

                total_time = end_time - start_time

                with open("CorrectEdgeDepth" + str(depth_num) + "Analysis.csv", 'a+', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([iteration, node[1].heuristic_function, len(node[1].path), len(explored), total_time])

                return node[1].path

            # add the hashed value for the state
            explored.add(hash(tuple(node[1].current_state)))

            # find children of current node
            node[1].find_next_states()

            # iterate through current node's children
            for child in node[1].children:

                if child not in frontier.queue and hash(tuple(child.current_state)) not in explored:
                    frontier.put((child.f, child))
                    bf_sum += 1

h1_puzzle = Puzzle("h1")
h2_puzzle = Puzzle("h2")
h3_puzzle = Puzzle("h3")

depth_num = 10

# Perform experiments
for i in range(800):

    h1_puzzle.set_initial_state([1,2,3,4,5,6,7,8,0])
    node = Node(h1_puzzle.heuristic, h1_puzzle.initial_state, 0)
    node.set_board_with_num_of_moves(depth_num)

    h1_puzzle.initial_state = node.current_state
    h2_puzzle.initial_state = node.current_state
    h3_puzzle.initial_state = node.current_state

    a_star_search(h1_puzzle, depth_num, i)
    a_star_search(h2_puzzle, depth_num, i)
    a_star_search(h3_puzzle, depth_num, i) 
	