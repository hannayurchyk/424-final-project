# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys

# Additional imports
from copy import deepcopy
from math import log, sqrt, inf
from random import randint

class Node:
    ''' 
    MCTS node.
    Args:
        move ((int, int), str) : the position of this node on the chess_board that comes from a move made by the parent node
        N (int) : times this node was visited
        Q (int) : average winning rate from this position
        children (dict) : stores the next possible moves from this position
        win (bint) : if a node is a leaf, will be set to 1 if the game is won,
                     0 if the game is lost and -1 otherwise
    '''
    def __init__(self, move = None, parent = None):
        ''' 
        Initializes a new node with a move and an optional parent node.
        '''
        self.parent = parent
        self.move = move
        self.N = 0 # Number of rollouts or simulations
        self.Q = 0 # Number of wins per rollouts
        self.children = {}
        # self.outcome = -1
        
    def add_children(self, children):
        '''
        Adds children nodes to this node.
        '''
        for child in children:
            self.children[child.move] = child
            
    def value(self):
        '''
        Calculates the value of taking this move using upper confidence.
        The constant `c` represents how much an agent favors the node given 
        how much it was explored i.e. the exploration rate.
        We will arbitrarily set c to 0.5.
        '''
        # Exploration rate
        c = 0.5
        
        # If the node was not visited, we will prioritize the exploration of the  node 
        # by setting its value to infinity
        if self.N == 0:
            return inf
        # Otherwise return its confidence bound value
        return self.Q / self.N + c * sqrt(log(self.parent.N)/self.N)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {0: 'u', 1: 'r', 2: 'd', 3: 'l'}
        self.root = Node()
        
    def generate_moves(self, moves, chess_board, my_pos, adv_pos, max_step, curr_step_size):
        '''
        Returns all possible next moves for a given position and a chess board.
        Recursive algorithm that does a Breadth-First Search of the possible moves around the agent. 
        Args:
            moves (list) : list of tuples representing the possible positions for next moves; always initialized with the current position
            chess_board (list) : nested list such that [row, column, walls] where walls is a list of boolean each representing if there is a wall Up, Right, Down and Left
            my_pos (tuple) : tuple representing agent's position (row, col)
            adv_pos (tuple) : tuple representing adversary's position
            max_step (int) : max step size
            curr_step_size (int) : current step size            
        '''
        adv_r, adv_c = adv_pos # Adversary row and column
        my_r, my_c = my_pos # My row and column
        move_directions = ((-1, 0), (0, 1), (1, 0), (0, -1)) # Move Up, Right, Down, Left 
        next_moves = []
    
        if curr_step_size <= max_step:
            
            for move in moves:
                move_r, move_c = move
                
                for dir in range(0, len(move_directions)): # For all walls in the position
                    
                    # If there is no wall blocking us 
                    if not chess_board[move_r, move_c, dir]: 
                        # Generate the next move and verify if it is valid
                        new_move_r, new_move_c = move_r + move_directions[dir][0], move_c + move_directions[dir][1] 
                        
                        # Check if we are not going on the adversary 
                        if new_move_r == adv_r and new_move_c == adv_c:
                            continue
                        
                        # Check if we are coming back to our initial position
                        # If we just started exploring, then the move is valid if we want to place a wall around us
                        if len(moves) > 1 and new_move_r == my_r and new_move_c == my_c: 
                            continue
                        
                        # If the move is not a preceding move, add it to new moves
                        if (new_move_r, new_move_c) not in moves:
                            next_moves.append((new_move_r, new_move_c))
            
            # Recursive call 
            return moves + self.generate_moves(next_moves, chess_board, my_pos, adv_pos, max_step, curr_step_size+1)
        
        # Return an empty list when when we reach the maximal step size
        return next_moves 
            
    def generate_children(self, moves, chess_board):
        '''
        Converts the possible moves into the children nodes of the current position and adds them to the node.
        Args:
            moves (list) : list of tuples representing next valid moves from the current position
            chess_board (list) : nested list of [row, col, walls] where walls is a list of boolean values indicating if there is a wall U, R, D or L
        '''
        for (x, y) in moves:
            for dir in range(0, len(chess_board[x, y])):
                    # If there is no wall, add it as a next possible move 
                    if not chess_board[x, y, dir]:
                        child = Node(move=((x,y), self.dir_map[dir]), parent=self.root)
                        self.root.children[child.move] = child
                        
    def simulation(self, child, adv_pos, chess_board):
        pass
        

    # Helper functions copied and adapted from World class
    def random_step(self, chess_board, my_pos, adv_pos, max_step):
        '''
        This is a copy of the world `random_walk` that will be used to simulate a random move in MCTS algorithm.
        It is adapted such that we do not have to use numpy.
        '''
        ori_pos = deepcopy(my_pos) # Copy original position
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # List of possible move directions
        steps = randint(0, max_step + 1) # A random step size
        
        # Random Walk
        for _ in range(steps):
            r, c = my_pos # Position is row, col
            dir = randint(0, len(moves)-1) # Choose a random direction
            m_r, m_c = moves[dir] 
            my_pos = (r + m_r, c + m_c) # Take a single step in that direction

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos: # while there is a wall or we are at adversary position
                k += 1
                if k > 300:
                    break
                dir = randint(0, len(moves)-1) 
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = randint(0, len(moves)-1)
        r, c = my_pos
        while chess_board[r, c, dir]:

            dir = randint(0, len(moves)-1)

        return my_pos, dir

    def endgame(self, p0_pos, p1_pos, chess_board):
        '''
        Returns true if the game has ended, and false otherwise. Is used my MCTS to verify the endgame during rollouts.
        '''
        board_size = len(chess_board[0])
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        
        # Game isn't finished
        if p0_r == p1_r:
            return False, p0_score, p1_score
        
        # Game is finished 
        if p0_score > p1_score:
            return True, p0_score, p1_score
        
        
    
    def random_autoplay():
        '''
        Simulates several random runs of the game. Outputs the winning percentage of the branch. 
        Takes as input the number of random simulations.
        '''
        pass


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.


        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        
        My implementation:
        Computes the winning percentages for each next move using MCTS algorithm.
        
        """

        # test `generate_moves`
        #print(f'my position: {my_pos} with max step: {max_step}')
        #print(chess_board)

        # test `generate_children``
        self.generate_children(self.generate_moves([my_pos], chess_board, my_pos, adv_pos, max_step, 0), chess_board)
        # for child in self.root.children:
        #     print(self.root.children[child].move)

        # test `endgame`
        # print(self.endgame(my_pos, adv_pos, chess_board))
        
        return my_pos, dir




