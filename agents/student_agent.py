# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys

# Additional imports
from copy import deepcopy
from math import log, sqrt, inf
from random import randint

class Node:    
    def __init__(self, parent, chess_board, my_pos, my_dir, adv_pos, agent_turn=True):
        self.parent = parent
        self.chess_board = chess_board
        self.wins = 0 # Number of wins in rollouts (includes children nodes)
        self.visits = 0 # Number of rollouts (includes children)
        self.children = []
        self.possible_children = []
        self.my_pos = my_pos
        self.my_dir = my_dir
        self.adv_pos = adv_pos
        self.agent_turn = agent_turn
        self.expanded = False
        
    def __eq__(self, other):
        return self.my_pos == other.my_pos and self.dir == other.dir and self.adv_pos == self.adv_pos
    
    def __contains__(self, item):
        return any(item, lambda x : self.__eq__(item))
    
    def heuristic(self):
        return 1
            
    def uct_value(self, current_state):
        # Exploration rate
        c = 0.5
        
        # If the node was not visited, we will prioritize the exploration of the  node 
        # by setting its value to infinity
        if self.visits == 0:
            return inf
        # Otherwise return its confidence bound value
        return self.wins / self.visits + c * sqrt(log(current_state.wins)/self.visits)
    
    def generate_possible_moves(self, max_step):
        move_directions = ((-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)) # Move Up, Right, Down, Left 
        queue = [(self, 0)] # List of possible children nodes
        
        if self.agent_turn:
            current_pos = lambda s: s.my_pos
            other_pos_fn = lambda s: s.adv_pos
            new_node = lambda new_chess_board, pos, dir: Node(new_chess_board, pos, dir, self.adv_pos, not self.agent_turn)
        else:
            current_pos = lambda s: s.adv_pos
            other_pos_fn = lambda s: s.my_pos
            new_node = lambda new_chess_board, pos, dir: Node(new_chess_board, self.my_pos, dir, pos, not self.agent_turn)
            
        while queue:
            child, step_size = queue.pop(0)
            # Check if the child is in the possible children
            if child in self.possible_children:
                continue
            
            # TODO call to heuristic         
            self.possible_children.append((child, child.heursitic())) 
            
            # Check max step size
            if step_size >= max_step:
                continue
            
            pos = current_pos(child)
            other = other_pos_fn(child)
            
            for (row, col, move_dir) in move_directions: # Positions
                
                if self.chess_board[pos[0]][pos[1]][move_dir]:
                    continue
                
                new_row = pos[0] + row
                new_col = pos[1] + col
                
                if (new_row, new_col) == other:
                    continue
                
                for (_, _, dir) in move_directions: # Walls 
                    if self.chess_board[new_row][new_col][dir]:
                        continue
                    
                    new_chess_board = deepcopy(self.chess_board)
                    new_chess_board[new_row][new_col][dir] = True
                    
                    new_child = new_node(new_chess_board, (new_row, new_col), dir)
                    self.possible_children.append(new_child)
                    queue.append((new_child, step_size + 1))
                    
        self.possible_children = sorted(self.possible_children, key = lambda _, val: val, reverse=True)
        
    def push_next_children(self, n):
        self.children += self.possible_children[:n]
        self.possible_children = self.possible_children[n:]
        
    def expand(self, n, max_step):
        if self.expanded:
            self.push_next_children(n)
        else:
            self.generate_possible_moves(max_step)
            self.push_next_children(n)
            self.expanded = True
        
class Rollout:
    def __init__(self, curr_state):
        
            pass
        
        
        
        
        
        
        
        
    

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
        #self.root_state = Node() # Initial state of the game
        #self.current_state = self.root_state # Progress of the game
        Node.heuristic = lambda state: 2

        
    def generate_moves(self, moves, moves_so_far, chess_board, my_pos, adv_pos, max_step, curr_step_size):
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
                        if (new_move_r, new_move_c) not in moves_so_far:
                            next_moves.append((new_move_r, new_move_c))
            
            # Recursive call 
            return moves + self.generate_moves(next_moves, moves+next_moves, chess_board, my_pos, adv_pos, max_step, curr_step_size+1)
        
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
                        child = Node(move=((x,y), dir), parent=self.current_state)
                        self.current_state.children[child.move] = child
                        
    def modify_chess_board(self, move, chess_board):
        # Extract the position and the direction from the move
        (row, col), dir = move
        # Put a wall
        chess_board[row, col, dir] = True
        # Return the modified chess board
        return chess_board
                        
    def rollout(self, child, adv_pos, chess_board, max_step):
        '''
        Simulates a single game starting with the child state and the adversary's current position.
        Returns 1 if the game was won, and 0 if the game was lost.
        '''
        # Modify the chess board using child state mode
        my_pos, my_dir = child.move
        chess_board = self.modify_chess_board((my_pos, my_dir), chess_board)
        
        # Simulate the moves alternatiely and modify the game state until the game is won by one of the players
        endgame, my_score, adv_score = self.endgame(my_pos, adv_pos, chess_board)
        print(f"is endgame1: {endgame}")
        if endgame:
            print("next move1")
            if my_score > adv_score: 
                    return 1 
            return 0
        
        while not endgame:
            
            # Let the adversary take a random step
            print(f"adv_pos1: {adv_pos}")
            print(f"my_pos1: {my_pos}, {my_dir}")
            adv_pos, adv_dir = self.random_step(chess_board, adv_pos, my_pos, max_step)
            chess_board = self.modify_chess_board((adv_pos, adv_dir), chess_board)
            endgame, my_score, adv_score = self.endgame(my_pos, adv_pos, chess_board)
            print(f"is endgame2: {endgame}")
            # Check if adversary move was a winning step
            if endgame:
                print("next move2")
                if my_score > adv_score: 
                    return 1 
                return 0
            print ("hu")
            # If we didn't win the game, let the agent take a random state 
            print(f"adv_pos2: {adv_pos}, {adv_dir}")
            print(f"my_pos2: {my_pos}, {my_dir}")
            my_pos, my_dir = self.random_step(chess_board, my_pos, adv_pos, max_step)
            chess_board = self.modify_chess_board((my_pos, my_dir), chess_board)
            endgame, my_score, adv_score = self.endgame(my_pos, adv_pos, chess_board)
            
            print(f"is endgame3: {endgame}")
            # Check if agent's move was a winning step
            if endgame:
                print("next move3")
                if my_score > adv_score: 
                    return 1 
                return 0

    # Helper functions `random_step` and `endgame` adapted from World class
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
        Returns true if the game has ended, and false otherwise. Is used by MCTS to verify the endgame during rollouts.
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
        return True, p0_score, p1_score

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
        print(chess_board)

        # test `generate_children``
        #self.generate_children(self.generate_moves([my_pos], chess_board, my_pos, adv_pos, max_step, 0), chess_board)
        #print(self.generate_moves([my_pos], [my_pos], chess_board, my_pos, adv_pos, max_step, 0))

        # for child in self.current_state.children:
        #     print(self.current_state.children[child].move)

        # test `endgame`
        # print(self.endgame(my_pos, adv_pos, chess_board))
        
        # test `rollout`
        #for child in self.current_state.children:
        #    print(self.rollout(self.current_state.children[child], adv_pos, chess_board, max_step))
        
        my_pos, dir = self.random_step(chess_board, my_pos, adv_pos, max_step)
        
        return my_pos, dir




