# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys

# Additional imports
from copy import deepcopy
import math 
from random import randint

def endgame(p0_pos, p1_pos, chess_board):
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
        return self.my_pos == other.my_pos and self.dir == other.dir and self.adv_pos == self.adv_pos and self.chess_board == other.chess_board
    
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
            return math.inf
        # Otherwise return its confidence bound value
        return self.wins / self.visits + c * math.sqrt(math.log(current_state.wins)/self.visits)
    
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
                    
        self.possible_children = sorted(self.possible_children, key = lambda _, val: val, reverse=self.agent_turn)
        
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
     
    def backpropagate(self, wins, visits):
        node = self
        
        while node:
            node.wins += wins
            node.visits += visits
            node = node.parent
           
    def rollout(self, num_rollouts):
        # rollout = HeuristicRollout...
        # wins = rollout.rollout(num_rollouts, self.heuristic)
        # backpropagate
        pass        
        
class HeuristicRollout:
    def __init__(self, curr_state):
        self.curr_state = curr_state # Node
    
    # TODO define a decay schedule
    @staticmethod
    def rollout_decay(num_rollouts):
        return num_rollouts*0.5
    
    # TODO misuse of num_rollouts + acc
    def rec_rollout(self, state_to_explore, num_rollouts):
        state_to_explore.expand(0)
        
        acc = 0
        
        for next_state in state_to_explore.possible_children[:num_rollouts]:
            
            (is_endgame, my_score, adv_score) = endgame(next_state.my_pos, next_state.adv_pos, next_state.chess_board) # TODO approximate connected component 
            
            if is_endgame:
                next_state.backpropagate(1 if my_score > adv_score else 0 , 1)
                acc+=1
                continue
            
            acc += self.rec_rollout(next_state, HeuristicRollout.rollout_decay(num_rollouts))
        
        return acc
            
    def rollout(self, num_rollouts):
        self.rec_rollout(self.curr_state, num_rollouts)
        
        

        
        
        
        
        
        
    

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {0: 'u', 1: 'r', 2: 'd', 3: 'l'}
        self.root_state = Node() # Initial state of the game
        self.current_state = self.root_state # Progress of the game
        Node.heuristic = lambda state: 2


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

    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        
        
        
        my_pos, dir = self.random_step(chess_board, my_pos, adv_pos, max_step)
        
        return my_pos, dir




