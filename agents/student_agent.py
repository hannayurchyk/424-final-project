# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys

# Additional imports
from copy import deepcopy
import math 
from random import randint
import time

TIME_DELTA = 0.05
COMPUTATION_TIME = 2

def heuristic(state):
    return 2

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
        return self.my_pos == other.my_pos and self.adv_pos == self.adv_pos and self.chess_board == other.chess_board
    
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
        temp = self.possible_children[:n]
        self.children += temp
        self.possible_children = self.possible_children[n:]
        return temp
        
    def expand(self, n, max_step):
        if self.expanded:
            return self.push_next_children(n)
        else:
            self.generate_possible_moves(max_step)
            self.expanded = True
            return self.push_next_children(n)

     
    def backpropagate(self, wins, visits):
        node = self
        
        while node:
            node.wins += wins
            node.visits += visits
            node = node.parent
           
    def rollout(self, num_rollouts):
        rollout = HeuristicRollout(self)
        rollout.run(num_rollouts)
  
        
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
            
    def run(self, num_rollouts):
        self.rec_rollout(self.curr_state, num_rollouts)
        

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {0: 'u', 1: 'r', 2: 'd', 3: 'l'}
        self.root_state = None # Initial state of the game
        #self.current_state = self.root_state # Progress of the game
        Node.heuristic = heuristic

    def update_tree(self, chess_board, my_pos, adv_pos):
        if not self.current_state:
            self.current_state = Node(None, chess_board, my_pos, None, adv_pos, True)
            return

        next_state = Node(None, chess_board, my_pos, None, adv_pos, True)

        def depth_2_search(node, depth):
            if depth == 2:
                if node == next_state:
                    return node
                return None

            if node.expanded:
                for child in node.children:
                    if t := depth_2_search(child, depth + 1):
                        return t

                for child in node.possible_children:
                    if t := depth_2_search(child, depth + 1):
                        return t
            
            return None

        if node := depth_2_search(self.current_state, 0):
            node.parent = None
            self.root_state = node
        else:
            self.root_state = next_state


    def step(self, chess_board, my_pos, adv_pos, max_step):
        initial_time = time.time()

        self.update_tree(chess_board, my_pos, adv_pos)

        n = self.root_state.visits

        def find_best_uct(node):
            uct_max, uct_max_child = max(map(find_best_uct, node.children + node.possible_children), key=lambda pair: pair[0]) \
                if not node.children or not node.possible_children else (-math.inf, None)
            
            return (uct_max, uct_max_child) if uct_max > (uct := node.uct_value(n)) else (uct, node)

        while time.time() - initial_time < COMPUTATION_TIME - TIME_DELTA:
            (uct, best_child) = find_best_uct(self.root_state)
            expanded = best_child.expand(1, max_step)
            for child in expanded:
                child.rollout(100)
        
        best_child = None
        best_child_uct = -math.inf

        for direct_children in self.root_state.children:
            min_child_uct, min_child = min(map(lambda child: (child.uct_value(n), child), direct_children.children + direct_children.possible_children), key=lambda pair: pair[0])

            if not min_child:
                continue

            if min_child_uct > best_child_uct:
                best_child = min_child
                best_child_uct = min_child_uct

        return best_child.parent.my_pos, best_child.parent.dir




