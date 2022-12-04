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
COMPUTATION_TIME = 1.999
FIRST_COMPUTATION_TIME = 29.990
NUM_ROLLOUTS = 20
ROLLOUT_DECAY = 0.3
NODES_TO_EXPAND = 3
UCT_EXPLORATION_RATE = 0.5

# Define the heuristic according to which we want to rank the states of the game
def heuristic(state):
    # state: Node
    def get_walls(state, adv=True):
        walls = []
        if adv:
            walls = state.chess_board[state.adv_pos[0]][state.adv_pos[1]]
        else: 
            walls = state.chess_board[state.my_pos[0]][state.my_pos[1]]
        return list(walls).count(True)
    
    # Compute the ranking features 
    adv_walls = get_walls(state)
    my_walls = get_walls(state, adv=False)
    dist_to_adv = abs(state.my_pos[0]-state.adv_pos[0]) + abs(state.my_pos[1]-state.adv_pos[1]) # Compute the Manhattan distance to the opponent 
    
    # Return the weights of the heuristic features 
    return adv_walls - (0 if my_walls < 3 else -math.inf) + dist_to_adv


# Helper method to compare chess_board used to override equals method in Node class
def flatten(l):
    return [item for sublist in l for subsublist in sublist for item in subsublist]

# Code used by the world class to check if an endgame is reached 
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
        self.children = [] # List of tuples containing (Node, Node.heuristic())
        self.possible_children = [] # List of tuples containing (Node, Node.heuristic())
        self.my_pos = my_pos # Tuple (row, col)
        self.my_dir = my_dir # Int from 0 to 3 for directions up, right, down, left 
        self.adv_pos = adv_pos
        self.agent_turn = agent_turn
        self.expanded = False
        self.lost = False
        self.win = False
        
    def __eq__(self, other):
        if not (self.my_pos == other.my_pos and self.adv_pos == other.adv_pos):
            return False

        return all(flatten(self.chess_board == other.chess_board))
    
    def __contains__(self, item):
        return any(item, lambda x : self.__eq__(item))
    
    # Heuristic is overriden in StudentAgent class
    def heuristic(self):
        return 1
            
    # Computes the Upper Confidence Tree of the Node        
    def uct_value(self, current_state):       
        # If the node was not visited, we will prioritize the exploration of the  node 
        # by setting its value to infinity
        if self.visits == 0 or current_state == 0:
            return math.inf

        # Grooming UCT to always favor/disfavor node we know are either winning or losing
        if self.win:
            return math.inf
        elif self.lost:
            return -math.inf
        
        # Otherwise return its confidence bound value
        return self.wins / self.visits + UCT_EXPLORATION_RATE * math.sqrt(math.log(current_state)/self.visits)
    
    # Generates the possible children of the Node
    def generate_possible_moves(self, max_step):
        move_directions = ((-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)) # Move Up, Right, Down, Left 
        
        # Make a difference betweeen agent's turn and opponent's turn
        if self.agent_turn:
            current_pos = self.my_pos
            other_pos = self.adv_pos
            new_node_fn = lambda new_chess_board, pos, dir: Node(self, new_chess_board, pos, dir, self.adv_pos, not self.agent_turn) 
        else:
            current_pos = self.adv_pos
            other_pos = self.my_pos
            new_node_fn = lambda new_chess_board, pos, dir: Node(self, new_chess_board, self.my_pos, dir, pos, not self.agent_turn) # Note we swap the position with adv
        
        # Initialize a queue to keep track of the nodes we are adding to possible children 
        queue = [(current_pos, 0)]
        # A list to keep track of the nodes we visited so far to avoid coming back on our steps 
        visited = []

        # Perform Breadth-First Search to find all possible moves
        while queue:
            pos, step = queue.pop(0)
            
            if pos in visited:
                continue

            visited.append(pos)

            if step >= max_step:
                continue
            
            # Find all the possible possitions we can move to
            for row, col, dir in move_directions:
                if self.chess_board[pos[0], pos[1], dir]:
                    continue

                if (pos[0] + row, pos[1] + col) == other_pos:
                    continue

                queue.append(((pos[0] + row, pos[1] + col), step + 1))
        
        # For all possible positions, count how many walls we can place
        for pos in visited:
            for _, _ , dir in move_directions:
                if self.chess_board[pos[0], pos[1], dir]:
                    continue
                new_chess_board = deepcopy(self.chess_board)
                new_chess_board[pos[0], pos[1], dir] = 1
                new_node = new_node_fn(new_chess_board, pos, dir)
                # Add a possible child here
                self.possible_children.append((new_node, new_node.heuristic()))

        # Sort the possible children according to a heuristic
        # Sorting in ascending order for adversary because they seek to minimize our utility
        # Sorting in descending order for agent because they want to maximize utility 
        self.possible_children = sorted(self.possible_children, key = lambda val: val[1], reverse=self.agent_turn)
        
    # Selects the next children to expand according to our guess/heuristic of what a logical next move is 
    def push_next_children(self, n):
        temp = self.possible_children[:n]
        self.children += temp
        self.possible_children = self.possible_children[n:]
        return temp
        
    # Generates the next possible children from a node and pushes them as children
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
           
    def rollout(self, num_rollouts, max_step):
        rollout = HeuristicRollout(self)
        rollout.run(num_rollouts, max_step)
  
        
class HeuristicRollout:
    def __init__(self, curr_state):
        self.curr_state = curr_state # Node
    
    # TODO define a decay schedule
    @staticmethod
    def rollout_decay(num_rollouts):
        return num_rollouts*ROLLOUT_DECAY
    
    # TODO misuse of num_rollouts + acc
    # Recursive helper method to perform a rollout
    def rec_rollout(self, state_to_explore, num_rollouts, max_step):
        # state_to_explore is a node
        state_to_explore.expand(0, max_step)
        
        acc = 0
        
        for (next_state, heur) in state_to_explore.possible_children[:int(num_rollouts)]:
            
            (is_endgame, my_score, adv_score) = endgame(next_state.my_pos, next_state.adv_pos, next_state.chess_board) # TODO approximate connected component 
            
            if is_endgame:
                if my_score < adv_score:
                    next_state.lost = True
                    next_state.backpropagate(0 , 1)
                else:
                    next_state.win = True
                    next_state.backpropagate(1 , 1)
                
                acc+=1
                continue
            
            acc += self.rec_rollout(next_state, HeuristicRollout.rollout_decay(num_rollouts), max_step)
        
        return acc
            
    def run(self, num_rollouts, max_step):
        self.rec_rollout(self.curr_state, num_rollouts, max_step)
        

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.root_state = None
        self.first_run = True # Initial state of the game
        Node.heuristic = heuristic # Redefine heuristic here

    def update_tree(self, chess_board, my_pos, adv_pos):
        # 1st run of the game
        if not self.root_state:
            self.root_state = Node(None, chess_board, my_pos, None, adv_pos, True)
            return

        # We are in the middle of the game
        next_state = Node(None, chess_board, my_pos, None, adv_pos, True)

        # Perform a search with depth 2 best move for us given that the adversary will pick the worst for for us (minimax)
        def depth_2_search(node, depth):
            if depth == 2:
                if node == next_state:
                    return node
                return None

            if node.expanded:
                for child in node.children:
                    if t := depth_2_search(child[0], depth + 1):
                        return t

                for child in node.possible_children:
                    if t := depth_2_search(child[0], depth + 1):
                        return t
            
            return None

        if node := depth_2_search(self.root_state, 0):
            node.parent = None
            self.root_state = node
        else:
            self.root_state = next_state


    def step(self, chess_board, my_pos, adv_pos, max_step):
        initial_time = time.time()

        self.update_tree(chess_board, my_pos, adv_pos)

        def find_best_uct(init_node):
            uct_max = -math.inf
            uct_max_child = None
            uct_to_explore = [init_node]

            while uct_to_explore:
                node = uct_to_explore.pop(0)
                if node.expanded:
                    for child in node.children:
                        uct_to_explore.append(child[0])
                
                if uct_max < node.uct_value(self.root_state.visits):
                    uct_max = node.uct_value(self.root_state.visits)
                    uct_max_child = node
            
            return uct_max, uct_max_child

        # Leveraging the 30 seconds to expand our search
        if self.first_run:
            self.first_run = False
            z = FIRST_COMPUTATION_TIME
        else:
            z = COMPUTATION_TIME
        
        while time.time() - initial_time < z - TIME_DELTA:
            (uct, best_child) = find_best_uct(self.root_state)
            expanded = best_child.expand(NODES_TO_EXPAND, max_step) # Expand children (Search)
            for child in expanded:
                child[0].rollout(NUM_ROLLOUTS, max_step) # Set number of rollouts (apprx)
        
        best_child = self.root_state.children[0][0]
        best_child_uct = -math.inf
        n = self.root_state.visits

        for direct_children in self.root_state.children:
            if direct_children[0].win:
                best_child = direct_children[0]
                break

            if direct_children[0].lost:
                continue

            children = [x for x in direct_children[0].children]
            if not children:
                min_child_uct = direct_children[0].uct_value(n)
                min_child = direct_children[0]

                if min_child_uct > best_child_uct:
                    best_child = min_child
                    best_child_uct = min_child_uct
            else:
                min_child_uct, min_child = min([(x.uct_value(n), x) for x, _ in children], key = lambda x: x[0])

                if min_child_uct > best_child_uct:
                    best_child = min_child.parent
                    best_child_uct = min_child_uct

        return best_child.my_pos, best_child.my_dir




