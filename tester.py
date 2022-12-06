import simulator
import agents.student_agent

# No multithreading allowed b/c of the global variables

sim_args = [{"board_size": 5}]
agent_args = [{"TIME_DELTA": 0.1, "COMPUTATION_TIME":1.99, "FIRST_COMPUTATION_TIME": 29.99, "NUM_ROLLOUTS":100, "ROLLOUT_DECAY":0.3, "NODES_TO_EXPAND":3, "UCT_EXPLORATION_RATE":0.5, "METRICS_CONSTANT": 0.1},\
                {"TIME_DELTA": 0.1, "COMPUTATION_TIME":1.99, "FIRST_COMPUTATION_TIME": 29.99, "NUM_ROLLOUTS":100, "ROLLOUT_DECAY":0.3, "NODES_TO_EXPAND":3, "UCT_EXPLORATION_RATE":0.5, "METRICS_CONSTANT": 0.1},\
                {"TIME_DELTA": 0.1, "COMPUTATION_TIME":1.99, "FIRST_COMPUTATION_TIME": 29.99, "NUM_ROLLOUTS":100, "ROLLOUT_DECAY":0.3, "NODES_TO_EXPAND":3, "UCT_EXPLORATION_RATE":0.5, "METRICS_CONSTANT": 0.1},\
                {"TIME_DELTA": 0.1, "COMPUTATION_TIME":1.99, "FIRST_COMPUTATION_TIME": 29.99, "NUM_ROLLOUTS":100, "ROLLOUT_DECAY":0.3, "NODES_TO_EXPAND":3, "UCT_EXPLORATION_RATE":0.5, "METRICS_CONSTANT": 0.1},\
                {"TIME_DELTA": 0.1, "COMPUTATION_TIME":1.99, "FIRST_COMPUTATION_TIME": 29.99, "NUM_ROLLOUTS":100, "ROLLOUT_DECAY":0.3, "NODES_TO_EXPAND":3, "UCT_EXPLORATION_RATE":0.5, "METRICS_CONSTANT": 0.1},]


default_get_args = simulator.get_args

def args_init(args_dict):
    for key, value in args_dict.items():
        if hasattr(agents.student_agent, key):
            setattr(agents.student_agent, key, value)
        else:
            print("Invalid key {}".format(key)) 

def args_override(sim_args):
    working_copy = default_get_args()
    working_copy.player_1 = "random_agent"
    working_copy.player_2 = "student_agent"
    for key, value in sim_args.items():
        if hasattr(working_copy, key):
            setattr(working_copy, key, value)
        else:
            print("Invalid key {}".format(key))

    return lambda: working_copy


def run_sim(agent_args, sim_args):
    args = args_override(sim_args)()
    args_init(agent_args)
    sim = simulator.Simulator(args)
    return sim.run()

if __name__ == "__main__":
    for sim in sim_args:
        for agent in agent_args:
            run_sim(agent, sim)


