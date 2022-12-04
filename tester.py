import simulator
import agents.student_agent

# No multithreading allowed b/c of the global variables

sim_args = []
agent_args = []

default_get_args = simulator.get_args

def args_init(args_dict):
    for key, value in args_dict.items():
        if hasattr(agents.student_agent, key):
            setattr(agents.student_agent, key, value)
        else:
            print("Invalid key {}".format(key)) 

def args_override(sim_args):
    working_copy = default_get_args()
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
    run_sim({}, {})

