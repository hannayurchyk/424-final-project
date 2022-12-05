import ui
import numpy
import world
import agents.student_agent

list = numpy.array([[[ True, False, False,  True],\
        [ True,  True,  True, False],\
        [ True, False, False,  True],\
        [ True,  True, False, False]],\
       [[ True, False, False,  True],\
        [ True, False, False, False],\
        [ True, False,  True, False],\
        [ True,  True, False, False]],\
       [[ True,  True,  True,  True],\
        [ True,  True,  True, False],\
        [ True, False,  True, False],\
        [ True,  True,  True,  True]],\
       [[False, False,  True,  True],\
        [False,  True,  True, False],\
        [ True, False,  True,  True],\
        [False,  True,  True, False]]])

my = (2, 3)
adv = (2, 0)

print(agents.student_agent.endgame(my, adv, list))

# list = numpy.array([[[ True,  True, False,  True],\
#         [ True, False, False,  True],\
#         [ True, False,  True, False],\
#         [ True,  True, False, False]],\
#        [[ True, False, False,  True],\
#         [ True,  True,  True, False],\
#         [ True, False, False, False],\
#         [ True,  True, False, False]],\
#        [[ True, False,  True,  True],\
#         [ True,  True,  True,  True],\
#         [ True,  True,  True,  True],\
#         [ True,  True, False, False]],\
#        [[ True, False,  True,  True],\
#         [ True, False,  True, False],\
#         [False,  True,  True,  True],\
#         [False,  True,  True,  True]]])

# my = (2, 2)
# adv = (2, 1)

# print(agents.student_agent.endgame(my, adv, list))


viz = ui.UIEngine(grid_width=4, world=world.World())

viz.render(list, my, adv)

input()