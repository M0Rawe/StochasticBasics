import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tree_gen

(nu,nx,N,lf,lr,ts) = (2,4,20,1,1,0.1)       #nu = Number of inputs, nx = number of states
ltot = lf+lr

(xref, yref, psiref, betaref) = (1,1,0,0)
(q,qpsi,qbeta,r,qN,qPsiN,qBetaN) = (10.0, 0.1, 0.1, 1, 100, 2,2)

u = cs.SX.sym('u', nu*N)
z0 = cs.SX.sym('z0', nx)

(x,y,psi,beta) = (z0[0], z0[1], z0[2], z0[3])

p = np.array([[0.3,0.15,0.2,0.2,0.15],      #Tranisitional probabilitiy of tree
              [0.2,0.3,0.15,0,0.35],
              [0.25,0.2,0.3,0.25,0],
              [0.3,0,0.25,0.2,0.25],
              [0.2,0.25,0,0.3,0.25]])
v_tree = np.array([0.4, 0.15, 0.15,0.15,0.15])       #initial probability of starting node
(N_tree, tau) = (3, 2)                   #N -> number of stages, tau -> stage at tree becomes stopped tree.
tree = tree_gen.MarkovChainScenarioTreeFactory(p, v_tree, N_tree, tau).create()

all_nodes = []
for i in range(tree.num_stages):
    all_nodes.append(tree.nodes_at_stage(i))

probabilities = []
for i in all_nodes:
    innerlist = []
    for j in i:
        innerlist.append(tree.probability_of_node(j))
    probabilities.append(innerlist)

all_nodes_list = []
for i in all_nodes:
    all_nodes_list.append(i.tolist())

combined = zip(all_nodes_list,probabilities)
nodes_flat = [item for sublist in all_nodes_list for item in sublist]
probabilities_flat = [item for sublist in probabilities for item in sublist]

tree.bulls_eye_plot()
#print(probabilities)
#print(all_nodes)
ancestry_list = [[0 for x in range(tree.num_stages)] for y in range(len(all_nodes[-1]))]
ancestry_list_test = [[0 for x in range(tree.num_stages)] for y in range(len(all_nodes[-1]))]
ancestry_list_probabilities = [[0 for x in range(tree.num_stages)] for y in range(len(all_nodes[-1]))]
ancestry_list_probabilities_test = [[0 for x in range(tree.num_stages)] for y in range(len(all_nodes[-1]))]
cost = 0
# for i in all_nodes[-1]:
#     print(f"Current node is {i}, its ancestor is {tree.ancestor_of(i)}")
#     ancestry_list[i][tree.ancestor_of(i)] = 

for count, node in enumerate(all_nodes[-1]):
    #print(node)
    
    ancestry_list_probabilities[count][0] = tree.probability_of_node(node)
    ancestry_list_probabilities[count][1] = tree.probability_of_node(tree.ancestor_of(node))
    ancestry_list_probabilities[count][2] = tree.probability_of_node(tree.ancestor_of(tree.ancestor_of(node)))
    ancestry_list_probabilities[count][3] = tree.probability_of_node(tree.ancestor_of(tree.ancestor_of(tree.ancestor_of(node))))

for count, node in enumerate(all_nodes[-1]):
    #print(node)
    #print(f"count = {count}, node = {node}")
    ancestry_list[count][0] = node
    ancestry_list[count][1] = tree.ancestor_of(node)
    ancestry_list[count][2] = tree.ancestor_of(tree.ancestor_of(node))
    ancestry_list[count][3] = tree.ancestor_of(tree.ancestor_of(tree.ancestor_of(node)))

for count, node in enumerate(all_nodes[-1]):
    for j in range(tree.num_stages):
        #print(f"count = {count}, j = {j}, node = {node}")
        ancestry_list_test[count][j] = node
        node = tree.ancestor_of(node)

for count, node in enumerate(all_nodes[-1]):
    for j in range(tree.num_stages):
        #print(f"count = {count}, j = {j}, node = {node}")
        ancestry_list_probabilities_test[count][j] = tree.probability_of_node(node)
        node = tree.ancestor_of(node)
        


print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in ancestry_list_probabilities]))
print("--------------------------------------------")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in ancestry_list_probabilities_test]))
# for i in tree.nodes_at_stage(tree.num_stages-1)[::-1]:
#     print(f"At node {i}, its ancestor is {tree.ancestor_of(i)} with probabilities of {tree.probability_of_node(i)}"\
#             f" and {tree.probability_of_node(tree.ancestor_of(i))} respectively")

# for i in all_nodes[::-1]:
#     for j in i[::-1]:
#         if j==0:
#             break
#         print(f"At node {j}, its ancestor is {tree.ancestor_of(j)} with probabilities of {tree.probability_of_node(j)}"\
#             f" and {tree.probability_of_node(tree.ancestor_of(j))} respectively")


problem = og.builder.Problem(u, z0, cost)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("bicycle")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                            meta,
                                            build_config,
                                            solver_config)

#builder.build()

# Use TCP server
# ------------------------------------
# mng = og.tcp.OptimizerTcpManager('basic_optimizer/bicycle')
# mng.start()

# mng.ping()
# solution = mng.call([-0.8, -0.8, 0.0, 0.0], initial_guess=[1.0] * (nu*N))
# mng.kill()


# # Plot solution
# # ------------------------------------
# time = np.arange(0, ts*N, ts)
# u_star = solution['solution']
# ux = u_star[0:nu*N:2]
# uy = u_star[1:nu*N:2]
# upsi = u_star[2:nu*N:2]
# ubeta = u_star[3:nu*N:2]

# plt.subplot(4,1,1)
# plt.plot(time, ux, '-o')
# plt.ylabel('u_x')
# plt.subplot(4,1,2)
# plt.plot(time, uy, '-o')
# plt.ylabel('u_y')
# plt.subplot(4,1,3)
# plt.plot(time[0:-1], upsi, '-o')
# plt.ylabel('u_psi')
# plt.subplot(4,1,4)
# plt.plot(time[0:-1], ubeta, '-o')
# plt.ylabel('u_beta')
# plt.xlabel('Time')
# plt.show()

# # Plot trajectory
# # ------------------------------------
# x_init = [-1.0,-1.0,0.0,0.0]
# x_states = [0.0] * (nx*(N+2))
# x_states[0:nx+1] = x_init
# for t in range(0, N):
#     u_t = u_star[t*nu:(t+1)*nu]

#     x = x_states[t * nx]
#     y = x_states[t * nx + 1]
#     psi = x_states[t* nx + 2]
#     beta = x_states[t * nx + 3]

#     beta_dot = cs.atan((lr/ltot)*cs.tan(u_t[1]))
#     psi_dot = (u_t[0]/ltot)*cs.cos(beta_dot)*cs.tan(u_t[1])*ts

#     x_states[(t+1)*nx]  = x + ts*(u_t[0]*cs.cos(psi_dot+beta_dot))
#     x_states[(t+1)*nx+1] = y + ts*(u_t[0]*cs.sin(psi_dot+beta_dot))
#     x_states[(t+1)*nx+2] = psi + ts*psi_dot
#     x_states[(t+1)*nx+3] = beta + ts*beta_dot

# xx = x_states[0:nx*N:nx]
# xy = x_states[1:nx*N:nx]

# # print(x_states)
# print(xx)
# plt.plot(xx, xy, '-o')
# plt.show()

# print(f"Solution exit status "+str(solution["exit_status"]))
# print(f"Solution penalty "+str(solution["penalty"]))
# print(f"Solution time "+str(solution["solve_time_ms"]))