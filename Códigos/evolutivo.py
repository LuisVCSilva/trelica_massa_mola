from problema import *
from scipy.optimize import differential_evolution

for i in range(99999):
 bounds = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.1, -0.1]] + [[-float(i%2),abs(float(i%2)-1)] for i in range(36)]#+ [[]] [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
 P2 = differential_evolution(funcao_objetivo, bounds,strategy='best1bin', maxiter=None, popsize=80,disp=True, tol=0.01, mutation=(0.5, 1), recombination=0.7, init='latinhypercube').x.reshape((-1, 2),)
 print(P2)
 print(funcao_objetivo(P2))
 plota(P2)
 savefig(str(funcao_objetivo(P2))+".png")
 savetxt(str(funcao_objetivo(P2))+'.csv', P2, delimiter=',',header='x,y')
