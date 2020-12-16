from numpy import *
from scipy.constants import g
from scipy.optimize import minimize,differential_evolution,basinhopping,dual_annealing
from shgo import shgo
from matplotlib.pyplot import *
import pyswarms as pso

#import matplotlib
#matplotlib.rcParams['savefig.dpi'] = matplotlib.rcParams['figure.dpi'] = 144

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

m = .1   # massa sobre a estrutura (kg)
n = 20    # qtde de vertices/massas
e = .1    # distancia inicial entre vertices
l = e     # comprimento das molas em situacoes sem deformacao
k = 10000 # constante de rigidez da mola

P0 = zeros((n, 2))
P0[:, 0] = repeat(e * arange(n // 2), 2)
P0[:, 1] = tile((0, -e), n // 2)

A = eye(n, n, 1) + eye(n, n, 2)

L = l * (eye(n, n, 1) + eye(n, n, 2))
for i in range(n // 2 - 1):
    L[2 * i + 1, 2 * i + 2] *= sqrt(2)

I, J = nonzero(A)

dist = lambda P: sqrt((P[:, 0] - P[:, 0][:, newaxis])**2 +
                         (P[:, 1] - P[:, 1][:, newaxis])**2)


def spring_color_map(c):
    min_c, max_c = -0.00635369422326, 0.00836362559722
    ratio = (max_c - c) / (max_c - min_c)
    color = cm.coolwarm(ratio)
    shading = sqrt(abs(ratio - 0.5) * 2)
    return (0.0,0.0,0.0)#(shading * color[0], shading * color[1], shading * color[2], color[3])


def show_bar(P):
    figure(figsize=(5, 4))
    # Wall.
    axvline(0, color='k', lw=3)
    # Distance matrix.
    D = dist(P)
    # We plot the springs.
    for i, j in zip(I, J):
        # The color depends on the spring tension, which
        # is proportional to the spring elongation.
        c = D[i, j] - L[i, j]
        plot(P[[i, j], 0], P[[i, j], 1],
                 lw=2, color=cm.copper(c*150))
    # We plot the masses.
    plot(P[[I, J], 0], P[[I, J], 1], 'ok',)
    # We configure the axes.
    axis('equal')
    xlim(P[:, 0].min() - e / 2, P[:, 0].max() + e / 2)
    ylim(P[:, 1].min() - e / 2, P[:, 1].max() + e / 2)
    xticks([])
    yticks([])

show_bar(P0)
title("Configuracao inicial")


def funcao_objetivo(P):
    # Matriz de pesos
    P = P.reshape((-1, 2))

    # Matriz de distancias
    D = dist(P)

    # Energia potencial total = energia gravitacional + energia elastica
    #print(g * m * P[:, 1].sum() + .5 * (k * A * (D - L)**2).sum())
    return (g * m * P[:, 1].sum() + .5 * (k * A * (D - L)**2).sum())

funcao_objetivo(P0.ravel())
bounds = c_[P0[:2, :].ravel(), P0[:2, :].ravel()].tolist() + [[-1.0, 1.0]] * (2 * (n - 2))
#print(len(bounds))
x = np.ones(2)
eps = np.sqrt(np.finfo(float).eps)
P1 = minimize(funcao_objetivo, P0.ravel(),method=["L-BFGS-B","TNC","SLSQP","trust-ncg"][0],options={'gtol': 1e-6, 'disp': False},bounds=bounds).x.reshape((-1, 2),)
print(P1)
print(funcao_objetivo(P1))
#show_bar(P1)
#show()
#print(bounds)
#lb = None
#ub = None
#print(lb)
#print(ub)
#options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
#optimizer = pso.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
#P2 = optimizer.optimize(funcao_objetivo, iters=1000)
minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
lw = [bound[0] for bound in bounds]
up = [bound[1] for bound in bounds]
print((lw))
print((up))
x0 = P0.ravel()
P2 = dual_annealing(funcao_objetivo, bounds=list(zip(lw, up)), seed=1234).x.reshape((-1, 2),)
print(P2)
print(funcao_objetivo(P2))
