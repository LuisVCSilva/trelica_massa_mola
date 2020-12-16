from numpy import *
from scipy.constants import g
from matplotlib.pyplot import *
from numpy import eye as identidade

m = .1    # massa sobre a estrutura
n = 20    # qtde de vertices/massas
e = .1    # Distancias inicial entre vertices
l = e     # comprimento das molas em situacoes sem deformacao
k = 10000 # constante de rigidez da mola

pesos_iniciais = zeros((n, 2))
pesos_iniciais[:, 0] = repeat(e * arange(n // 2), 2)
pesos_iniciais[:, 1] = tile((0, -e), n // 2)


p_0 = np.zeros((n, 2))
p_0[:, 0] = np.repeat(e * np.arange(n // 2), 2)
p_0[:, 1] = np.tile((0, -e), n // 2)

matriz_adjacencia = identidade(n, n, 1) + identidade(n, n, 2)
matriz_molas = l * (identidade(n, n, 1) + identidade(n, n, 2))
for i in range(n // 2 - 1):
    matriz_molas[2 * i + 1, 2 * i + 2] = matriz_molas[2 * i + 1, 2 * i + 2]*sqrt(2)

I, J = nonzero(matriz_adjacencia)

distancias = lambda pos: sqrt((pos[:, 0] - pos[:, 0][:, newaxis])**2 + (pos[:, 1] - pos[:, 1][:, newaxis])**2)


def cria_cor_mola(c):
    min_c, max_c = -0.00635, 0.00836
    normalizacao = (max_c - c) / (max_c - min_c)
    color = cm.gist_heat(normalizacao)
    tom = sqrt(abs(normalizacao - 0.5) * 2)
    return (tom * color[0], tom * color[1], tom * color[2], color[3])


def plota(P):
    figure(figsize=(5, 4))
    # Parede.
    axvline(0, color='k', lw=3)
    # matriz de distancias
    D = distancias(P)
    # Plotagem das molas.
    for i, j in zip(I, J):
        # Cor é proporcional a tensão na mola/esticamento da mola
        c = D[i, j] - matriz_molas[i, j]
        plot(P[[i, j], 0], P[[i, j], 1],lw=2, color=cm.copper(c*150))
    # Plota os nós
    plot(P[[I, J], 0], P[[I, J], 1], 'ok',)
    # Configuracao dos eixos.
    axis('equal')
    xlim(P[:, 0].min() - e / 2, P[:, 0].max() + e / 2)
    ylim(P[:, 1].min() - e / 2, P[:, 1].max() + e / 2)
    xticks([])
    yticks([])
    title("f(min) = " + str(funcao_objetivo(P)))

def funcao_objetivo(y):
    # Matriz de pesos
    y = y.reshape((-1, 2))

    # Matriz de Distancias
    D = distancias(y)

    # Energia potencial total = energia gravitacional + energia elastica
    return (m * g * y[:, 1].sum() + 0.5 * (k * matriz_adjacencia * (D - matriz_molas)**2).sum())

#plota(p_0)
#savefig("configuracao_inicial.png")
