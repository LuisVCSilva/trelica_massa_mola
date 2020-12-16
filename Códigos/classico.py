from problema import *
from scipy.optimize import minimize

global k
k = 0
metodo = ["L-BFGS-B","TNC","SLSQP"][1]

fx = []

def save (x):
 figure(1)
 global k

 #plota(x.reshape((-1, 2),))

 fx.append(funcao_objetivo(x))
 plot(fx)
 title(metodo)
 xlabel("Iterações")
 ylabel("min f(x)")
 savefig(str(k)+".png")
 k = k+1
 if k%10==0:
  close('all')

#plota(p_0)
#savefig("configuracao_inicial.png")


print("Executando " + metodo)
bounds = c_[pesos_iniciais[:2, :].ravel(), pesos_iniciais[:2, :].ravel()].tolist() + [[-1.0,1.0]] * (2 * (n - 2))
P1 = minimize(funcao_objetivo, pesos_iniciais.ravel(),method=metodo,callback=save,options={'disp': True,'maxfun': 99999999, 'maxiter': 99999999,'maxls':9999999,'eta':0.6,'stepmx':0.00001},bounds=bounds).x.reshape((-1, 2),)
exato = funcao_objetivo(P1)
print(P1)
print(exato)
#plota(P1)
#savefig(metodo+".png")
#savetxt(metodo+".csv", P1, delimiter=',',header='x,y')

