from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import *
from scipy.optimize import minimize
import inspect

penalidade_interior = lambda x: f(x)+rp*sum([(-1/g(x) if g(x)<=(-C*(rp)**a) else -(2*(-C*(rp)**a)-g(x))/((-C*(rp)**a)**2)) for g in restricoes])
penalidade_exterior = lambda x: f(x)+rp*sum([g(x) for g in restricoes])

C = 0.15
a = 0.5

global metodo
metodo = None

global x0
x0     = None

global max_it
max_it = None

global tol
tol    = None

global gama
gama   = None

global rp
rp     = None

global f
f      = None

global _restricoes
restricoes = None

global modo
modo = None

def set_param (problema):
 global metodo
 metodo = problema["metodo"]

 global x0
 x0 = problema["x0"]

 global max_it
 max_it = problema["max_it"]

 global tol
 tol    = problema["tol"]

 global gama
 gama   = problema["gama"]

 global rp
 rp     = problema["rp"]

 global f
 f      = problema["f"]

 global restricoes
 restricoes = problema["restricoes"]

 global modo
 modo = problema["modo"]

def MPFI():
 global gama
 gama = 1/(10.0*gama)
 f_obj = lambda x: f(x)+rp*sum([(-1/g(x) if g(x)<=(-C*(rp)**a) else -(2*(-C*(rp)**a)-g(x))/((-C*(rp)**a)**2)) for g in restricoes])
 x = x0

 serie_erro = []
 serie_fx = []

 print("{}\t{}".format("f(x)","erro"))
 for i in range(max_it):
    x_ant=x
    x=minimize(f_obj, x, method=metodo, tol=tol).x
    erro = abs(f_obj(x)-f_obj(x_ant))
    serie_erro.append(erro)
    serie_fx.append(f_obj(x))
    
    print("{}\t{}".format(f(x),erro))
    if erro<=tol:
        #plotaConvergencia(serie_erro,serie_fx)
        return x
    global rp
    rp = gama*rp

def MPFE():
 f_obj = lambda x: f(x)+rp*sum([g(x) for g in restricoes])**2
 xant = x0

 serie_erro = []
 serie_fx = []
 print("{}\t{}".format("f(x)","erro"))
 for i in range(max_it):
    xatual = minimize(f_obj, xant, method=metodo, tol=tol).x
    erro = abs(f_obj(xatual)-f_obj(xant))
    serie_erro.append(erro)
    serie_fx.append(f_obj(xatual))
    print("{}\t{}".format(f(xatual),erro))
    if erro.all()<=tol:
        plotaConvergencia(serie_erro,serie_fx)
        return xatual
    else:
        xant = xatual
        global rp        
        rp = gama*rp
 return xatual

def LA():
 global lamb
 lamb = 0.0
 rpmax=10**5
 h = restricoes[0]
 f_obj = lambda x: f(x)+sum([lamb*g(x) for g in restricoes]) + rp*sum([g(x) for g in restricoes])**2
 x = x0
 f0 = f_obj(x)

 serie_erro = []
 serie_fx = []

 print("{}\t{}\t{}\t{}".format("x_1","x_2","f(x)","erro"))
 for i in range(max_it):
    x = minimize(f_obj, x, method=metodo, tol=tol).x
    f1 = f_obj(x)
    erro = abs(f1-f0)
    serie_erro.append(erro)
    serie_fx.append(f_obj(x))


    f0 = f1
    print("{}\t{}\t{}\t{}".format(x[0],x[1],f(x),erro))
    if erro<=tol:
        plotaConvergencia(serie_erro,serie_fx)
        return x
    else:
        lamb0 = lamb
        global rp
        lamb = lamb+2*rp*h(x);#h eh restricao de igualdade
        if abs(lamb-lamb0)<=tol:
            plotaConvergencia(serie_erro,serie_fx)
            return x
        else:
            rp = gama*rp
            rp = rpmax if rp>rpmax else rp
    if i==max_it:
            plotaConvergencia(serie_erro,serie_fx)
            return x

def plota (xMin,xMax,h,pontos):
   fig = figure()
   ax  = fig.add_subplot(111, projection='3d')

   X = np.arange(xMin, xMax, h)
   Y = np.arange(xMin, xMax, h)
   X, Y = np.meshgrid(X, Y)
   Z = f([X,Y])


   superficie = ax.contour(X, Y, Z, cmap=cm.hot, antialiased=True,alpha=0.8)
   for ponto in pontos:
      ax.scatter(ponto[0],ponto[1],ponto[2],marker=["o","^"][0],c=["r","b"][1],s=10)
      ax.text(ponto[0],ponto[1],ponto[2],s=str(ponto))
   savefig("contorno.png")

   superficie = ax.plot_surface(X, Y, Z, cmap=cm.hot,linewidth=1.0, antialiased=True,alpha=0.8)
   for ponto in pontos:
      ax.scatter(ponto[0],ponto[1],ponto[2],marker=["o","^"][0],c=["r","b"][1],s=10)
      ax.text(ponto[0],ponto[1],ponto[2],s=str(ponto))
   savefig("superficie.png")

def plotaConvergencia (x,y):
 plot(range(len(x)),x,'-g^',label = "Erro")
 plot(range(len(x)),y,'-r^',label = "f(x)")
 xlabel('Iteração')
 ylabel('Valor')
 legend()
 suptitle("Min f(x) = " + str(y[-1]) + "\nIterações = " + str(len(x)) + "\nErro = " + str(x[-1]),fontsize=10)
 savefig("convergencia.png")

def solve ():
 x = modo()
 return x
