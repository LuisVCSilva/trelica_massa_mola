from numpy import *
from util import *
from problema import *
'''
default

 "C"          : 0.15,\           #0.15
 "a"          : 0.5,\            #0.5
 "rp"         : 10000.0,\        #10.0
 "x0"         : [1.0,1.0],\      #1.0, 1.0
 "tol"        : 10**(-6),\       #10^-6
 "gama"       : 3.00,\           #3.0 para MPFE 0.003 para MPFI
 "max_it"     : 100,\            #100
 
 
 "metodo"     : "Nelder-Mead",\ #Nelder-Mead
 "modo"       : MPFI,\          #MPFE
'''

def main():
 problema = {\
 "C"          : 0.15,\
 "a"          : 0.5,\
 "rp"         : 10000.0,\
 "x0"         : p_0,\
 "tol"        : 10**(-6),\
 "gama"       : 3.00,\
 "max_it"     : 100,\
 
 
 "metodo"     : "BFGS",\
 "modo"       : MPFE,\
 "f"          : funcao_objetivo,\
 "restricoes" : [\
		lambda x: x[0],\
                ],\
 }


 set_param(problema)
 minimo = solve().reshape((-1, 2),)
 print((minimo,problema["f"](minimo)))

 plota(minimo)
 show()

main()
