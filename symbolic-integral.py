from sympy import *

x = Symbol('x')
l = Symbol('l')
t = Symbol('t')

result = integrate(1/(1-l*(exp(t) * (1 + t))), t)
print(result)

