from sympy import *
init_printing(use_unicode=False, wrap_line=False)
x = Symbol('x')
a = 1.5
b = 1e6
pdf = a * x **(-a-1) / (1-b**-a)
result = integrate(pdf * x, x)
result -= result.subs(x, 1)
print(result, result.subs(x, 1))
out = []
for i in range(1,99):
    out.append(solve(result/2.997 - i/100, x)[0])
print(out)
