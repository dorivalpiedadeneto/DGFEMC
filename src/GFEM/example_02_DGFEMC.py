from DGFEMC import *
from numpy import linspace

nelements = 1
nnodes = nelements + 1
enrich = True
xi = 0.0
xf = 2.0
p = 0.5
Element._number_of_gauss_points = 3

e = Structure()

xs = linspace(xi,xf,nnodes)

for i in range(nnodes):
    e.insert_node(i,xs[i])

for i in range(nelements):
    e.insert_element(i,(i,i+1),100.0)


e.apply_imposed_displacement(0,0.0)

L = xf - xi
for i in range(nelements):
    xni = e._nodes[i].coord
    xnf = e._nodes[i+1].coord
    pi = p * (L-xni)/L
    pf = p * (L-xnf)/L
    e.apply_pressure(i,(pi,pf))


e.create_regular_gdofs()

if enrich:
    for i in range(len(e._nodes)):
        n = e._nodes[i]
        n.insert_gdof(Generalized_dof(enrichment=Shifted(n,1)))
#       n.insert_gdof(Generalized_dof(enrichment=Shifted(n,2)))
#       n.insert_gdof(Generalized_dof(enrichment=Shifted(n,3)))

e.solve_system_of_equation(perturbation=True)
e.print_results()

def exact_u(x):
    L = xf - xi
    return p * x ** 3 / (6.0 * 100.0 * L) - p * x **2 / (2.0 * 100.0) + ((p * L / 2.0) / 100.0) * x

def exact_N(x):
    L = xf - xi
    return p * x ** 2 / (2.0 * L) - p * x + p * L / 2.0

e.plot_results(exact_u,exact_N,100)
e.evaluate_error(exact_u,exact_N)
