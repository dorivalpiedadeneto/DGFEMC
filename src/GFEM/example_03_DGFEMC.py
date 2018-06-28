from DGFEMC import *
from numpy import linspace
from numpy import sin as npsin
from numpy import cos as npcos

nelements = 1
nnodes = nelements + 1
enrich = True
xi = 0.0
xf = 2.0
p = 0.5
Element._number_of_gauss_points = 6

e = Structure()

xs = linspace(xi,xf,nnodes)

for i in range(nnodes):
    e.insert_node(i,xs[i])

for i in range(nelements):
    e.insert_element(i,(i,i+1),100.0)


e.apply_imposed_displacement(0,0.0)

def pressure_function(x):
    global xi, xf, p
    L = xf - xi
    return p * sin(pi*x/L)

e.insert_global_pressure_function(pressure_function)


e.create_regular_gdofs()

if enrich:
    L = xf - xi
    for i in range(len(e._nodes)):
        n = e._nodes[i]
        n.insert_gdof(Generalized_dof(enrichment=Shifted(n,1)))
#       n.insert_gdof(Generalized_dof(enrichment=Shifted(n,2)))
#       n.insert_gdof(Generalized_dof(enrichment=Shifted(n,3)))
#       n.insert_gdof(Generalized_dof(enrichment=Sine(n,L)))

e.solve_system_of_equation(perturbation=True)
e.print_results()

def exact_u(x):
    L = xf - xi
    return p * L ** 2 * npsin(pi * x / L)/(100.0 * pi ** 2) + (p * L / (100.0 * pi)) * x

def exact_N(x):
    L = xf - xi
    return p * L * npcos( pi * x / L ) / pi + p * L / pi

e.plot_results(exact_u,exact_N,100)
e.evaluate_error(exact_u,exact_N)
