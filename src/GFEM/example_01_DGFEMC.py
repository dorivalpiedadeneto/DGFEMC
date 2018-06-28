from DGFEMC import *
from numpy import linspace

nelements = 1
nnodes = nelements + 1
enrich = True


e = Structure()

xs = linspace(0.0,2.0,nnodes)

for i in range(nnodes):
    e.insert_node(i,xs[i])

for i in range(nelements):
    e.insert_element(i,(i,i+1),100.0)


e.apply_imposed_displacement(0,0.0)

for i in range(nelements):
    e.apply_pressure(i,(0.5,0.5))


e.create_regular_gdofs()

if enrich:
    for i in range(len(e._nodes)):
        n = e._nodes[i]
        n.insert_gdof(Generalized_dof(enrichment=Shifted(n,1)))


#n0 = e._nodes[0]
#n0.insert_gdof(Generalized_dof(enrichment=Shifted(n0,1)))

#n1 = e._nodes[1]
#n1.insert_gdof(Generalized_dof(enrichment=Shifted(n1,1)))

#n2 = e._nodes[2]
#n2.insert_gdof(Generalized_dof(enrichment=Shifted(n2,1)))


e.solve_system_of_equation(perturbation=True)
e.print_results()

def exact_u(x):
    return -0.5*x**2/(2.0*100.0) + 0.5 * x * 2.0 / 100.0 #+ 10.0 * 2.0 / 100.0

def exact_N(x):
    return -0.5 * x  + 0.5 * 2.0 #+ 10.0

e.plot_results(exact_u,exact_N,100)
e.evaluate_error(exact_u,exact_N)
