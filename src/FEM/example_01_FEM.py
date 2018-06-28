from FEM import *

e = Structure()

e.insert_node(0,0.0)
e.insert_node(1,1.0)
e.insert_node(2,2.0)

e.insert_element(0,(0,1),100.0)
e.insert_element(1,(1,2),100.0)

e.apply_imposed_displacement(0,0.0)
#e.apply_force(2,1.0)

e.apply_pressure(0,(0.5,0.5))
e.apply_pressure(1,(0.5,0.5))

e.solve_system_of_equation()
e.print_results()

def exact_u(x):
    return -0.5*x**2/(2.0*100.0) + 0.5 * x * 2.0 / 100.0

def exact_N(x):
    return -0.5 * x  + 0.5 * 2.0

e.plot_results(exact_u,exact_N)
