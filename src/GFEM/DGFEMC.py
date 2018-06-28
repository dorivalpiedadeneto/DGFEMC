# -*- coding: utf-8 -*- 

# DGFEMC - A Didactic Generalized Finite Element Method Code
# Copyright (C) 2016 Dorival Piedade Neto & Sergio Persival Baroncini Proença
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from numpy import zeros, array, dot, linspace
from numpy.linalg import solve
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# Import for perturbation method (Babuska)
from numpy import eye, diag
from numpy import sqrt as npsqrt
from sys import exit

from math import sqrt, sin, cos, pi

class Node(object):

    def __init__(self, index, coord):
        self._index = index
        self._coord = coord
##        self._disp = 0.0
        self._gdofs = []
    
    def __repr__(self):
        return 'Node # %d at x = %.5f'%(self._index, self._coord)

    def __str__(self):
        return 'Node # %d at x = %.5f'%(self._index, self._coord)

    @property
    def index(self):
        return self._index

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = value

#   @property
#   def displacement(self):
#       return self._disp

#   @displacement.setter
#   def displacement(self, value):
#       self._disp = value

    @property
    def gdofs(self):
        return self._gdofs

    def insert_gdof(self, gdof):
        self._gdofs.append(gdof)

    def number_of_gdofs(self):
        return len(self._gdofs)

    @property
    def displacement(self):
        x = self._coord
        disp = 0.0
        for gdof in self._gdofs:
            value = gdof.value
            Le = gdof.enrichment.Le(x)
            disp += value * Le
        return disp

class Imposed_displacement(object):

    def __init__(self, node_number, value):
        self._node_number = node_number
        self._value = value

    @property
    def node_number(self):
        return self._node_number

    @node_number.setter
    def node_number(self, number):
        self._node_number = number

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

class Force(object):

    def __init__(self, node_number, value):
        self._node_number = node_number
        self._value = value

    @property
    def node_number(self):
        return self._node_number

    @node_number.setter
    def node_number(self, number):
        self._node_number = number

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

class Pressure(object):

    def __init__(self, element_number, values):
        self._element_number = element_number
        self._values = values

    @property
    def element_number(self):
        return self._element_number

    @element_number.setter
    def element_number(self, number):
        self._element_number = number

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val):
        self._values = val


class Element(object):

    _number_of_gauss_points = 3

    def __init__(self, index, nodes, ES):
        self._index = index
        self._nodes = nodes
        self._ES = ES
        self._PU = Partition_of_unity(nodes)

    def __str__(self):
        return 'Element # %d from node # %d to node # %d, ES = %.3f'%(self._index, self._nodes[0].index, self._nodes[1].index, self._ES)

    def __repr__(self):
        return 'Element # %d from node # %d to node # %d, ES = %.3f'%(self._index, self._nodes[0].index, self._nodes[1].index, self._ES)

    @staticmethod
    def number_of_gauss_points(self, n):
        Element._number_of_gauss_points = n

    @property
    def index(self):
        return self._index

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nds):
        self._nodes = nds

    @property
    def ES(self):
        return self._ES

    @ES.setter
    def ES(self, value):
        self._ES = value

    @property
    def PU(self):
        return self._PU

    def lenght(self):
        n0, n1 = self._nodes
        x0 = n0.coord
        x1 = n1.coord
        return x1 - x0

#   def stiffness_matrix(self):
#       ES = self.ES
#       L = self.lenght()
#       kl = ES / L * array([[1.0,-1.0],[-1.0,1.0]])
#       return kl

    def B_matrix(self, qsi):
        B = []
        pou_number = 0
        dx_dqsi = self._PU.dx_dqsi(qsi)
        x = self._PU.x_coord(qsi)
        for node in self._nodes:
            for gdof in node.gdofs:
                phi = self._PU.phi(pou_number,qsi)
                dphi_dqsi = self._PU.dphi_dqsi(pou_number,qsi)
                dphi_dx = dphi_dqsi / dx_dqsi
                enrichment = gdof.enrichment
                Le = enrichment.Le(x)
                dLe_dx = enrichment.dLe_dx(x)
                dphie_dx = dphi_dx * Le + phi * dLe_dx
                B.append(dphie_dx)
            pou_number += 1
        B = array([B])
        return B

    def stiffness_matrix(self):
        ES = self.ES
        L = self.lenght()
        ngdof = 0
        for node in self._nodes:
            ngdof += node.number_of_gdofs()
        kl = zeros((ngdof,ngdof))
        qsis, ws = leggauss(self._number_of_gauss_points)
        for i in range(len(qsis)):
            qsi = qsis[i]
            w = ws[i]
            B = self.B_matrix(qsi)
            J = self._PU.dx_dqsi(qsi)
            kl += ES * dot(B.T,B) * w * J
        return kl

#   def force_vector(self, values):
#       pi, pj = values
#       L = self.lenght()
#       f = array([(2.0*pi+pj)*L/6.0, (pi+2.0*pj)*L/6.0])
#       return  f

    def phi_vector(self, qsi):
        v = []
        pou_number = 0
        x = self._PU.x_coord(qsi)
        for node in self._nodes:
            for gdof in node.gdofs:
                phi = self._PU.phi(pou_number,qsi)
                enrichment = gdof.enrichment
                Le = enrichment.Le(x)
                phie = phi * Le
                v.append(phie)
            pou_number += 1
        v = array(v)
        return v

    def force_vector(self, values):
        pi, pj = values
        PU = Partition_of_unity(self._nodes)
        ngdof = 0
        for node in self._nodes:
            ngdof += node.number_of_gdofs()
        fl = zeros(ngdof)
        qsis, ws = leggauss(self._number_of_gauss_points)
        for i in range(len(qsis)):
            qsi = qsis[i]
            w = ws[i]
            p = pi * PU.phi(0, qsi) + pj * PU.phi(1, qsi)
            J = self._PU.dx_dqsi(qsi)
            v = self.phi_vector(qsi)
            fl += p * v * w * J
        return fl

    def global_pressure_force_vector(self, function):
        ngdof = 0
        for node in self._nodes:
            ngdof += node.number_of_gdofs()
        fl = zeros(ngdof)
        qsis, ws = leggauss(self._number_of_gauss_points)
        for i in range(len(qsis)):
            qsi = qsis[i]
            w = ws[i]
            x = self._PU.x_coord(qsi)
            p = function(x)
            J = self._PU.dx_dqsi(qsi)
            v = self.phi_vector(qsi)
            fl += p * v * w * J
        return fl

#   def indexes(self):
#       ni, nj = self._nodes
#       nni = ni.index
#       nnj = nj.index
#       return (nni, nnj)

    def indexes(self):
        inds = []
        for node in self._nodes:
            for gdof in node.gdofs:
                inds.append(gdof.index)
        return inds

    def displacement(self, qsi):
        x = self._PU.x_coord(qsi)
        disp = 0.0
        pou_number = 0
        for node in self._nodes:
            phi = self._PU.phi(pou_number, qsi)
            for gdof in node.gdofs:
               value = gdof.value
               Le = gdof.enrichment.Le(x)
               disp += value * phi * Le
            pou_number += 1
        return disp

#   def normal_force(self):
#       ni, nj = self._nodes
#       xi = ni.coord; xj = nj.coord
#       di = ni.displacement; dj = nj.displacement
#       ES = self._ES
#       N = ES * (dj - di) / (xj - xi)
#       return N

    def normal_force(self, qsi):
        ES = self._ES
        N  = 0.0
        pou_number = 0
        dx_dqsi = self._PU.dx_dqsi(qsi)
        x = self._PU.x_coord(qsi)
        for node in self._nodes:
            for gdof in node.gdofs:
                phi = self._PU.phi(pou_number,qsi)
                dphi_dqsi = self._PU.dphi_dqsi(pou_number,qsi)
                dphi_dx = dphi_dqsi / dx_dqsi
                enrichment = gdof.enrichment
                Le = enrichment.Le(x)
                dLe_dx = enrichment.dLe_dx(x)
                dphie_dx = dphi_dx * Le + phi * dLe_dx
                value = gdof.value
                N += ES * dphie_dx * value
            pou_number += 1
        return N

    def displacement_error(self, exact_disp):
        error = 0.0
        qsis, ws = leggauss(self._number_of_gauss_points)
        for i in range(len(qsis)):
            qsi = qsis[i]
            w = ws[i]
            x = self._PU.x_coord(qsi)
            J = self._PU.dx_dqsi(qsi)
            d = self.displacement(qsi)
            ed = exact_disp(x)
            error += (ed - d) ** 2 * w * J
        error = sqrt(error)
        return error

    def normal_error(self, exact_N):
        error = 0.0
        qsis, ws = leggauss(self._number_of_gauss_points)
        for i in range(len(qsis)):
            qsi = qsis[i]
            w = ws[i]
            x = self._PU.x_coord(qsi)
            J = self._PU.dx_dqsi(qsi)
            N = self.normal_force(qsi)
            eN = exact_N(x)
            error += (eN - N) ** 2 * w * J
        error = sqrt(error)
        return error


class Structure(object):

    def __init__(self):
        self._nodes = {}
        self._elements = {}
        self._forces = []
        self._imposed_displacements = []
        self._pressures = []
        self._ngdof = 0     # Number of generalized degrees of freedom
        self._global_pressure_function = None

    def insert_node(self, index, coord):
        self._nodes[index] = Node(index,coord)

    def insert_element(self, index, nodes, ES):
        nni, nnj = nodes  # Indexes
        ni = self._nodes[nni]
        nj = self._nodes[nnj]
        self._elements[index] = Element(index, [ni, nj], ES)

    def apply_pressure(self, element_number, values):
        self._pressures.append(Pressure(element_number, values))

    def apply_force(self, node_number, value):
        self._forces.append(Force(node_number,value))

    def apply_imposed_displacement(self, node_number, value):
        self._imposed_displacements.append(Imposed_displacement(node_number, value))

    def insert_global_pressure_function(self, function):
        self._global_pressure_function = function

    def create_regular_gdofs(self):
        for node in self._nodes.values():
            node.insert_gdof(Generalized_dof(enrichment=One(node)))

    def number_gdofs(self):
        count = 0
        for node in self._nodes.values():
            for gdof in node.gdofs:
                gdof.index = count
                count += 1
        self._ngdof = count

    def solve_system_of_equation(self, perturbation=False):
        self.number_gdofs()
#       ndof = len(self._nodes)
        ndof = self._ngdof
        kg = zeros((ndof,ndof))
        fg = zeros(ndof)

        # local matrices
        for element in self._elements.values(): # self._elements is a dictonary
#           ni, nj = element.indexes()
#           row = array((ni,ni,nj,nj))
#           col = array((ni,nj,ni,nj))
#           lrow = array((0,0,1,1))
#           lcol = array((0,1,0,1))
#           kl = element.stiffness_matrix()
#           kg[row, col] += kl[lrow,lcol]
            inds =  element.indexes()
            row = []; col = []
            lrow = []; lcol = []
            ninds = len(inds)
            for i in range(ninds):
                for j in range(ninds):
                    row.append(inds[i])
                    col.append(inds[j])
                    lrow.append(i)
                    lcol.append(j)
            kl = element.stiffness_matrix()
            kg[row, col] += kl[lrow,lcol]
            
        # local vector - pressures
        for pressure in self._pressures:
            element_number = pressure.element_number
            values = pressure.values
            element = self._elements[element_number]
            indexes = element.indexes()
            fl = element.force_vector(values)
            indexes = array(indexes)
            fg[indexes] += fl[:]
       
        # local vector due pressure applied by means of a global function (all over the structure domain)
        if self._global_pressure_function:
            for element in self._elements.values():
                function = self._global_pressure_function
                indexes = element.indexes()
                fl = element.global_pressure_force_vector(function)
                indexes = array(indexes)
                fg[indexes] += fl[:]

        # local vector - forces
        for force in self._forces:
            node_number = force.node_number
            value = force.value
            fg[node_number] += value

        # displacement boundary condition
        self.impose_displacement_boundary_conditions(kg,fg)
       
        # solving the system of equation
        if perturbation:
            sol = solve_perturbation(kg,fg)
        else:
            sol = solve(kg,fg)
        self.update_degrees_of_freedom(sol)


    def impose_displacement_boundary_conditions(self,kg,fg):
        for imp_disp in self._imposed_displacements:
            nn = imp_disp.node_number
            value = imp_disp.value
            fg -= kg[nn,:] * value
            kg[nn,:] = 0.0
            kg[:,nn] = 0.0
            kg[nn,nn] = 1.0
            fg[nn] = value

    def update_degrees_of_freedom(self, sol):   
        for node in self._nodes.values():
            for gdof in node.gdofs:
                index = gdof.index
                gdof.value = sol[index]

    def print_results(self):
        print 'Displacement at nodes'
        print ' Node #     displacement'
        for ni in sorted(self._nodes.keys()):
            node = self._nodes[ni]
            disp = node.displacement
            print '     %d        %.6f'%(ni,disp)
        print '\n'
        print 'Normal force at elements'
        print ' Element #   Normal node i    Normal node j'
        for ei in sorted(self._elements.keys()):
            element = self._elements[ei]
            Ni = element.normal_force(-1.0)
            Nj = element.normal_force(1.0)
            print '    %d          %.5f          %.5f'%(ei,Ni,Nj)

    def plot_results(self, exact_disp = None, exact_N = None, ndiv=10):
        xs = []
        ys = []
        xs_ = []
        disps = []
        Ns = []
        for ind in sorted(self._elements.keys()):
            element = self._elements[ind]
            ni, nj = element.nodes
            xi = ni.coord; di = ni.displacement
            xj = nj.coord; dj = nj.displacement
#           N = element.normal_force()
            xs += [xi, xj]
            ys += [0.0, 0.0]
#           disps += [di, dj]
#           Ns += [N, N]
            for qsi in linspace(-1.0,1.0,ndiv):
                xs_.append(element.PU.x_coord(qsi))
                disps.append(element.displacement(qsi))
                Ns.append(element.normal_force(qsi))

        xmin = min(xs)
        xmax = max(xs)
        dx = xmax - xmin
        if exact_disp and exact_N:
            x = linspace(xmin,xmax,1000)
            ed = exact_disp(x)
            eN = exact_N(x)

        fig = plt.figure(1)
        # Displacement subplot
        ax1 = fig.add_subplot(121)
        ax1.plot(xs_,disps,'bx-',lw=2)
        leg1 = ['Displacement (FEM)']
        if exact_disp and exact_N:
            ax1.plot(x,ed,'g-',lw=2)
            leg1 += ['Exact displacement']
            ymin=min(ys+disps+ed.tolist())
            ymax=max(ys+disps+ed.tolist())
            dy = ymax - ymin
        else:
            ymin=min(ys+disps)
            ymax=max(ys+disps)
            dy = ymax - ymin

        ax1.plot(xs,ys,'ko-',lw=3)
        ax1.set_xlim((xmin-0.1*dx,xmax+0.1*dx))
        ax1.set_ylim((ymin-0.1*dy,ymax+0.4*dy))
        ax1.set_xlabel('x')
        ax1.set_ylabel('Disp.')
        ax1.legend(leg1)

        # Displacement subplot
        ax2 = fig.add_subplot(122)
        ax2.plot(xs_,Ns,'bx-',lw=2)
        leg2 = ['Normal force (FEM)']
        if exact_disp and exact_N:
            ax2.plot(x,eN,'g-',lw=2)
            leg2 += ['Exact Normal Force']
            ymin=min(ys+Ns+eN.tolist())
            ymax=max(ys+Ns+eN.tolist())
            dy = ymax - ymin
        else:
            ymin=min(ys+Ns)
            ymax=max(ys+Ns)
            dy = ymax - ymin

        ax2.plot(xs,ys,'ko-',lw=3)
        ax2.set_xlim((xmin-0.1*dx,xmax+0.1*dx))
        ax2.set_ylim((ymin-0.1*dy,ymax+0.4*dy))
        ax2.set_xlabel('x')
        ax2.set_ylabel('Normal Force')
        ax2.legend(leg2)

        plt.show()

    def evaluate_error(self, exact_disp, exact_N):
        des = 0.0
        Nes = 0.0
        print '\nElement     Disp. error     Normal force error'
        for ei in sorted(self._elements.keys()):
            element = self._elements[ei]
            de = element.displacement_error(exact_disp)
            Ne = element.normal_error(exact_N)
            des += de ** 2
            Nes += Ne ** 2
            print '  %d            %e                 %e'%(ei,de,Ne)
        print '\nGlobal displacement error = %e'%sqrt(des)
        print 'Global normal force error = %e'%sqrt(Nes)


def solve_perturbation(kg, fg, tol=1.0e-8, maxit=100):
    '''
    Solves system of equation using the perturbation method proposed by prof. Ivo Babuska.
    See (Strouboulis, Babuska and Copps, 2000).
    '''
    T = diag(1.0/npsqrt(diag(kg))) 
    A = dot(T,dot(kg,T)) 
    A_ = A + eye(len(kg)) * tol
    b = dot(T,fg)
    sol_ = solve(A_,b)
    error = 1.0 / tol 
    it  = 0
    while error > tol:
        res_ = b - dot(A,sol_)
        dsol = solve(A_,res_)
        sol_ += dsol
        sol = dot(T,sol_)
        res = fg - dot(kg,sol)
        error = npsqrt(dot(res,res))
        print ' -> Iteration %d: error norm =%.8e'%(it,error)
        it += 1
        if it > maxit:
            print 'Perturbation method did not converged in %d iterations!'%maxit
            exit()
    print 'Perturbation method converged'
    return sol


class Partition_of_unity(object):

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes

    def phi(self, function_number, qsi):
        if function_number == 0 :
            value = 0.5 - 0.5 * qsi
        elif function_number == 1:
            value = 0.5 + 0.5 * qsi
        return value

    def dphi_dqsi(self, function_number, qsi):
        if function_number == 0:
            value = -0.5
        elif function_number == 1:
            value = 0.5
        return value

    def x_coord(self, qsi):
        n0, n1 = self._nodes
        x0 = n0.coord
        x1 = n1.coord
        phi0 = self.phi(0,qsi)
        phi1 = self.phi(1,qsi)
        value = x0 * phi0 + x1 * phi1
        return value

    def dx_dqsi(self, qsi):
        n0, n1 = self._nodes
        x0 = n0.coord
        x1 = n1.coord
        dphi0_dqsi = -0.5
        dphi1_dqsi = 0.5
        value = x0 * dphi0_dqsi + x1 * dphi1_dqsi
        return value

class Generalized_dof(object):

    def __init__(self, index = None, value = 0.0, enrichment = None):
        self._index = index
        self._value = value
        self._enrichment = enrichment
    
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, ind):
        self._index = ind

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def enrichment(self):
        return self._enrichment

    @enrichment.setter
    def enrichment(self, enr):
        self._enrichment = enr

class Enrichment(object):
    
    def __init__(self, node):
        self._node = node

    @property
    def node(self):
        return self._node

class One(Enrichment):

    def __init__(self, node):
        super(One, self).__init__(node)

    def Le(self, x):
        return 1.0

    def dLe_dx(self, x):
        return 0.0

class Poly(Enrichment):
    
    def __init__(self, node, exponent):
        super(Poly, self).__init__(node)
        self._exponent = exponent
        
    def Le(self, x):
        n = self._exponent
        return x ** n
    
    def dLe_dx(self, x):
        n = self._exponent
        return n * x ** (n - 1)
        
class Shifted(Enrichment):
    
    def __init__(self, node, exponent):
        super(Shifted, self).__init__(node)
        self._exponent = exponent
        
    def Le(self, x):
        n = self._exponent
        xn = self._node.coord
        return (x - xn)** n
    
    def dLe_dx(self, x):
        n = self._exponent
        xn = self._node.coord
        return n * (x - xn)** (n - 1)        
    
class Sine(Enrichment):

    def __init__(self, node, L):
        super(Sine, self).__init__(node)
        self._L = L

    def Le(self, x):
        L = self._L
        return sin(pi*x/L)

    def dLe_dx(self, x):
        L = self._L
        return pi * cos(pi*x/L) / L
    
