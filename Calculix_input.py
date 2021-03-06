
"""
Created on Thu May 26 12:53:58 2016

@author: herman
"""

# Python program to generate input files for Calculix
    # calls own written function hex_mesher which generates a
    # mesh for calculix and AEM_basic
#! Need to complete for 2d input file generation

import numpy as np
from hex_mesher import hex_mesh
#import os
#import sys
#import time

# Input deck for mesh generator
# input required:
    #2D or 3D                                   =   option      integer
    #element size: l,w,h                        =   size        array   [x,y,z]
    #rows, columns, stacks                      =   grid        array   [x,y,z]
    #gap direction; in x direction y or both    =   gap_drct    string
    #gap, note only in x and y                  =   gap         array   [x,y]
    #linear or quadratic                        =   quad        boolean
    #displaying mesh                            =   dispM       boolean


# output given
    #x,y and z co-ordinates         =   nodes       matrix
    #element number and nodes       =   element     matrix

        #not yet included in mesh generator
    #element center co-ordinates    =   el_center   matrix

option = 3
size = np.array([500, 500, 700])   # units in mm, adjust Material prop for m
grid = np.array([5, 5, 1])
gap = np.array([250, 250])
quad = False
dispM = True
gap_drct = 'xy'
node, element = hex_mesh(option, size, grid, gap, gap_drct, quad, dispM)

#-----------------------------------------------------------------------------#
# Input deck for additional information
# Material Properties:
E = 210e3           # Young's Modulus
nu = 0.3            # Poisson ratio
density = 7800      # Density
conduc = 1000       # Thermal Conductivity
expans = 25e-6      # Coeficient of expansion
spec_heat = 5000    # Spcific Heat of Material
# Physical constants:
abs_zero = 0
stef_boltz = 5.669e-8
# Initial temperature for all nodes
ini_temp = 0
# Analysis info
analysis = 'disp'   # specifying what is prescribed

# add things like steady state, number of increments,
# automatic incrementation on or off etc

# Contact Specifications
contact = True
contact_d = False
contact_option = 1
contact_type = ["LINEAR", "EXPONENTIAL", "TIED"]
slope_K = 1e7
c0 = 0.01
p0 = 13
Friction = False
mu = 0.2        # Usually between 0.1 and 0.5
lmbda = 5000    # Usually tn times smaler than spring constant

# Stabilising Springs
s_springs = False
spring_linear = True
spring_const = 1       # can extend to force, elongation and temperature
# add for non linear spring properties

#-----------------------------------------------------------------------------#
# Specifying variables to be used in writing file
# Element types
if option == 2:
    if quad is True:
        #note plane stress elements for plane strain CPE8 and CPE4
        ele_type = 'CPS8'
        dimention = '2D'
        order = 'quad'
    else:
        ele_type = 'CPS4'
        dimention = '2D'
        order = 'linear'
    # Defining node sets
    x_array = node[:, 1]
    y_array = node[:, 2]
else:
    if quad is True:
        ele_type = 'C3D20'
        dimention = '3D'
        order = 'quad'
    else:
        ele_type = 'C3D8'
        dimention = '3D'
        order = 'linear'
    # Defining node sets
    x_array = node[:, 1]
    y_array = node[:, 2]
    z_array = node[:, 3]

    cubes_xy_plane = node[z_array == 0, 0]
    cubes_yz_plane = node[x_array == 0, 0]
    cubes_xz_plane = node[y_array == 0, 0]

    cubes_x_max = node[x_array == max(x_array), 0]
    cubes_y_max = node[y_array == max(y_array), 0]
    cubes_z_max = node[z_array == max(z_array), 0]
#-----------------------------------------------------------------------------#
# Start of .inp file writing
if s_springs is True:
    f = open('005_%s_%s_%s_x%iy%iz%i_gap_%s_ss.inp' % (analysis, dimention,
                                                       order, grid[0],
                                                       grid[1], grid[2],
                                                       gap_drct), 'wb')
else:
    f = open('005_%s_%s_%s_x%iy%iz%i_gap_%s.inp' % (analysis, dimention, order,
                                                    grid[0], grid[1], grid[2],
                                                    gap_drct), 'wb')
# Writing opening comments
f.write("**Calculix input file generated by Calculix_input.py \n")
f.write("**Created for investigation 005_Calculix_grid with mesh "
        "of %i %i %i \n" % (grid[0], grid[1], grid[2]))
f.write("**element sizes are l=%i t=%i and h=%i  and gaps between elements "
        "in direction %s\n" % (size[0], size[1], size[2], gap_drct))
f.write("**gap sizes are: \n**if gap direction xy then x = %2.2f and "
        "y = %2.2f \n" % (gap[0], gap[1]))
f.write("**if gap direction x then x = %2.2f and y = 0 \n**if gap direction "
        "y then x = 0 and y = %2.2f\n" % (gap[0], gap[1]))
f.write("\n")
# Writing node number and co-ordinates
f.write("*NODE, NSET=Nall\n")
for dolphin in range(0, len(node), 1):
    f.writelines("\t%i, %4.4f, %4.4f, %4.4f\n" % (node[dolphin, 0],
                                                  node[dolphin, 1],
                                                  node[dolphin, 2],
                                                  node[dolphin, 3]))
f.write("*ELEMENT, TYPE=%s, ELSET=Eall\n" % ele_type)
# Writing element number and nodes
if option == 2:
    if quad is True:
        for zebra in range(0, len(element), 1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i\n" %
                        (element[zebra, 0], element[zebra, 1],
                         element[zebra, 2], element[zebra, 3],
                         element[zebra, 4], element[zebra, 5],
                         element[zebra, 6], element[zebra, 7],
                         element[zebra, 8]))
    else:
        for zebra in range(0, len(element), 1):
            f.writelines("\t%i, %i, %i, %i, %i\n" % (element[zebra, 0],
                                                     element[zebra, 1],
                                                     element[zebra, 2],
                                                     element[zebra, 3],
                                                     element[zebra, 4]))
else:
    if quad is True:
        for zebra in range(0, len(element), 1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i, %i,\n%"
                         "i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" %
                         (element[zebra, 0], element[zebra, 1],
                          element[zebra, 2], element[zebra, 3],
                          element[zebra, 4], element[zebra, 5],
                          element[zebra, 6], element[zebra, 7],
                          element[zebra, 8], element[zebra, 9],
                          element[zebra, 10], element[zebra, 11],
                          element[zebra, 12], element[zebra, 13],
                          element[zebra, 14], element[zebra, 15],
                          element[zebra, 16], element[zebra, 17],
                          element[zebra, 18], element[zebra, 19],
                          element[zebra, 20]))
    else:
        for zebra in range(0, len(element), 1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i\n" %
                        (element[zebra, 0], element[zebra, 1],
                         element[zebra, 2], element[zebra, 3],
                         element[zebra, 4], element[zebra, 5],
                         element[zebra, 6], element[zebra, 7],
                         element[zebra, 8]))
# Writing spring elements if stabilising springs are activated


def spring_x_gap(f, grid, element, spring_num):
    ele_num = 0
    for calf in range(1, grid[1]+1):
        for lamb in range(1, grid[0]):
            f.write("\t%i, %i, %i\n" % (spring_num,
                                        element[ele_num, 2],
                                        element[ele_num + 1, 1]))
            f.write("\t%i, %i, %i\n" % (spring_num + 1,
                                        element[ele_num, 3],
                                        element[ele_num + 1, 4]))
            f.write("\t%i, %i, %i\n" % (spring_num + 2,
                                        element[ele_num, 6],
                                        element[ele_num + 1, 5]))
            f.write("\t%i, %i, %i\n" % (spring_num + 3,
                                        element[ele_num, 7],
                                        element[ele_num + 1, 8]))
            spring_num += 4
            ele_num += 1
        ele_num += 1
    return spring_num


def spring_y_gap(f, grid, element, spring_num):
    ele_num = 0
    for calf in range(1, grid[1]+1):
        for lamb in range(1, grid[0]):
            f.write("\t%i, %i, %i\n" % (spring_num,
                                        element[ele_num, 4],
                                        element[ele_num + grid[0], 1]))
            f.write("\t%i, %i, %i\n" % (spring_num + 1,
                                        element[ele_num, 3],
                                        element[ele_num + grid[0], 2]))
            f.write("\t%i, %i, %i\n" % (spring_num + 2,
                                        element[ele_num, 7],
                                        element[ele_num + grid[0], 6]))
            f.write("\t%i, %i, %i\n" % (spring_num + 3,
                                        element[ele_num, 8],
                                        element[ele_num + grid[0], 5]))
            spring_num += 4
            ele_num += 1
    return spring_num

if (s_springs is True) and (option == 3):
    f.write("*ELEMENT,TYPE=SPRINGA,ELSET=S_SPRINGS\n")
    if gap_drct == 'x':
        spring_num = element[-1, 0] + 1
        spring_num = spring_x_gap(f, grid, element, spring_num)
    elif gap_drct == 'y':
        spring_num = element[-1, 0] + 1
        spring_num = spring_y_gap(f, grid, element, spring_num)
    else:
        spring_num = element[-1, 0] + 1
        spring_num = spring_x_gap(f, grid, element, spring_num)
        spring_num = spring_y_gap(f, grid, element, spring_num)

# Writing node sets
f.write("\n")
f.write("*NSET,NSET=cubes_xy_plane\n")
for nodes_xy_plane in cubes_xy_plane:
    f.write("%i,\n" % nodes_xy_plane)
f.write("\n")

f.write("*NSET,NSET=cubes_xz_plane\n")
for nodes_xz_plane in cubes_xz_plane:
    f.write("%i,\n" % nodes_xz_plane)
f.write("\n")

f.write("*NSET,NSET=cubes_yz_plane\n")
for nodes_yz_plane in cubes_yz_plane:
    f.write("%i,\n" % nodes_yz_plane)
f.write("\n")

f.write("*NSET,NSET=cubes_x_max\n")
for nodes_x_max in cubes_x_max:
    f.write("%i,\n" % nodes_x_max)
f.write("\n")

f.write("*NSET,NSET=cubes_y_max\n")
for nodes_y_max in cubes_y_max:
    f.write("%i,\n" % nodes_y_max)
f.write("\n")

f.write("*NSET,NSET=cubes_z_max\n")
for nodes_z_max in cubes_z_max:
    f.write("%i,\n" % nodes_z_max)
f.write("\n")

f.write("*NSET,NSET=cube9\n")
for cube9_nodes in element[-1, 1:9]:
    f.write("%i,\n" % cube9_nodes)
f.write("\n")

# Writing Boundary conditions
f.write("*BOUNDARY\n")
# The grid is constrained to not move in the negative x,y and z directions
f.write("cubes_xy_plane,3,3\n")
f.write("cubes_xz_plane,2,2\n")
f.write("cubes_yz_plane,1,1\n")
f.write("\n")

# Physical Constants
f.write("*PHYSICAL CONSTANTS, ABSOLUTE ZERO=%6.4f, STEFAN BOLTZMANN=%.2E\n"
        % (abs_zero, stef_boltz))
f.write("\n")
# Initial conditions
f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
f.write("Nall,%8.4f\n" % ini_temp)
f.write("\n")
# Assigning material properties
f.write("*MATERIAL,NAME=EL\n")
f.write("*ELASTIC\n")
f.write("%8.3f, %8.4f\n" % (E, nu))
f.write("*DENSITY\n")
f.write("%8.3f\n" % density)
f.write("*CONDUCTIVITY\n")
f.write("%8.3f\n" % conduc)
f.write("*EXPANSION\n")
f.write("%.2E\n" % expans)
f.write("*SPECIFIC HEAT\n")
f.write("%8.3f\n\n" % spec_heat)

# Assigning Material Properties to Elements
f.write("*SOLID SECTION,ELSET=Eall,MATERIAL=EL\n\n")

if (s_springs is True) and (option == 3):
    f.write("*SPRING,ELSET=S_SPRINGS\n %1.2E\n\n" % spring_const)

# Specify contact properties
    # It's importnat to note that contact is specified in such a manner that
    # each block, consisting of one element can only be in contact with its
    # horizontal and vertical (x and y) neigbours and not diagonaly
# for x direction gaps


def write_x_gap(f, grid):
    gap_count = 1
    for calf in range(1, grid[1]+1, 1):
        for lamb in range(1, grid[0], 1):
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+1))
            gap_count += 1
        gap_count += 1
# for y direction gaps


def write_y_gap(f, grid):
    gap_count = 1
    for calf in range(1, grid[1], 1):
        for lamb in range(1, grid[0]+1, 1):
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]))
            gap_count += 1


def write_x_gapd(f, grid):
    gap_count = 1
    for lamb in range(1, grid[0], 1):
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+1))
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+grid[0]+1))
        gap_count += 1
    for calf in range(1, grid[1]-1, 1):
        gap_count += 1
        for lamb in range(1, grid[0], 1):
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count-grid[0]+1))
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+1))
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+grid[0]+1))
            gap_count += 1
    gap_count += 1
    for lamb in range(1, grid[0], 1):
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count-grid[0]+1))
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_x_p,el%i_x_n\n" % (gap_count, gap_count+1))
        gap_count += 1
# for y direction gaps


def write_y_gapd(f, grid):
    gap_count = 1
    for calf in range(1, grid[1], 1):
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]))
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]+1))
        gap_count += 1
        for lamb in range(1, grid[0]-1, 1):
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]-1))
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]))
            f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
            f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]+1))
            gap_count += 1
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]-1))
        f.write("*CONTACT PAIR,INTERACTION=SI1,TYPE=SURFACE TO SURFACE\n")
        f.write("el%i_y_p,el%i_y_n\n" % (gap_count, gap_count+grid[0]))
        gap_count += 1


if (contact is True) and (option == 3):
    for sheep in element[:, 0]:
        f.write("*SURFACE,NAME=el%i_x_p\n" % sheep)
        f.write("%i,S4\n" % sheep)
        f.write("*SURFACE,NAME=el%i_x_n\n" % sheep)
        f.write("%i,S6\n" % sheep)
        f.write("*SURFACE,NAME=el%i_y_p\n" % sheep)
        f.write("%i,S5\n" % sheep)
        f.write("*SURFACE,NAME=el%i_y_n\n" % sheep)
        f.write("%i,S3\n" % sheep)

    if (gap_drct == 'x') and (contact_d is False):
        write_x_gap(f, grid)
    elif (gap_drct == 'y') and (contact_d is False):
        write_y_gap(f, grid)
    elif (gap_drct == 'xy') and (contact_d is False):
        write_x_gap(f, grid)
        write_y_gap(f, grid)
    elif (gap_drct == 'x') and (contact_d is True):
        write_x_gapd(f, grid)
    elif (gap_drct == 'y') and (contact_d is True):
        write_y_gapd(f, grid)
    elif (gap_drct == 'xy') and (contact_d is True):
        write_x_gapd(f, grid)
        write_y_gapd(f, grid)

    f.write("*SURFACE INTERACTION,NAME=SI1\n")
    f.write("*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=%s\n"
            % contact_type[contact_option-1])
    if (contact_option == 1) or (contact_option == 3):
        f.write("%.2E\n" % slope_K)
    else:
        f.write("%.2E,%.2E\n" % (c0, p0))

    f.write("\n")

# Add Amplitudes if applicable

# Allocating the analysis properties: type, step size, and loadings
f.write("*STEP, INC=10000, NLGEOM\n")
f.write("*STATIC\n")  # static analysis with manual incrementation
f.write("0.001, 1., 0.001, 0.009\n")
f.write("*BOUNDARY\n")
f.write("cubes_x_max,1,1,-30\n")
f.write("cubes_y_max,2,2,-30\n")
#f.write("cube9,2,2,-30\n")
f.write("\n")
# Writing output files
f.write("**Specify output \n")
f.write("*NODE PRINT, NSET=Nall, FREQUENCY=1\nU\n")
f.write("*EL PRINT, ELSET=Eall, FREQUENCY=1\nS,E\n")
f.write("*NODE FILE, FREQUENCY=1\nU\n")
f.write("*EL FILE, FREQUENCY=1\nS,E\n")
f.write("*END STEP")

f.close()
