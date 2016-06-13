# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:53:58 2016

@author: herman
"""

# Python program to generate input files for Calculix
    # calls own written function hex_mesher which generates a mesh for calculix and AEM_basic

import numpy as np
from hex_mesher import hex_mesh
import os
import sys
import time

# Input deck for mesh generator
#input required:
    #2D or 3D                                   =   option      integer
    #element size: l,w,h                        =   size        array   [x,y,z]
    #rows, columns, stacks                      =   grid        array   [x,y,z]
    #gap direction; in x direction y or both    =   gap_drct    string
    #gap, note only in x and y                  =   gap         array   [x,y]
    #linear or quadratic                        =   quad        boolean
    #displaying mesh                            =   dispM       boolean


#output given
    #x,y and z co-ordinates         =   nodes       matrix  
    #element number and nodes       =   element     matrix
        
        #not yet included
    #element center co-ordinates    =   el_center   matrix  

option = 3
size = np.array([500,500,700])   #units is mm, if using meters Material properties need to be adjusted
grid = np.array([3,3,1])
gap = np.array([10,10])
quad = False
dispM = True
gap_drct = 'none'
node, element = hex_mesh(option, size, grid, gap, gap_drct, quad, dispM)

#-----------------------------------------------------------------------------#
# Input deck for additional information
# Material Properties:
E = 210e3       #Young's Modulus
nu = 0.3        #Poisson ratio

#-----------------------------------------------------------------------------#
# Specifying variables to be used in writing file
# Element types
if option == 2:
    if quad == True:
        #note plane stress elements for plane strain CPE8 and CPE4
        ele_type = 'CPS8'
    else:
        ele_type = 'CPS4'
else:
    if quad == True:
        ele_type = 'C3D20'
    else:
        ele_type = 'C3D8'
# Defining node sets
x_array = node[:,1]
y_array = node[:,2]
z_array = node[:,3]

cubes_xy_plane = node[z_array==0,0]
cubes_yz_plane = node[x_array==0,0]
cubes_xz_plane = node[y_array==0,0]

cubes_x_max = node[x_array==max(x_array),0]
cubes_y_max = node[y_array==max(y_array),0]
cubes_z_max = node[z_array==max(z_array),0]
#-----------------------------------------------------------------------------#
# Start of .inp file writing
analysis = 'disp'
f = open('005_%s_x%iy%iz%i_gap_%s.inp' % (analysis, grid[0], grid[1], grid[2], gap_drct) , 'wb')
# Writing opening comments
f.write("**Calculix input file generated by Calculix_input.py \n")
f.write("**Created for investigation 005_Calculix_grid with mesh of %i %i %i \n" % (grid[0], grid[1], grid[2]))
f.write("**element sizes are l=%i t=%i and h=%i  and gaps between elements in direction %s\n" % (size[0], size[1], size[2], gap_drct))
f.write("**gap sizes are: \n**if gap direction xy then x = %2.2f and y = %2.2f \n" % (gap[0], gap[1])) 
f.write("**if gap direction x then x = %2.2f and y = 0 \n**if gap direction y then x = 0 and y = %2.2f\n" % (gap[0], gap[1]))
f.write("\n")
# Writing node number and co-ordinates
f.write("*NODE, NSET=Nall\n")
for dolphin in range(0,len(node),1):
    f.writelines("\t%i, %4.4f, %4.4f, %4.4f\n" % (node[dolphin,0], node[dolphin,1], node[dolphin,2], node[dolphin,3]))
f.write("*ELEMENT, TYPE=%s, ELSET=Eall\n" % ele_type)
# Writing element number and nodes
if option == 2:
    if quad == True:
        for zebra in range(0,len(element),1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i\n" % (element[zebra,0], element[zebra,1], element[zebra,2], element[zebra,3],
                                                                                             element[zebra,4], element[zebra,5], element[zebra,6], element[zebra,7],
                                                                                             element[zebra,8]))
    else:
        for zebra in range(0,len(element),1):
            f.writelines("\t%i, %i, %i, %i, %i\n" % (element[zebra,0], element[zebra,1], element[zebra,2], element[zebra,3],
                                                                 element[zebra,4]))
else:
    if quad == True:
        for zebra in range(0,len(element),1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i, %i,\n%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % 
                                                                    (element[zebra,0], element[zebra,1], element[zebra,2], element[zebra,3],
                                                                     element[zebra,4], element[zebra,5], element[zebra,6], element[zebra,7],
                                                                     element[zebra,8], element[zebra,9], element[zebra,10], element[zebra,11],
                                                                     element[zebra,12], element[zebra,13], element[zebra,14], element[zebra,15],
                                                                     element[zebra,16], element[zebra,17], element[zebra,18], element[zebra,19],
                                                                     element[zebra,20]))
    else:
        for zebra in range(0,len(element),1):
            f.writelines("\t%i, %i, %i, %i, %i, %i, %i, %i, %i\n" % (element[zebra,0], element[zebra,1], element[zebra,2], element[zebra,3],
                                                                                             element[zebra,4], element[zebra,5], element[zebra,6], element[zebra,7],
                                                                                             element[zebra,8]))        
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
 
# Writing Boundary conditions
f.write("*BOUNDARY\n")
# The grid is constrained to not move in the negative x,y and z directions
f.write("cubes_xy_plane,3,3\n")
f.write("cubes_xz_plane,2,2\n")
f.write("cubes_yz_plane,1,1\n")
f.write("\n")    

# Assigning material properties
f.write("*MATERIAL,NAME=EL\n")
f.write("*ELASTIC\n")
f.write("%8.3f, %8.4f\n\n" % (E, nu))

# Assigning Material Properties to Elements
f.write("*SOLID SECTION,ELSET=Eall,MATERIAL=EL\n\n")

# Specify contact properties

# Allocating the analysis properties: type, step size, and loadings
f.write("*STEP, INC=100, NLGEOM\n")
f.write("*STATIC, DIRECT\n")           #static analysis without automatic incrementation
f.write("0.05, 1.\n")
f.write("*BOUNDARY\n")
f.write("cubes_x_max,1,1,-200\n")
f.write("\n")
# Writing output files
f.write("**Specify output \n")
f.write("*NODE PRINT, NSET=Nall, FREQUENCY=1\nU\n")
f.write("*EL PRINT, ELSET=Eall, FREQUENCY=1\nS,E\n")
f.write("*NODE FILE, FREQUENCY=1\nU\n")
f.write("*EL FILE, FREQUENCY=1\nS,E\n")



f.close()

