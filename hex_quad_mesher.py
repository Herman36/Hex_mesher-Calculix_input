# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:50:08 2016

@author: herman
"""

#function hex_mesh_quad to be called by hex_mesh

#input required:
    #2D or 3D                           =   option      integer
    #element size: l, w, h              =   size        array   [x, y, z]
    #rows, columns, stacks              =   grid        array   [x, y, z]
    #gap direction                      =   gap_drct    string
    #gap, note only in x and y          =   gap         array   [x, y]


#output given
    #x, y and z co-ordinates            =   nodes      matrix
    #element number and nodes           =   element     matrix
    #element center co-ordinates        =   el_center   matrix

import numpy as np


def hex_mesh_quad(option, size, grid, gap, gap_drct, dispM):
    l = grid[0]
    t = grid[1]
    h = grid[2]
    if option == 2:         # 2 dimentional mesh
        if gap_drct == 'none':    # no gap between elements
            # defining nodes
            ele_num = l*t
            cr_num = (l+1)*(t+1)
            hzn_num = l*(t+1)
            node_num = (2*l+1)*(2*t+1) - ele_num
            nodes = np.zeros(shape=(node_num, 3))
            rhino = 0
            # Corner nodes of elements
            for y in range(0, (t+1)*size[1], size[1]):
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = y
                    rhino += 1
            # Midside nodes on horizontal element sides
            for y in range(0, (t+1)*size[1], size[1]):
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x
                    nodes[rhino, 2] = y
                    rhino += 1
            # Midside nodes on vertical element sides
            for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = q_y
                    rhino += 1
            # defining elements
            ele_num = l*t
            element = np.zeros(shape=(ele_num, 9))
            zebra = 0
            for m in range(0, t, 1):
                for n in range(0, l, 1):
                    element[zebra, 0] = zebra+1
                    element[zebra, 1] = zebra + m + 1
                    element[zebra, 2] = zebra + m + 2
                    element[zebra, 3] = zebra + m + l + 3
                    element[zebra, 4] = zebra + m + l + 2
                    element[zebra, 5] = zebra + cr_num + 1
                    element[zebra, 7] = zebra + cr_num + l + 1
                    element[zebra, 6] = zebra + cr_num + hzn_num + m + 2
                    element[zebra, 8] = zebra + cr_num + hzn_num + m + 1
                    zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'x':   # gap in x direction only
            # defining nodes
            ele_num = l*t
            cr_num = (2*l)*(t+1)
            hzn_num = l*(t+1)
            node_num = (3*l)*(2*t+1) - ele_num
            nodes = np.zeros(shape=(node_num, 3))
            rhino = 0
            # Corner nodes of elements
            for y in range(0, (t+1)*size[1], size[1]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = y
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = y
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = y
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = y
                rhino += 1
            # Midside nodes on horizontal element sides
            for y in range(0, (t+1)*size[1], size[1]):
                dolphin = 0
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x + dolphin*gap[0]
                    nodes[rhino, 2] = y
                    rhino += 1
                    dolphin += 1
            # Midside nodes on vertical element sides
            for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = q_y
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = q_y
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = q_y
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = q_y
                rhino += 1
            # defining elements
            ele_num = l*t
            element = np.zeros(shape=(ele_num, 9))
            zebra = 0
            for m in range(0, t, 1):
                for n in range(0, l, 1):
                    element[zebra, 0] = zebra+1
                    element[zebra, 1] = 2*zebra + 1
                    element[zebra, 2] = 2*zebra + 2
                    element[zebra, 3] = 2*zebra + 2*l + 2
                    element[zebra, 4] = 2*zebra + 2*l + 1
                    element[zebra, 5] = zebra + cr_num + 1
                    element[zebra, 7] = zebra + cr_num + l + 1
                    element[zebra, 6] = 2*zebra + cr_num + hzn_num + 2
                    element[zebra, 8] = 2*zebra + cr_num + hzn_num + 1
                    zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'y':   # gap in y direction only
            # defining nodes
            ele_num = l*t
            cr_num = (l+1)*(2*t)
            hzn_num = 2*l*t
            node_num = (2*l+1)*(3*t) - ele_num
            nodes = np.zeros(shape=(node_num, 3))
            rhino = 0
            # Corner nodes of elements
            for x in range(0, (l+1)*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = x
                nodes[rhino, 2] = 0
                rhino += 1
            for y in range(1, t, 1):
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    rhino += 1
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    rhino += 1
            for x in range(0, (l+1)*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = x
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                rhino += 1
            # Midside nodes on horizontal element sides
            for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = q_x
                nodes[rhino, 2] = 0
                rhino += 1
            for y in range(1, t, 1):
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    rhino += 1
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    rhino += 1
            for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = q_x
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                rhino += 1
            # Midside nodes on vertical element sides
            dolphin = 0
            for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = q_y + dolphin*gap[1]
                    rhino += 1
                dolphin += 1
            # defining elements
            ele_num = l*t
            element = np.zeros(shape=(ele_num, 9))
            zebra = 0
            for m in range(0, t, 1):
                for n in range(0, l, 1):
                    element[zebra, 0] = zebra+1
                    element[zebra, 1] = zebra + m*(2+l) + 1
                    element[zebra, 2] = zebra + m*(2+l) + 2
                    element[zebra, 3] = zebra + m*(2+l) + l + 3
                    element[zebra, 4] = zebra + m*(2+l) + l + 2
                    element[zebra, 5] = zebra + m*l + cr_num + 1
                    element[zebra, 7] = zebra + m*l + cr_num + l + 1
                    element[zebra, 6] = zebra + m + cr_num + hzn_num + 2
                    element[zebra, 8] = zebra + m + cr_num + hzn_num + 1
                    zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'xy':      # gap in both directions
            # defining nodes
            ele_num = l*t
            cr_num = 4*l*t
            hzn_num = 2*l*t
            node_num = (2*l+1)*(3*t) - ele_num
            node_num = (3*l)*(3*t) - l*t
            nodes = np.zeros(shape=(node_num, 3))
            rhino = 0
            # Corner nodes of elements
            nodes[rhino, 0] = rhino+1
            nodes[rhino, 1] = 0
            nodes[rhino, 2] = 0
            rhino += 1
            for x in range(1, l, 1):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                nodes[rhino, 2] = 0
                nodes[rhino+1, 0] = rhino+2
                nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                nodes[rhino+1, 2] = 0
                rhino += 2
            nodes[rhino, 0] = rhino+1
            nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
            nodes[rhino, 2] = 0
            rhino += 1
            for y in range(1, t, 1):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = y*size[1]+(y-1)*gap[1]
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                rhino += 1
                #---------------------------------------#
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = y*size[1]+y*gap[1]
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = y*size[1]+y*gap[1]
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = y*size[1]+y*gap[1]
                rhino += 1
            #-----------------------------------------#
            nodes[rhino, 0] = rhino+1
            nodes[rhino, 1] = 0
            nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
            rhino += 1
            for x in range(1, l, 1):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                nodes[rhino+1, 0] = rhino+2
                nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                nodes[rhino+1, 2] = (y+1)*size[1]+y*gap[1]
                rhino += 2
            nodes[rhino, 0] = rhino+1
            nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
            nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
            rhino += 1
            # Midside nodes on horizontal element sides
            dolphin = 0
            for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = q_x + dolphin*gap[0]
                nodes[rhino, 2] = 0
                rhino += 1
                dolphin += 1
            for y in range(1, t, 1):
                dolphin = 0
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x + dolphin*gap[0]
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    rhino += 1
                    dolphin += 1
                dolphin = 0
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x + dolphin*gap[0]
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    rhino += 1
                    dolphin += 1
            dolphin = 0
            for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = q_x + dolphin*gap[0]
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                rhino += 1
                dolphin += 1
            # Midside nodes on vertical element sides
            seal = 0
            for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = q_y + seal*gap[1]
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = q_y + seal*gap[1]
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = q_y + seal*gap[1]
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = q_y + seal*gap[1]
                rhino += 1
                seal += 1
            # defining elements
            ele_num = l*t
            element = np.zeros(shape=(ele_num, 9))
            zebra = 0
            for m in range(0, t, 1):
                for n in range(0, l, 1):
                    element[zebra, 0] = zebra+1
                    element[zebra, 1] = 2*zebra + 2*m*l + 1
                    element[zebra, 2] = 2*zebra + 2*m*l + 2
                    element[zebra, 3] = 2*zebra + 2*m*l + 2*l + 2
                    element[zebra, 4] = 2*zebra + 2*m*l + 2*l + 1
                    element[zebra, 5] = zebra + m*l + cr_num + 1
                    element[zebra, 7] = zebra + m*l + cr_num + l + 1
                    element[zebra, 6] = 2*zebra + cr_num + hzn_num + 2
                    element[zebra, 8] = 2*zebra + cr_num + hzn_num + 1
                    zebra += 1
           #end of gap
        #-----------------------------------------------------------------#
        #Plotting of nodes
        if dispM is True:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(nodes[:, 1], nodes[:, 2], 'b*')
            tiger = 0
            for xy in zip(nodes[:, 1], nodes[:, 2]):
                tiger += 1
                ax.annotate('[%s]' % tiger, xy=xy, textcoords='data')
            plt.grid()
            plt.show()
        #end of option 2
    #-------------------------------------------------------------------------#
    elif option == 3:         # 3 dimentional mesh
        if gap_drct == 'none':      # no gap between elements
            # defining nodes
            base_ele_num = l*t
            base_node_num = (2*l+1)*(2*t+1) - base_ele_num
            base_cr_num = (l+1)*(t+1)
            base_hzn_num = l*(t+1)
            node_num = base_node_num*(h+1) + h*base_cr_num
            nodes = np.zeros(shape=(node_num, 4))
            rhino = 0
            # Corner nodes of elements
            for z in range(0, (h+1)*size[2], size[2]):
                for y in range(0, (t+1)*size[1], size[1]):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = z
                        rhino += 1
                # Midside nodes on horizontal element sides
                for y in range(0, (t+1)*size[1], size[1]):
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = z
                        rhino += 1
                # Midside nodes on vertical element sides
                for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = q_y
                        nodes[rhino, 3] = z
                        rhino += 1
            # Midside nodes for cube z direction
            for q_z in np.arange(float(size[2])/2, (h)*size[2], size[2]):
                for y in range(0, (t+1)*size[1], size[1]):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = q_z
                        rhino += 1
            # defining elements
            ele_num = l*t*h
            element = np.zeros(shape=(ele_num, 21))
            (zebra = 0
            for k in range(0, h, 1):
                for m in range(0, t, 1):
                    for n in range(0, l, 1):
                        element[zebra, 0] = zebra+1
                        element[zebra, 1] = (zebra + m + k*base_node_num -
                                             k*base_ele_num + 1)
                        element[zebra, 2] = (zebra + m + k*base_node_num -
                                             k*base_ele_num + 2)
                        element[zebra, 3] = (zebra + m + k*base_node_num - k*base_ele_num + l + 3)
                        element[zebra, 4] = (zebra + m + k*base_node_num - k*base_ele_num + l + 2)
                        element[zebra, 9] = (zebra + base_cr_num + k*(base_node_num - base_ele_num) + 1)
                        element[zebra, 11] = (zebra + base_cr_num + k*(base_node_num - base_ele_num) + l + 1)
                        element[zebra, 10] = (zebra + base_cr_num + base_hzn_num + m + k*(base_node_num - base_ele_num) + 2)
                        element[zebra, 12] = (zebra + base_cr_num + base_hzn_num + m + k*(base_node_num - base_ele_num) + 1)
                        element[zebra, 5] = (zebra + m + (k+1)*base_node_num - k*base_ele_num + 1)
                        element[zebra, 6] = (zebra + m + (k+1)*base_node_num - k*base_ele_num + 2)
                        element[zebra, 7] = (zebra + m + (k+1)*base_node_num - k*base_ele_num + l + 3)
                        element[zebra, 8] = (zebra + m + (k+1)*base_node_num - k*base_ele_num + l + 2)
                        element[zebra, 13] = (zebra + base_cr_num + (k+1)*base_node_num - k*base_ele_num + 1)
                        element[zebra, 15] = (zebra + base_cr_num + (k+1)*base_node_num - k*base_ele_num + l + 1)
                        element[zebra, 14] = (zebra + base_cr_num + base_hzn_num + m + (k+1)*base_node_num - k*base_ele_num + 2)
                        element[zebra, 16] = (zebra + base_cr_num + base_hzn_num + m + (k+1)*base_node_num - k*base_ele_num + 1)                       
                        element[zebra, 17] = (zebra + base_node_num*(h+1) + m + k*(base_cr_num - base_ele_num) + 1)
                        element[zebra, 18] = (zebra + base_node_num*(h+1) + m + k*(base_cr_num - base_ele_num) + 2)
                        element[zebra, 19] = (zebra + base_node_num*(h+1) + m + k*(base_cr_num - base_ele_num) + l + 3)
                        element[zebra, 20] = (zebra + base_node_num*(h+1) + m + k*(base_cr_num - base_ele_num) + l + 2)
                        zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'x':       # gap in x direction only
            # defining nodes
            base_ele_num = l*t
            base_node_num = (3*l)*(2*t+1) - base_ele_num
            base_cr_num = (2*l)*(t+1)
            base_hzn_num = l*(t+1)
            node_num = base_node_num*(h+1) + h*base_cr_num
            nodes = np.zeros(shape=(node_num, 4))
            rhino = 0
            for z in range(0, (h+1)*size[2], size[2]):
                # Corner nodes of elements
                for y in range(0, (t+1)*size[1], size[1]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y
                    nodes[rhino, 3] = z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y
                        nodes[rhino+1, 3] = z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y
                    nodes[rhino, 3] = z
                    rhino += 1
                # Midside nodes on horizontal element sides
                for y in range(0, (t+1)*size[1], size[1]):
                    dolphin = 0
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x + dolphin*gap[0]
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = z
                        rhino += 1
                        dolphin += 1
                # Midside nodes on vertical element sides
                for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = q_y
                    nodes[rhino, 3] = z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = q_y
                        nodes[rhino, 3] = z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = q_y
                        nodes[rhino+1, 3] = z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = q_y
                    nodes[rhino, 3] = z
                    rhino += 1
            # Midside nodes for cube z direction
            for q_z in np.arange(float(size[2])/2, (h)*size[2], size[2]):
                for y in range(0, (t+1)*size[1], size[1]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y
                    nodes[rhino, 3] = q_z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y
                        nodes[rhino, 3] = q_z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y
                        nodes[rhino+1, 3] = q_z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y
                    nodes[rhino, 3] = q_z
                    rhino += 1
            # defining elements
            ele_num = l*t*h
            element = np.zeros(shape=(ele_num, 21))
            zebra = 0
            for k in range(0, h, 1):
                for m in range(0, t, 1):
                    for n in range(0, l, 1):
                        element[zebra, 0]    = zebra+1
                        element[zebra, 1]    = 2*zebra + k*base_node_num - 2*k*base_ele_num + 1
                        element[zebra, 2]    = 2*zebra + k*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 3]    = 2*zebra + k*base_node_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 4]    = 2*zebra + k*base_node_num - 2*k*base_ele_num + 2*l + 1
                        
                        element[zebra, 9]    = zebra + base_cr_num + k*base_node_num - k*base_ele_num + 1
                        element[zebra, 11]   = zebra + base_cr_num + k*base_node_num - k*base_ele_num + l + 1
                        element[zebra, 10]   = 2*zebra + base_cr_num + base_hzn_num + k*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 12]   = 2*zebra + base_cr_num + base_hzn_num + k*base_node_num - 2*k*base_ele_num + 1
                        
                        element[zebra, 5]    = 2*zebra + (k+1)*base_node_num - 2*k*base_ele_num + 1
                        element[zebra, 6]    = 2*zebra + (k+1)*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 7]    = 2*zebra + (k+1)*base_node_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 8]    = 2*zebra + (k+1)*base_node_num - 2*k*base_ele_num + 2*l + 1
                        
                        element[zebra, 13]   = zebra + base_cr_num + (k+1)*base_node_num - k*base_ele_num + 1
                        element[zebra, 15]   = zebra + base_cr_num + (k+1)*base_node_num - k*base_ele_num + l + 1
                        element[zebra, 14]   = 2*zebra + base_cr_num + base_hzn_num + (k+1)*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 16]   = 2*zebra + base_cr_num + base_hzn_num + (k+1)*base_node_num - 2*k*base_ele_num + 1
                        
                        element[zebra, 17]   = 2*zebra + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 1
                        element[zebra, 18]   = 2*zebra + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2
                        element[zebra, 19]   = 2*zebra + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 20]   = 2*zebra + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2*l + 1
                        zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'y':       # gap in y direction only
            # defining nodes
            base_ele_num = l*t
            base_node_num = (2*l+1)*(3*t) - base_ele_num
            base_cr_num = (l+1)*(2*t)
            base_hzn_num = 2*l*t
            node_num = base_node_num*(h+1) + h*base_cr_num
            nodes = np.zeros(shape=(node_num, 4))
            rhino = 0
            for z in range(0, (h+1)*size[2], size[2]):
                # Corner nodes of elements
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = z
                    rhino += 1
                for y in range(1, t, 1):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                # Midside nodes on horizontal element sides
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = z
                    rhino += 1
                for y in range(1, t, 1):
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                # Midside nodes on vertical element sides
                dolphin = 0
                for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = q_y + dolphin*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                    dolphin += 1
            # Midside nodes for cube z direction
            for q_z in np.arange(float(size[2])/2, (h)*size[2], size[2]):
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = q_z
                    rhino += 1
                for y in range(1, t, 1):
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = q_z
                        rhino += 1
                    for x in range(0, (l+1)*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = q_z
                        rhino += 1
                for x in range(0, (l+1)*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = q_z
                    rhino += 1

            # defining elements
            ele_num = l*t*h
            element = np.zeros(shape=(ele_num, 21))
            zebra = 0
            for k in range(0, h, 1):
                for m in range(0, t, 1):
                    for n in range(0, l, 1):
                        element[zebra, 0]    = zebra+1
                        element[zebra, 1]    = zebra + m*(2+l) + k*base_node_num - k*base_ele_num + 1
                        element[zebra, 2]    = zebra + m*(2+l) + k*base_node_num - k*base_ele_num + 2
                        element[zebra, 3]    = zebra + m*(2+l) + k*base_node_num - k*base_ele_num + l + 3
                        element[zebra, 4]    = zebra + m*(2+l) + k*base_node_num - k*base_ele_num + l + 2
                        
                        element[zebra, 9]    = zebra + m*l + base_cr_num + k*(base_node_num - base_ele_num) + 1
                        element[zebra, 11]   = zebra + m*l + base_cr_num + k*(base_node_num - base_ele_num) + l + 1
                        element[zebra, 10]   = zebra + m + base_cr_num + base_hzn_num + k*(base_node_num - base_ele_num) + 2
                        element[zebra, 12]   = zebra + m + base_cr_num + base_hzn_num + k*(base_node_num - base_ele_num) + 1
                        
                        element[zebra, 5]    = zebra + m*(2+l) + (k+1)*base_node_num - k*base_ele_num + 1
                        element[zebra, 6]    = zebra + m*(2+l) + (k+1)*base_node_num - k*base_ele_num + 2
                        element[zebra, 7]    = zebra + m*(2+l) + (k+1)*base_node_num - k*base_ele_num + l + 3
                        element[zebra, 8]    = zebra + m*(2+l) + (k+1)*base_node_num - k*base_ele_num + l + 2
                        
                        element[zebra, 13]   = zebra + m*l + base_cr_num + (k+1)*base_node_num - k*base_ele_num + 1
                        element[zebra, 15]   = zebra + m*l + base_cr_num + (k+1)*base_node_num - k*base_ele_num + l + 1
                        element[zebra, 14]   = zebra + m + base_cr_num + base_hzn_num + (k+1)*base_node_num - k*base_ele_num + 2
                        element[zebra, 16]   = zebra + m + base_cr_num + base_hzn_num + (k+1)*base_node_num - k*base_ele_num + 1
                        
                        element[zebra, 17]   = zebra + m*(2+l) + base_node_num*(h+1) + k*(base_cr_num - base_ele_num) + 1
                        element[zebra, 18]   = zebra + m*(2+l) + base_node_num*(h+1) + k*(base_cr_num - base_ele_num) + 2
                        element[zebra, 19]   = zebra + m*(2+l) + base_node_num*(h+1) + k*(base_cr_num - base_ele_num) + l + 3
                        element[zebra, 20]   = zebra + m*(2+l) + base_node_num*(h+1) + k*(base_cr_num - base_ele_num) + l + 2
                        zebra += 1
        #-----------------------------------------------------------------#
        elif gap_drct == 'xy':      # gap in both directions
            # defining nodes
            base_ele_num = l*t
            base_node_num = (3*l)*(3*t) - base_ele_num
            base_cr_num = 4*l*t
            base_hzn_num = 2*l*t
            node_num = base_node_num*(h+1) + h*base_cr_num
            nodes = np.zeros(shape=(node_num, 4))
            rhino = 0
            for z in range(0, (h+1)*size[2], size[2]):
                # Corner nodes of elements
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = 0
                nodes[rhino, 3] = z
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = z
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = 0
                    nodes[rhino+1, 3] = z
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = 0
                nodes[rhino, 3] = z
                rhino += 1
                for y in range(1, t, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino+1, 3] = z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    #---------------------------------------#
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y*size[1]+y*gap[1]
                        nodes[rhino+1, 3] = z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                #-----------------------------------------#
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                nodes[rhino, 3] = z
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino+1, 3] = z
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                nodes[rhino, 3] = z
                rhino += 1
                # Midside nodes on horizontal element sides
                dolphin = 0
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x + dolphin*gap[0]
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = z
                    rhino += 1
                    dolphin += 1
                for y in range(1, t, 1):
                    dolphin = 0
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x + dolphin*gap[0]
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                        dolphin += 1
                    dolphin = 0
                    for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = q_x + dolphin*gap[0]
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = z
                        rhino += 1
                        dolphin += 1
                dolphin = 0
                for q_x in np.arange(float(size[0])/2, l*size[0], size[0]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = q_x + dolphin*gap[0]
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    dolphin += 1
                # Midside nodes on vertical element sides
                seal = 0
                for q_y in np.arange(float(size[1])/2, (t)*size[1], size[1]):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = q_y + seal*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = q_y + seal*gap[1]
                        nodes[rhino, 3] = z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = q_y + seal*gap[1]
                        nodes[rhino+1, 3] = z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = q_y + seal*gap[1]
                    nodes[rhino, 3] = z
                    rhino += 1
                    seal += 1
            # Midside nodes for cube z direction
            for q_z in np.arange(float(size[2])/2, (h)*size[2], size[2]):
                # Corner nodes of elements
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = 0
                nodes[rhino, 3] = q_z
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = 0
                    nodes[rhino, 3] = q_z
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = 0
                    nodes[rhino+1, 3] = q_z
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = 0
                nodes[rhino, 3] = q_z
                rhino += 1
                for y in range(1, t, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    nodes[rhino, 3] = q_z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino, 3] = q_z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino+1, 3] = q_z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y*size[1]+(y-1)*gap[1]
                    nodes[rhino, 3] = q_z
                    rhino += 1
                    #---------------------------------------#
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = 0
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    nodes[rhino, 3] = q_z
                    rhino += 1
                    for x in range(1, l, 1):
                        nodes[rhino, 0] = rhino+1
                        nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino, 2] = y*size[1]+y*gap[1]
                        nodes[rhino, 3] = q_z
                        nodes[rhino+1, 0] = rhino+2
                        nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                        nodes[rhino+1, 2] = y*size[1]+y*gap[1]
                        nodes[rhino+1, 3] = q_z
                        rhino += 2
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino, 2] = y*size[1]+y*gap[1]
                    nodes[rhino, 3] = q_z
                    rhino += 1
                #-----------------------------------------#
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = 0
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                nodes[rhino, 3] = q_z
                rhino += 1
                for x in range(1, l, 1):
                    nodes[rhino, 0] = rhino+1
                    nodes[rhino, 1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino, 3] = q_z
                    nodes[rhino+1, 0] = rhino+2
                    nodes[rhino+1, 1] = x*size[0]+x*gap[0]
                    nodes[rhino+1, 2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino+1, 3] = q_z
                    rhino += 2
                nodes[rhino, 0] = rhino+1
                nodes[rhino, 1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino, 2] = (y+1)*size[1]+y*gap[1]
                nodes[rhino, 3] = q_z
                rhino += 1

            # defining elements
            ele_num = l*t*h
            element = np.zeros(shape=(ele_num, 21))
            zebra = 0
            for k in range(0, h, 1):
                for m in range(0, t, 1):
                    for n in range(0, l, 1):
                        element[zebra, 0]    = zebra+1
                        element[zebra, 1]    = 2*zebra + 2*m*l + k*base_node_num - 2*k*base_ele_num + 1
                        element[zebra, 2]    = 2*zebra + 2*m*l + k*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 3]    = 2*zebra + 2*m*l + k*base_node_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 4]    = 2*zebra + 2*m*l + k*base_node_num - 2*k*base_ele_num + 2*l + 1
                        
                        element[zebra, 9]    = zebra + m*l + base_cr_num + k*(base_node_num - base_ele_num) + 1
                        element[zebra, 11]   = zebra + m*l + base_cr_num + k*(base_node_num - base_ele_num) + l + 1
                        element[zebra, 10]   = 2*zebra + base_cr_num + base_hzn_num + k*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 12]   = 2*zebra + base_cr_num + base_hzn_num + k*base_node_num - 2*k*base_ele_num + 1
                        
                        element[zebra, 5]    = 2*zebra + 2*m*l + (k+1)*base_node_num - 2*k*base_ele_num + 1
                        element[zebra, 6]    = 2*zebra + 2*m*l + (k+1)*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 7]    = 2*zebra + 2*m*l + (k+1)*base_node_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 8]    = 2*zebra + 2*m*l + (k+1)*base_node_num - 2*k*base_ele_num + 2*l + 1
                        
                        element[zebra, 13]   = zebra + m*l + base_cr_num + (k+1)*base_node_num - k*base_ele_num + 1
                        element[zebra, 15]   = zebra + m*l + base_cr_num + (k+1)*base_node_num - k*base_ele_num + l + 1
                        element[zebra, 14]   = 2*zebra + base_cr_num + base_hzn_num + (k+1)*base_node_num - 2*k*base_ele_num + 2
                        element[zebra, 16]   = 2*zebra + base_cr_num + base_hzn_num + (k+1)*base_node_num - 2*k*base_ele_num + 1
                        
                        element[zebra, 17]   = 2*zebra + 2*m*l + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 1
                        element[zebra, 18]   = 2*zebra + 2*m*l + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2
                        element[zebra, 19]   = 2*zebra + 2*m*l + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2*l + 2
                        element[zebra, 20]   = 2*zebra + 2*m*l + base_node_num*(h+1) + k*base_cr_num - 2*k*base_ele_num + 2*l + 1
                        zebra += 1
            #end of gap
        #-----------------------------------------------------------------#
        #Plotting of nodes
        if dispM is True:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            tiger = -1
            for c in ('r', 'b'):
                for lion in range((h+1)*base_node_num/(h+1)):
                    tiger += 1
                    ax.scatter(nodes[tiger, 1], nodes[tiger, 2], nodes[tiger, 3], c=c, marker='*')
                    ax.text(nodes[tiger, 1], nodes[tiger, 2], nodes[tiger, 3], '%s' % (str(nodes[tiger, 0])), size=10, 
                            zorder=1, color=c)
            for lion in range((h+1)*base_node_num, len(nodes), 1):
                tiger += 1
                ax.scatter(nodes[tiger, 1], nodes[tiger, 2], nodes[tiger, 3], c='c', marker='*')
                ax.text(nodes[tiger, 1], nodes[tiger, 2], nodes[tiger, 3], '%s' % (str(nodes[tiger, 0])), size=10, 
                        zorder=1, color='c')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        #end of option 3
    #-------------------------------------------------------------------------#
    return nodes, element  # el_center
