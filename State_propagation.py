# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:27:16 2022

@author: Tanmay Ganguli
"""

import numpy as np
import matplotlib.pyplot as plt

def state_t(t_i,t_f,omega_i,q_i,inertia,const=0.1,wxplot=False,
            wyplot=False,wzplot=False):
    #returns omega, quaternion at time t_f, given t_i,omega_initial,quaternion
    #_initial
    
    [qx_i,qy_i,qz_i,qs_i] = q_i
    [wx_i,wy_i,wz_i] = omega_i
    
    #for plotting purpose
    #global wx_list,wy_list,wz_list,t_list
    wx_list = [wx_i]
    wy_list = [wy_i]
    wz_list = [wz_i]
    t_list = [t_i]
    
    
    def qx_dot(t,qx,qy,qz,qs):
        return 0.5*(qs*wx_i - qz*wy_i + qy*wz_i)
    def qy_dot(t,qx,qy,qz,qs):
        return (qz*wx_i + qs*wy_i - qx*wz_i)*0.5
    def qz_dot(t,qx,qy,qz,qs):
        return (-qy*wx_i + qx*wy_i + qs*wz_i)*0.5
    def qs_dot(t,qx,qy,qz,qs):
        return (-qx*wx_i - qy*wy_i - qz*wz_i)*0.5
    
    def omega_dot(t,wx,wy,wz):
        I = inertia
        w = np.array([[wx],[wy],[wz]])
        L = I @ w
        L_1 = L.transpose()
        w_1 = w.transpose()
        cross_T = np.cross(w_1,L_1)
        cross = np.array([[cross_T[0,0]],[cross_T[0,1]],[cross_T[0,2]]])
        I_inv = np.linalg.inv(I)
        o_dot = I_inv @ cross + const*(I_inv @ w)
        o_dot = -1*o_dot
        return o_dot

    def omega_x_dot(t,wx,wy,wz):
        return omega_dot(t,wx,wy,wz)[0,0]

    def omega_y_dot(t,wx,wy,wz):
        return omega_dot(t,wx,wy,wz)[1,0]

    def omega_z_dot(t,wx,wy,wz):
        return omega_dot(t,wx,wy,wz)[2,0]
    
    h = 0.05
    
    for i in range(int((t_f - t_i)/h)):
        #quaternion integration
        
        k1 = h*qx_dot(t_i,qx_i,qy_i,qz_i,qs_i)
        l1 = h*qy_dot(t_i,qx_i,qy_i,qz_i,qs_i)
        m1 = h*qz_dot(t_i,qx_i,qy_i,qz_i,qs_i)
        n1 = h*qs_dot(t_i,qx_i,qy_i,qz_i,qs_i)
        
        k2 = h*qx_dot(t_i+h/2,qx_i+k1/2,qy_i+l1/2,qz_i+m1/2,qs_i+n1/2)
        l2 = h*qy_dot(t_i+h/2,qx_i+k1/2,qy_i+l1/2,qz_i+m1/2,qs_i+n1/2)
        m2 = h*qz_dot(t_i+h/2,qx_i+k1/2,qy_i+l1/2,qz_i+m1/2,qs_i+n1/2)
        n2 = h*qs_dot(t_i+h/2,qx_i+k1/2,qy_i+l1/2,qz_i+m1/2,qs_i+n1/2)
        
        k3 = h*qx_dot(t_i+h/2,qx_i+k2/2,qy_i+l2/2,qz_i+m2/2,qs_i+n2/2)
        l3 = h*qy_dot(t_i+h/2,qx_i+k2/2,qy_i+l2/2,qz_i+m2/2,qs_i+n2/2)
        m3 = h*qz_dot(t_i+h/2,qx_i+k2/2,qy_i+l2/2,qz_i+m2/2,qs_i+n2/2)
        n3 = h*qs_dot(t_i+h/2,qx_i+k2/2,qy_i+l2/2,qz_i+m2/2,qs_i+n2/2)
        
        k4 = h*qx_dot(t_i+h,qx_i+k3,qy_i+l3,qz_i+m3,qs_i+n3)
        l4 = h*qy_dot(t_i+h,qx_i+k3,qy_i+l3,qz_i+m3,qs_i+n3)
        m4 = h*qz_dot(t_i+h,qx_i+k3,qy_i+l3,qz_i+m3,qs_i+n3)
        n4 = h*qs_dot(t_i+h,qx_i+k3,qy_i+l3,qz_i+m3,qs_i+n3)
        
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        l = (l1 + 2*l2 + 2*l3 + l4)/6
        m = (m1 + 2*m2 + 2*m3 + m4)/6
        n = (n1 + 2*n2 + 2*n3 + n4)/6
        
        qx1 = qx_i + k
        qy1 = qy_i + l
        #t1 = t_i + h 
        qz1 = qz_i + m
        qs1 = qs_i + n
        
        qx_i,qy_i,qz_i,qs_i = qx1,qy1,qz1,qs1

        #omega integration
        
        p1 = h*omega_x_dot(t_i,wx_i,wy_i,wz_i)
        r1 = h*omega_y_dot(t_i,wx_i,wy_i,wz_i)
        s1 = h*omega_z_dot(t_i,wx_i,wy_i,wz_i)
        
        p2 = h*omega_x_dot(t_i+h/2,wx_i+p1/2,wy_i+r1/2,wz_i+s1/2)
        r2 = h*omega_y_dot(t_i+h/2,wx_i+p1/2,wy_i+r1/2,wz_i+s1/2)
        s2 = h*omega_z_dot(t_i+h/2,wx_i+p1/2,wy_i+r1/2,wz_i+s1/2)
        
        p3 = h*omega_x_dot(t_i+h/2,wx_i+p2/2,wy_i+r2/2,wz_i+s2/2)
        r3 = h*omega_y_dot(t_i+h/2,wx_i+p2/2,wy_i+r2/2,wz_i+s2/2)
        s3 = h*omega_z_dot(t_i+h/2,wx_i+p2/2,wy_i+r2/2,wz_i+s2/2)
        
        p4 = h*omega_x_dot(t_i+h,wx_i+p3,wy_i+r3,wz_i+s3)
        r4 = h*omega_y_dot(t_i+h,wx_i+p3,wy_i+r3,wz_i+s3)
        s4 = h*omega_z_dot(t_i+h,wx_i+p3,wy_i+r3,wz_i+s3)
        
        p = (p1 + 2*p2 + 2*p3 + p4)/6
        r = (r1 + 2*r2 + 2*r3 + r4)/6
        s = (s1 + 2*s2 + 2*s3 + s4)/6
        
        wx1 = wx_i + p
        wy1 = wy_i + r
        t1 = t_i + h 
        wz1 = wz_i + s
        
        #to plot
        wx_list.append(wx1)
        wy_list.append(wy1)
        wz_list.append(wz1)
        t_list.append(t1)
        
        t_i,wx_i,wy_i,wz_i = t1,wx1,wy1,wz1
        
    if wxplot:
        plt.plot(t_list,wx_list)
        plt.xlabel('time in seconds')
        plt.ylabel('omega_x')
        plt.show()
    if wyplot:
        plt.plot(t_list,wy_list)
        plt.xlabel('time in seconds')
        plt.ylabel('omega_y')
        plt.show()
    if wzplot:
        plt.plot(t_list,wz_list)
        plt.xlabel('time in seconds')
        plt.ylabel('omega_z')
        plt.show()
        
    return [wx1,wy1,wz1],[qx1,qy1,qz1,qs1]

omega = [0.1,5.1,0.1]
mi = [[10,0,0],[0,15,0],[0,0,20]]
q = [1,.5,.5,1]
[omega_x,omega_y,omega_z]=state_t(0, 20, omega,q, mi,wxplot=True,wyplot=True,
                                  wzplot=True,const=0)[0]
'''
print('x component of omega:',omega_x)
print('y component of omega:',omega_y)
print('z component of omega:',omega_z)
'''
print('initial angular velocity:',omega)
print('inertia tensor:',mi)





    
    
