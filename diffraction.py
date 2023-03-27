import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import *
from PIL import Image
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.size']= 10
matplotlib.rcParams['axes.unicode_minus']=False
#弗朗和费矩孔衍射
class Lightwave(object):
    def __init__(self,A,k):
        self.A,self.k=A,k
    def __call__(self,L):
        return self.A*np.cos(self.k*L)

U=Lightwave(1,2*pi/0.6)

def delta_L(x,y,x_1,y_1,f):
    cos_alpha=x_1/sqrt(x_1**2+f**2)
    cos_beta=y_1/sqrt(y_1**2+f**2)
    return -x*cos_alpha-y*cos_beta

def ample_and_I(x_1,y_1,x,y,f):
    a=np.zeros([len(x_1),len(y_1)])
    I=np.zeros([len(x_1),len(y_1)])
    I_0=0
    for i in range(len(x_1)):
        for j in range(len(y_1)):
            a[i,j]=np.sum(U(delta_L(x,y,x_1[i,j],y_1[i,j],f)))
            I[i,j]=a[i,j]**2
            if I[i,j]>=I_0:
                I_0=I[i,j]
    return a/sqrt(I_0),I/I_0

def square_diffraction(a,b,f,x0,y0):
    x = np.linspace(x0-a / 2, x0+a / 2, 101)
    y = np.linspace(y0-b / 2, y0+b / 2, 101)
    x, y = np.meshgrid(x, y)  # 衍射屏处坐标
    x_1 = np.linspace(-30 * a, 30 * a, 501)
    y_1 = np.linspace(-30 * b, 30 * b, 501)
    x_1, y_1 = np.meshgrid(x_1, y_1)  # 接收屏处坐标

    A,I=ample_and_I(x_1,y_1,x,y,f)
    fig=plt.figure()
    ax_I = fig.add_subplot(1,2,1,projection='3d')
    ax_I.plot_surface(x_1, y_1, I,cmap='rainbow')
    ax_I.contour(x_1,y_1,I,zdir='I',offset=-0.1,cmap='rainbow')
    plt.title('矩孔衍射相对光强分布')

    ax_a =fig.add_subplot(1,2,2,projection='3d')
    ax_a.plot_surface(x_1,y_1,A,cmap='rainbow')
    plt.title('矩孔衍射相对振幅分布')
    plt.show()
    return A,I,x_1,y_1

a,i,x,y=square_diffraction(20,20,5000,0,0)

pic=Image.fromarray(i*255)
pic=pic.convert('L')
pic=pic.point(lambda x:x>3,'1')
pic.show()
#菲涅尔圆孔衍射


def chara_fun(r,rho):#示性函数
    con1=r<=rho
    con2=r>rho
    c=np.zeros([len(r),len(r)])
    c[con1]=1.0
    c[con2]=0.0
    return c

class Lightwave_point(object):
    def __init__(self,A,k):
        self.A,self.k=A,k
    def __call__(self,L,r,rho,s):
        return self.A*chara_fun(r,rho)/s*np.cos(self.k*L)

U_2=Lightwave_point(1,2*pi/0.6)

def delta_L2(x,y,x_1,y_1,R,b):
    return np.sqrt(x**2+y**2+(R-np.sqrt(R**2-x**2-y**2)+b)**2)+(x**2+y**2-2*x*x_1-2*y*y_1)/(2*(R-np.sqrt(R**2-x**2-y**2)+b))

def ample_and_I2(x,y,x_1,y_1,R,b,rho):
    a = np.zeros([len(x_1), len(y_1)])
    I = np.zeros([len(x_1), len(y_1)])
    I_0 = 0
    for i in range(len(x_1)):
        for j in range(len(y_1)):
            a[i, j] = np.sum(b*U_2(delta_L2(x, y, x_1[i, j], y_1[i, j], R,b),
                                                              np.sqrt(x**2+y**2),rho,(R-np.sqrt(R**2-x**2-y**2)+b)))
            I[i, j] = a[i, j] ** 2
            if I[i, j] >= I_0:
                I_0 = I[i, j]
    return a / sqrt(I_0), I / I_0

def fresnel_diffraction(R,b,rho):
    x = np.linspace(-rho , rho , 101)
    y = np.linspace(-rho , rho, 101)
    x, y = np.meshgrid(x, y)  # 衍射屏处坐标
    x_1 = np.linspace(-30 * rho, 30 * rho, 501)
    y_1 = np.linspace(-30 * rho, 30 * rho, 501)
    x_1, y_1 = np.meshgrid(x_1, y_1)  # 接收屏处坐标

    A,I=ample_and_I2(x,y,x_1,y_1,R,b,rho)
    fig=plt.figure()
    ax_I = fig.add_subplot(1,2,1,projection='3d')
    ax_I.plot_surface(x_1, y_1, I,cmap='rainbow')
    ax_I.contour(x_1,y_1,I,zdir='I',offset=-0.1,cmap='rainbow')
    plt.title('菲涅尔衍射相对光强分布')

    ax_a =fig.add_subplot(1,2,2,projection='3d')
    ax_a.plot_surface(x_1,y_1,A,cmap='rainbow')
    plt.title('菲涅尔衍射相对振幅分布')
    plt.show()
    return A,I,x_1,y_1


a,i,x,y=fresnel_diffraction(1000000.0,4000000.0,500)
pic=Image.fromarray(i*255)
pic=pic.convert('L')
pic=pic.point(lambda x:x>5,'1')
pic.show()
#对于数组使用函数需要用numpy包的！！！