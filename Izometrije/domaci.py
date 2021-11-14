import numpy as np
from math import sin, cos, pi, atan2, asin, acos, atan

#Pomocne funkcije i promenljive
tacnost = 10**-10
E = np.identity(3)

def intenzitet(p):
    return sum(x*x for x in p)**0.5

def is_ortogonal(A):
    return (abs(np.dot(A, A.transpose()) - E) < tacnost).all()

def det_is_1(A):
    return abs(np.linalg.det(A) - 1) < tacnost

def is_E(A):
    return (A - E < tacnost).all()


def gen_matrix(A):
    if(type(A) == type(np.ndarray(()))):
        return np.copy(A)
    else:
        return np.ndarray((3,3), buffer=np.array(A, dtype=np.float), dtype=np.float)

def gen_vector(p):
    if(type(p) == type(np.array([]))):
        return np.copy(p)
    else:
        return np.array(p, dtype=np.float)


#Prva funkcija: Euler2A [angle fi, angle teta, angle xi] -> [3x3 matrix A]
#A = Rz(xi)*Ry(teta)*Rx(fi)
def euler2a(fi, teta, xi): ##WORKS
    rx, ry, rz = gen_matrix(E), gen_matrix(E), gen_matrix(E)
    rx[1][1], rx[2][2] = cos(fi), cos(fi)
    rx[1][2], rx[2][1] = -sin(fi), sin(fi)
    ry[0][0], ry[2][2] = cos(teta), cos(teta)
    ry[2][0], ry[0][2] = -sin(teta), sin(teta)
    rz[0][0], rz[1][1] = cos(xi), cos(xi)
    rz[0][1], rz[1][0] = -sin(xi), sin(xi)
    return np.dot(np.dot(rz, ry), rx)


#Druga funkcija: AxisAngle [3x3 matrix A] -> [3 vector p, angle fi]
#A = Rp(fi)
def axisAngle(A):
    A = gen_matrix(A)
    if(is_E(A) or not is_ortogonal(A) or not det_is_1(A)):
        raise ValueError("Passed matrix must be an orthogonal matrix with determinant equal to 1 that isn't identity matrix!")
    p = A-E
    p = np.cross(p[0], p[1])
    p /= intenzitet(p)
    u = gen_vector([abs(p[0]-p[1])>0.001, 1, 0])
    u[2] = (u[0]*p[0] + u[1]*p[1]) / (-p[2])
    uprim = np.dot(A, u)
    fi = acos(np.dot(u, uprim) / (intenzitet(u)*intenzitet(uprim)))
    if(np.dot(np.cross(u, uprim), p) < 0):
        p = -p
    return p, fi


#Treca funkcija: Rodrigez [3 vector p, angle fi] -> [3x3 matrix A]
#A = Rp(fi)
def rodrigez(p, fi):
    p = gen_vector(p)
    p /= intenzitet(p)
    ppt = np.dot(p.reshape(3,1), p.reshape(1,3))
    px = gen_matrix([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    return ppt + cos(fi)*(gen_matrix(E)-ppt) + sin(fi)*px


#Cetvrta funkcija: A2Euler [3x3 matrix A] -> [angle fi, angle teta, angle xi]
#A = Rz(xi)*Ry(teta)*Rx(fi)
def a2euler(A):
    A = gen_matrix(A)
    if(not is_ortogonal(A) or not det_is_1(A)):
        raise ValueError("Passed matrix must be an orthogonal matrix with determinant equal to 1!")
    if(abs(A[2][0]) == 1):
        return 0, (-pi/2) * A[2][0], atan2(-A[0][1], a[1][1])
    else:
        return atan2(A[2][1], A[2][2]), asin(-A[2][0]), atan2(A[1][0], A[0][0])


#Peta funkcija: AxisAngle2Q [3 vector p, angle fi] -> [4 vector q]
#Cq = Rp(x)
def axisAngle2q(p, fi):
    p = sin(fi/2)*(p / intenzitet(p))
    q = gen_vector([p[0], p[1], p[2], cos(fi/2)])
    return q / intenzitet(q)



#Sesta funkcija: Q2AxisAngle [4 vector q] -> [3 vector p, angle fi]
#Cq = Rp(x)
def q2axisAngle(q):
    q = gen_vector(q)
    if((q == 0).all()):
        raise ValueError("Passed Quaternion must be not 0")
    q /= intenzitet(q)
    if(q[3]<0):
        q = -q
    fi, p = 2*acos(q[3]) , q[:3]
    return gen_vector([1,0,0]) if abs(q[3])==1 else p/intenzitet(p), fi


if __name__ == "__main__":
    fi, teta, xi = atan(1/2), asin(7/8), -atan(2)
    print("fi, teta, xi = atan(1/2), asin(7/8), -atan(2)")
    print()

    print("Euler2A:")
    A = euler2a(fi, teta, xi)
    print(A)
    print()

    print("AxisAngle:")
    p, alpha = axisAngle(A)
    print(p, alpha)
    print()

    print("Rodrigez:")
    A = rodrigez(p, alpha)
    print(A)
    print()

    print("A2Euler:")
    fi, teta, xi = a2euler(A)
    print(fi, teta, xi)
    print()

    print("AxisAngle2Q:")
    q = axisAngle2q(p, alpha)
    print(q)
    print()

    print("Q2AxisAngle:")
    p, alpha = q2axisAngle(q)
    print(p, alpha)
