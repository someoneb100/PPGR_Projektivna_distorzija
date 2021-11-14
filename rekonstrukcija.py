import math, itertools, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#pomocne f-je
def gen_matrix(A):
    if(type(A) == type(np.ndarray(()))):
        return np.copy(A)
    else:
        shape=(len(A),len(A[0]))
        return np.ndarray(shape, buffer=np.array(A, dtype=np.float), dtype=np.float)

def gen_vector(p):
    if(type(p) == type(np.array([]))):
        return np.copy(p)
    else:
        return np.array(p, dtype=np.float)

def gen_vec_matrix(p):
    return gen_matrix([[    0, -p[2],  p[1]]
                      ,[ p[2],     0, -p[0]]
                      ,[-p[1],  p[0],     0]])

sqrt2 = math.sqrt(2)

E = np.identity(3)

def gen_E(i = 3):
    return np.identity(i)

def intensity(p):
    return math.sqrt((p*p).sum())

def normalize(p):
    i = intensity(p)
    i = 1 if i == 0 else i
    return p / i

def distance(a, b):
    p = b-a
    return math.sqrt((p*p).sum())

#F-je za kombinatoriku
def choose(n, k):
    if n < k:
        return 0
    a = int(math.factorial(n))
    b = int(math.factorial(k)) * int(math.factorial(n - k))
    return int(a/b)

def combination(n, k, m):
    result = []
    a      = n
    b      = k
    x      = (choose(n, k) - 1) - m
    for i in range(0, k):
        a = a - 1
        while choose(a, b) > x:
            a = a - 1
        result.append(n - 1 - a)
        x = x - choose(a, b)
        b = b - 1
    return result



#F-ja za racunanje koordinata nevidljive tacke
def nevidljiva(T1, T2, T3, T5, T6, T7):
    INV1 = np.cross(np.cross(T5, T6), np.cross(T1, T2))
    INV2 = np.cross(np.cross(T3, T2), np.cross(T7, T6))
    res  = np.cross(np.cross(INV1, T7), np.cross(INV2, T5))
    if(res[-1] == 0): return res
    return res / res[-1]


# F-je vezane za DLT i RANSAC
def korespodencije_f(o, s):
    rez = np.zeros((len(o)*2,9))
    for i, (t, tp) in enumerate(zip(o, s)):
        rez[i][:3] = tp[0] * t
        rez[i][3:6] = tp[1] * t
        rez[i][6:] = tp[2] * t
    return rez

def normalizacija_dlp(xs):
    xs = gen_matrix(xs)
    for i, x in enumerate(x[-1] for x in xs):
        xs[i] /= x
    p_t = gen_E(len(xs[0]))
    p_s = gen_E(len(xs[0]))
    centroid = gen_vector([x.sum()/len(x) for x in xs.transpose()])
    prosek = (sqrt2)/(sum(distance(centroid, x) for x in xs)/len(xs))
    for i, t in enumerate(centroid[:-1]):
        p_s[i][i] = prosek
        p_t[i][-1] = -t
    p = np.dot(p_s, p_t)
    new_xs = np.zeros((len(xs),len(xs[0])))
    for i, x in enumerate(xs):
        new_xs[i] = np.dot(p, x)
    return p, new_xs

def svd_matrica(A):
    _, _, vt = np.linalg.svd(A, full_matrices=True)
    v = vt[-1]
    t = int(len(v)/3)
    return gen_matrix([v[:t], v[t:2*t], v[2*t:]])


def DLP(originali, slike, korespodencije_f):
    p_originali, norm_originali = normalizacija_dlp(originali)
    p_slike, norm_slike = normalizacija_dlp(slike)
    T = svd_matrica(korespodencije_f(norm_originali, norm_slike))
    T = np.dot(np.linalg.inv(p_slike), T)
    T = np.dot(T, p_originali)
    return T


def epipolovi_zimmermann(F):
    u, dd, vt = np.linalg.svd(F, full_matrices=True)
    e1 = vt[2]/vt[2][2]
    e2 = u.transpose()[2]/u.transpose()[2][2]

    dd1 = gen_E(3)
    dd1[2][2] = 0
    dd1 *= dd

    return e1, e2, np.dot(np.dot(u, dd1), vt)

def kamere(e1, e2, F):
    T1 = np.append(np.identity(3), [[0,0,0]], axis=0).transpose()
    E2 = gen_vec_matrix(e2)
    T2 = np.zeros((4,3))
    T2[:-1] = np.dot(E2, F).transpose()
    T2[-1] = e2
    T2 = T2.transpose()
    return T1, T2

def jne_triangulacije(x, y, T1, T2):
    A = np.zeros((4, 4))
    A[0] = x[1]  * T1[2] - T1[1]
    A[1] = -x[0] * T1[2] + T1[0]
    A[2] = y[1]  * T2[2] - T2[1]
    A[3] = -y[0] * T2[2] + T2[0]
    return A

def triangulacija(levi, desni, T1, T2):
    T = np.zeros((len(levi), 3))
    for i, (l, d) in enumerate(zip(levi, desni)):
        _, _, A = np.linalg.svd(jne_triangulacije(l, d, T1, T2))
        T[i] = (A[-1]/A[-1][-1])[:-1]
        T[i][-1] *= 400
    return T


def plot_boxes(T):
    box_lines = [(0,1),(1,2),(2,3),(3,0)
                ,(4,5),(5,6),(6,7),(7,4)
                ,(0,4),(1,5),(2,6),(3,7)]
    box1, box2, box3 = T[:8], T[8:16], T[16:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (frst, scnd) in box_lines:
        ax.plot([box1[frst][0], box1[scnd][0]], [box1[frst][1], box1[scnd][1]], [box1[frst][2], box1[scnd][2]], color="red")
        ax.plot([box2[frst][0], box2[scnd][0]], [box2[frst][1], box2[scnd][1]], [box2[frst][2], box2[scnd][2]], color="green")
        ax.plot([box3[frst][0], box3[scnd][0]], [box3[frst][1], box3[scnd][1]], [box3[frst][2], box3[scnd][2]], color="blue")
    plt.show()


if __name__ == "__main__":
    levi_originali = gen_matrix([[814, 111, 1] #Prva kutija
                                ,[951, 158, 1]
                                ,[989, 123, 1]
                                ,[855,  77, 1]
                                ,[791, 304, 1]
                                ,[912, 358, 1]
                                ,[950, 323, 1]
                                ,[  0,   0, 0]
                                ,[323, 344, 1] #Druga kutija
                                ,[454, 369, 1]
                                ,[511, 272, 1]
                                ,[386, 249, 1]
                                ,[365, 559, 1]
                                ,[478, 583, 1]
                                ,[526, 488, 1]
                                ,[  0,   0, 0]
                                ,[136, 550, 1] #Treca kutija
                                ,[434, 761, 1]
                                ,[818, 382, 1]
                                ,[547, 252, 1]
                                ,[175, 655, 1]
                                ,[451, 861, 1]
                                ,[806, 489, 1]
                                ,[  0,   0, 0]])


    desni_originali = gen_matrix([[ 911, 446, 1] #Prva kutija
                                 ,[ 811, 561, 1]
                                 ,[ 917, 611, 1]
                                 ,[1013, 491, 1]
                                 ,[   0,   0, 0]
                                 ,[ 771, 770, 1]
                                 ,[ 859, 823, 1]
                                 ,[ 956, 702, 1]
                                 ,[ 297,  73, 1] #Druga kutija
                                 ,[ 251, 120, 1]
                                 ,[ 371, 137, 1]
                                 ,[ 414,  89, 1]
                                 ,[   0,   0, 0]
                                 ,[ 287, 324, 1]
                                 ,[ 395, 343, 1]
                                 ,[ 434, 288, 1]
                                 ,[   0,   0, 0] #Treca kutija
                                 ,[ 136, 318, 1]
                                 ,[ 527, 531, 1]
                                 ,[ 745, 347, 1]
                                 ,[   0,   0, 0]
                                 ,[ 162, 426, 1]
                                 ,[ 531, 643, 1]
                                 ,[ 735, 454, 1]])


    leve_nepoznate, desne_nepoznate = {7, 15, 23}, {4, 12, 16, 20}
    vidljive = set(range(len(levi_originali))) - (leve_nepoznate | desne_nepoznate)

    desni_originali[16] = nevidljiva(desni_originali[21]
                                    ,desni_originali[22]
                                    ,desni_originali[23]
                                    ,desni_originali[17]
                                    ,desni_originali[18]
                                    ,desni_originali[19])

    for i in range(0, 3*8, 8):
        levi_originali[7+i] = nevidljiva(levi_originali[0+i]
                                        ,levi_originali[1+i]
                                        ,levi_originali[2+i]
                                        ,levi_originali[4+i]
                                        ,levi_originali[5+i]
                                        ,levi_originali[6+i])
        desni_originali[4+i] = nevidljiva(desni_originali[1+i]
                                         ,desni_originali[2+i]
                                         ,desni_originali[3+i]
                                         ,desni_originali[5+i]
                                         ,desni_originali[6+i]
                                         ,desni_originali[7+i])


    print("Nakon pronalazenja nevidljivih tacaka, tacke leve i desne slike (zaokruzene):")

    def print_int(x, num = 4):
        a = str(int(x))
        return " "*(num - len(a)) + a

    np.set_printoptions(suppress=True,
                        formatter={"all" : lambda x: print_int(x, 4)})
    for i, (x, y) in enumerate(zip(levi_originali, desni_originali)):
        print(print_int(i+1, 2), end=". ")
        print("{bx[0]}{x[0]} {x[1]} 1{bx[1]}  -  {by[0]}{y[0]} {y[1]} 1{by[1]}"
        .format(bx = "()" if i in leve_nepoznate else "[]"
               ,by = "()" if i in desne_nepoznate else "[]"
               ,x = tuple(map(print_int, x)), y = tuple(map(print_int, y))))
    np.set_printoptions()

    izbori = gen_vector(random.sample(vidljive, 8))
    izbori.sort()
    print("Izabrani parovi tacaka:")
    np.set_printoptions(suppress=True,
                        formatter={"all" : lambda x: print_int(x+1, 2)})
    print(izbori)
    np.set_printoptions()

    izabrani_levi = np.zeros((8,3))
    izabrani_desni = np.zeros((8,3))
    for i, val in enumerate(izbori):
        izabrani_levi[i]  = levi_originali[int(val)]
        izabrani_desni[i] = desni_originali[int(val)]

    print("Matrica F:")
    F = svd_matrica(korespodencije_f(izabrani_levi, izabrani_desni))
    np.set_printoptions(suppress=False)
    print(F)
    print("Provera da li je ytFx = 0:")
    test = gen_vector([np.dot(desni, np.dot(F, levi)) for levi, desni in zip(izabrani_levi, izabrani_desni)])
    print(test)
    np.set_printoptions()
    print("Determinanta od F:", np.linalg.det(F))
    e1, e2, F = epipolovi_zimmermann(F)
    print("Epipol 1:", e1)
    print("Epipol 2:", e2)
    print("Determinanta od F nakon Zimmermanna:", np.linalg.det(F))

    np.set_printoptions(suppress=True)
    T1, T2 = kamere(e1, e2, F)
    print("Matrica T1:")
    print(T1)
    print("Matrica T2:")
    print(T2)

    slike3d = triangulacija(levi_originali, desni_originali, T1, T2)
    plot_boxes(slike3d)
