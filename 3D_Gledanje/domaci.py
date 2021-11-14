import numpy as np


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

E = np.identity(3)

def gen_E(i = 3):
    return np.identity(i)

def intensity(p):
    return sum(x*x for x in p)**0.5

def normalize(p):
    i = intensity(p)
    i = 1 if i == 0 else i
    return p / i

def distance(a, b):
    return sum((y-x)**2 for (x,y) in zip(a, b))**0.5



#ParametriKamere [3x4 matrix T] -> [3x3 matrix K], [3x3 matrix A], [3 vector C]
def ParametriKamere(T):
    T = gen_matrix(T)
    Tt = T.transpose()
    T0 = Tt[0:3].transpose()
    if(np.linalg.det(T0)<0):
        T0 = -T0
    C = np.linalg.solve(T0, -Tt[3])
    Q, R = np.linalg.qr(np.linalg.inv(T0))
    K = np.linalg.inv(R)
    A = Q.transpose()

    sign = lambda x: -1 if x<0 else 1
    signs = list(map(sign, (K[i][i] for i in range(len(K)))))
    for i, s in enumerate(signs):
        A[i] *= s
        for j in range(len(K)):
            K[j][i] *= s

    return K, A, C

def korespodencije(o, s):
    rez = np.zeros((len(o)*2,12))
    for i, (t, tp) in enumerate(zip(o, s)):
        zero_pos = i*2
        one_pos = zero_pos + 1
        rez[zero_pos][4:8] = -tp[2] * t
        rez[zero_pos][8:] = tp[1] * t
        rez[one_pos][:4] = tp[2] * t
        rez[one_pos][8:] = -tp[0] * t
    return rez

def normalizacija_dlp(xs):
    xs = gen_matrix(xs)
    for i, x in enumerate(x[-1] for x in xs):
        xs[i] /= x
    p_t = gen_E(len(xs[0]))
    p_s = gen_E(len(xs[0]))
    centroid = [x.sum()/len(x) for x in xs.transpose()]
    prosek = (2**0.5)/(sum(distance(centroid, x) for x in xs)/len(xs))
    for i, t in enumerate(centroid[:-1]):
        p_s[i][i] = prosek
        p_t[i][-1] = -t
    p = np.dot(p_s, p_t)
    new_xs = np.zeros((len(xs),len(xs[0])))
    for i, x in enumerate(xs):
        new_xs[i] = np.dot(p, x)
    return p, new_xs


def CameraDLP(originali, slike):
    if(len(originali) != len(slike)):
        raise ValueError("Potreban je isti broj tacaka i slika tacaka")
    if(len(originali) < 6):
        raise ValueError("Potrebno je barem 6 korespodencija")
    for t, tp in zip(map(len, originali), map(len, slike)):
        if(t != 4):
            raise ValueError("Originali moraju imati 4 dimenzije")
        if(tp != 3):
            raise ValueError("Slike moraju imati 3 dimenzije")
    p_originali, norm_originali = normalizacija_dlp(originali)
    p_slike, norm_slike = normalizacija_dlp(slike)
    _, _, vt = np.linalg.svd(korespodencije(norm_originali, norm_slike)
                                           ,full_matrices=True)
    v = vt[-1]
    T = gen_matrix([v[:4], v[4:8], v[8:]])
    T = np.dot(np.linalg.inv(p_slike), T)
    T = np.dot(T, p_originali)
    T /= T[0][0]
    return T


if __name__ == "__main__":
    n = 3
    print("Broj indeksa: 353/2020")
    print("n =", n)
    print("Matrica T:")
    T = gen_matrix([[5, -1-2*n, 3, 18-3*n]
                   ,[0, -1, 5, 21]
                   ,[0, -1, 0, 1]])

    print(T)
    print()

    print("Testiranje funkcije ParametriKamere:")
    K, A, C = ParametriKamere(T)
    print("Matrica K:")
    print(K)
    print("Matrica A:")
    print(A)
    print("Vektor C:")
    print(C)
    print()

    print("Testiranje funkcije CameraDLP:")
    originali = [[460, 280, 250, 1]
                ,[50, 380, 350, 1]
                ,[470, 500, 100, 1]
                ,[380, 630, 50*n, 1]
                ,[30*n, 290, 0, 1]
                ,[580, 0, 130, 1]]

    slike = [[288, 251, 1]
            ,[79, 510, 1]
            ,[470, 440, 1]
            ,[520, 590, 1]
            ,[365, 388, 1]
            ,[365, 20, 1]]


    originali, slike = gen_matrix(originali), gen_matrix(slike)

    print("ORIGINAL\t\t->\tSLIKA")
    for t, tp in zip(originali, slike):
        print("{}\t->\t{}".format(t, tp))

    T = CameraDLP(originali, slike)
    print("Matrica T:")
    print(T)
    # for t, tp in zip(originali, slike):
    #     test = np.dot(T, t)
    #     test /= test[-1]
    #     print(t, tp, test)
