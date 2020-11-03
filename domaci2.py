#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:23:05 2020

@author: Petar Đorđević
"""

import numpy as np
from math import sin, cos, sqrt

#definisem klase Tacka i Preslikavanje radi lakseg rada 
class Tacka:
    def __init__(self, *args):
        self.v = (np.array(list(args) + [1], dtype = np.float) 
            if len(args) == 2 else 
            np.copy(args[0]))

    def __mul__(self, othr):
       if(not issubclass(othr.__class__, Tacka)): raise ValueError("Drugi operator mnozenja nije tacka!")
       return Tacka(np.cross(self.v, othr.v))
    def __neg__(self):
        return Tacka(self.v*(-1,-1,1))
    def __call__(self):
        h0, h1, h2 = self.v
        if(h2 == 0): return(float('inf'), float('inf'))
        return int(round(h0/h2)), int(round(h1/h2))
    def __str__(self):
        return str(self.v)
    def pokvari(self):
        return Tacka(np.array(list(map(round, self.v)), dtype = np.float))
    def dist(self, othr):
        if(not issubclass(othr.__class__, Tacka)): raise ValueError("Argument za racunanje rastojanja nije tacka!")
        if(self.v[2] == 0 or othr.v[2] == 0): return float('inf')
        return sqrt((self.v[0]/self.v[2] - othr.v[0]/othr.v[2])**2 + (self.v[1]/self.v[2] - othr.v[1]/othr.v[2])**2)
        
    
class Preslikavanje:
    def __init__(self, *args):
        self.m = np.array(list(args[0]) if len(args) == 1 else [args[:3], args[3:6], args[6:]], 
                                   subok=False, dtype=np.float, ndmin=2)

        
    def __call__(self, t):
        if(not issubclass(t.__class__, Tacka)): raise ValueError("Nije prosledjena Tacka, ne moze se preslikati!")
        return Tacka(np.dot(self.m, t.v.reshape(3,1)).transpose()[0]);
    
    def __mul__(self, othr):
        if(not issubclass(othr.__class__, Preslikavanje)): raise ValueError("Drugi operator kompozicije nije preslikavanje!")
        return Preslikavanje(np.dot(self.m, othr.m))
    def __invert__(self):
        return Preslikavanje(np.linalg.inv(self.m))
    def __str__(self):
        return str(self.m)
    def scale(self, othr=None, r = 0):
        if(othr is None): self.m /= self.m[2][2]
        if(not issubclass(othr.__class__, Preslikavanje)): raise ValueError("Skalira se u odnosu na preslikavanje!")
        for i, red in enumerate(othr.m):
            for j, val in enumerate(red):
                if(val != 0 and self.m[i][j] != 0):
                    self.m *= val/self.m[i][j]
                    break
        for i in range(3):
            for j in range(3):
                self.m[i][j] = round(self.m[i][j], r)
                
                
class Translacija(Preslikavanje):
    def __init__(self, *args):
        if(len(args) == 2): super().__init__(1, 0, args[0], 0, 1, args[1], 0, 0, 1)
        elif(not issubclass(args[0].__class__, Tacka)): raise ValueError("Nije prosledjena Tacka, ne moze se translirati!")
        elif(args[0].v[2] == 0): raise ValueError("Tacka se nalazi u beskonacnosti, ne moze se translirati!")
        else: super().__init__(1, 0, args[0].v[0] / args[0].v[2] , 0, 1, args[0].v[1] / args[0].v[2], 0, 0, 1)
        
class Rotacija(Preslikavanje):
    def __init__(self, phi):
        super().__init__(cos(phi), -sin(phi), 0, sin(phi), cos(phi), 0, 0, 0, 1)
        
class Skaliranje(Preslikavanje):
    def __init__(self, *args):
        if(len(args) == 2): super().__init__(args[0], 0, 0, 0, args[1], 0, 0, 0, 1)
        elif(not issubclass(args[0].__class__, Tacka)): raise ValueError("Nije prosledjena Tacka, ne moze se skalirati!")
        elif(args[0].v[2] == 0): raise ValueError("Tacka se nalazi u beskonacnosti, ne moze se skalirati!")
        else: super().__init__(args[0].v[0] / args[0].v[2], 0, 0 , 0, args[0].v[1] / args[0].v[2], 0, 0, 0, 1)


#naivni algoritam Tacka[4][2] --> Preslikavanje
def naivni_preslikavanje(A, B, C, D):
    mat = np.array([A.v] + [B.v] + [C.v] , subok=True, dtype=np.float, ndmin=2)
    p1 = np.linalg.inv(mat.transpose())
    greek = np.dot(p1, D.v).reshape(3,1)
    return Preslikavanje((greek * mat).transpose())

def naivni(originali, slike):
    A, B, C, D = originali
    Ap, Bp, Cp, Dp = slike
    p1 = naivni_preslikavanje(Ap, Bp, Cp, Dp)
    p2 = naivni_preslikavanje(A, B, C, D)
    return p1 * ~p2

#DLT algoritam slika Tacka[n][2] --> Preslikavanje
def korespodencija(T, Tp):
    rez = np.zeros((2,9))
    rez[0][3:6] = -Tp.v[2] * T.v
    rez[0][6:] = Tp.v[1] * T.v
    rez[1][:3] = Tp.v[2] * T.v
    rez[1][6:] = -Tp.v[0] * T.v
    return rez

def matrica_korespodencija(originali, slike):
    rez = list()
    for o, s in zip(originali, slike):
        rez += list(korespodencija(o, s))
    return np.array(rez) 

def dlt(o, s):
    _, _, vt = np.linalg.svd(matrica_korespodencija(o, s), full_matrices=True)
    return Preslikavanje(np.array([vt[-1][:3], vt[-1][3:6], vt[-1][6:]], subok=False, dtype=np.float, ndmin=2))


#DLT algoritam sa normalizacijom slika Tacka[n][2] --> Preslikavanje
def normalizuj(tx):
    x = sum(p.v[0]/p.v[2] for p in tx) / len(tx)
    y = sum(p.v[1]/p.v[2] for p in tx) / len(tx)
    centroid = Tacka(x, y)
    prosek = sum([centroid.dist(t) for t in tx])/len(tx)
    skala = sqrt(2)/prosek
    t = Translacija(-centroid)
    s = Skaliranje(skala, skala)
    return t*s

def normalizovani_dlt(o, s):
    t = normalizuj(o)
    tp = normalizuj(s)
    p = dlt([t(x) for x in o], [tp(x) for x in s])
    return ~tp * p * t



#main za test primer
if __name__ == "__main__":
    inicijalno_preslikavanje = Preslikavanje(0,3,5,4,0,0,-1,-1,6)
    print("Inicijalno preslikavanje:")
    print(inicijalno_preslikavanje)
    
    inicijalne_tacke = [(-3, 2, 1), (-2, 5, 2), (1, 0, 3), (-7, 3, 1), 
                        (2, 1, 2), (-1, 2, 1), (1, 1, 1)]
    originali = [Tacka(np.array(t, dtype=np.float)) for t in inicijalne_tacke]
    slike = [inicijalno_preslikavanje(t) for t in originali]
    print("Koriscene tacke (originali --> slike --> pokvarene):")
    for o, s in zip(originali, slike):
        print(o, "-->", s, sep="\t")
    print("Pokvarimo poslednje preslikavanje na drugoj decimali:")
    stara, nova = Tacka(slike[-1].v), Tacka(slike[-1].v)
    nova.v[0] += 0.02
    slike[-1] = nova
    print(stara, "-->", nova, sep="\t")
    
    p_naivno = naivni(originali[:4], slike[:4])
    p_naivno.scale(p_naivno, 14)
    print("Matrica preslikavanja naivnim algoritmom (zaokruzeno na 14 decimala):")
    print(p_naivno)
    
    p_dlt = dlt(originali,slike)
    print("Matrica preslikavanja DLT algoritmom:")
    print(p_dlt)
    
    print("Skaliracemo drugu matricu prema prvoj i zaokruziti na 6 decimala:")
    p_dlt.scale(othr = p_naivno, r = 6)
    print(p_dlt)
    
    p_dlt_norm = normalizovani_dlt(originali,slike)
    print("Matrica preslikavanja DLT algoritmom uz normalizaciju:")
    print(p_dlt_norm)
    
    print("Skaliracemo trecu matricu prema prvoj i zaokruziti na 6 decimala:")
    p_dlt_norm.scale(othr = p_dlt, r = 6)
    print(p_dlt_norm)
    
    print("Preslikavanje za proveru invarijantnosti u odnosu na promenu koordinata:")
    promena_koordinata = Preslikavanje(0, 1, 2, -1, 0, 3, 0, 0, 1)
    print(promena_koordinata)
    print("Nove tacke:")
    
    originali_koordinate = [promena_koordinata(t) for t in originali]
    slike_koordinate = [promena_koordinata(t) for t in slike]
    for o, s in zip(originali_koordinate, slike_koordinate):
        print(o, "-->", s, sep="\t")
        
    p_dlt_obican_koord = dlt(originali_koordinate, slike_koordinate)
    p_dlt_norm_koord = normalizovani_dlt(originali_koordinate, slike_koordinate)
    print("Matrica DLT skalirana sa originalnom zaokruzena na 6 decimala:")
    p_dlt_obican_koord_rez = ~promena_koordinata * p_dlt_obican_koord * promena_koordinata
    p_dlt_obican_koord_rez.scale(othr=inicijalno_preslikavanje, r=6)
    print(p_dlt_obican_koord_rez)
    print("Matrica DLT normalizovano skalirana sa originalnom zaokruzena na 6 decimala:")
    p_dlt_norm_koord_rez = ~promena_koordinata * p_dlt_norm_koord * promena_koordinata
    p_dlt_norm_koord_rez.scale(othr=inicijalno_preslikavanje, r=6)
    print(p_dlt_norm_koord_rez)
    
    
    
    
    
    
    
    