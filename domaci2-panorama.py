#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:35:11 2020

@author: Petar Đorđević
"""

from domaci2 import Tacka, normalizovani_dlt

from sys import argv
import numpy as np
try: import cv2
except:
    print("Treba instalirati cv2 da bi program radio. Da biste ga instalirali pokrenite sledecu komandu u terminalu:")
    print("python3 -m pip install opencv-python")
    exit(1)

def spoji(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

   
def napravi_sliku(ime):
    img = cv2.imread(ime)
    cv2.namedWindow(ime, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(ime, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)      
    cv2.setMouseCallback(ime, sacuvaj_pixel)
    global kliknuto
    global nadjena_tacka
    kliknuto = False
    nadjena_tacka = None
    while (not kliknuto):
        cv2.imshow(ime, img)
        if cv2.waitKey(2) & 0xFF == 27:       
            break
    cv2.destroyAllWindows()
    return nadjena_tacka
    

def sacuvaj_pixel(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN:
        global kliknuto
        global nadjena_tacka
        kliknuto = True
        nadjena_tacka = Tacka(x, y)
  
if __name__ == '__main__':
    #img = np.zeros((512, 512, 3), np.uint8)
    if(len(argv) != 3):
        print("Treba kao argument komandne linije navesti 2 slike!")
        exit(1)
    slike =  tuple(argv[1:])
    tacke = dict((i, list()) for i in slike)
    
    broj_tacaka = 0
    while(True):
        for i in slike:
            tacke[i].append(napravi_sliku(i))
        broj_tacaka += 1
        if(broj_tacaka >= 4):
            ulaz = input("Zelite li jos tacaka? napisite 'ne' ako ne zelite, u suprotnom pritisnite enter: ")
            if(ulaz) == "ne":
                break
        else:
            input("Potrebno je jos tacaka! Pritisnite enter da biste uneli jos jedan par.")
    p = normalizovani_dlt(tacke[slike[0]], tacke[slike[1]])
    
    panorama = spoji(cv2.imread(slike[1]), cv2.imread(slike[0]), p.m)
    cv2.namedWindow('Panorama', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Panorama', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    cv2.imshow('Panorama', panorama)
    cv2.waitKey()
    cv2.imwrite('Panorama_za_PPGR.jpg', panorama)
    print("Panorama sacuvana pod imenom 'Panorama_za_PPGR.jpg'!")
    