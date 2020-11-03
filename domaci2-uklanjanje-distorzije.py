#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 14:34:15 2020

@author: Petar Đorđević
"""

from domaci2 import Tacka, naivni

from sys import argv
import numpy as np
try: import cv2
except:
    print("Treba instalirati cv2 da bi program radio. Da biste ga instalirali pokrenite sledecu komandu u terminalu:")
    print("python3 -m pip install opencv-python")
    exit(1)


def napravi_projekciju(img, originali):
    img_w, img_h = img.shape[:2]
    w = round((tacke[0].dist(tacke[1]) + tacke[2].dist(tacke[3]))/2)
    h = round((tacke[1].dist(tacke[2]) + tacke[3].dist(tacke[0]))/2)
    slike = [Tacka(round((img_w-w)/2), round((img_h-h)/2)), Tacka(round(img_w - w/2), round((img_h-h)/2)),
             Tacka(round((img_w-w)/2), round(img_h - h/2)), Tacka(round(img_w - w/2), round(img_h - h/2))]
    return naivni(originali, slike)



def napravi_sliku(ime):
    img = cv2.imread(ime)
    cv2.namedWindow(ime, cv2.WINDOW_AUTOSIZE)
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
    if(len(argv) != 2):
        print("Treba kao argument komandne linije navesti sliku!")
        exit(1)
    slika =  argv[1]
    tacke = list()

    redosled = {0:"gore levo", 1:"gore desno", 2:"dole desno", 3:"dole levo"}

    broj_tacaka = 0
    while(broj_tacaka < 4):
        input("Izaberite tacku koja zelite da bude " + redosled[broj_tacaka] + ". (ENTER da nastavite dalje)")
        tacke.append(napravi_sliku(slika))
        print(tacke[-1])
        broj_tacaka += 1

    img = cv2.imread(slika)
    p = napravi_projekciju(img, tacke)
    normalizovana_slika = cv2.warpPerspective(img, p.m, img.shape[:2])
    cv2.namedWindow('Normalizovana slika', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Normalizovana slika', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Normalizovana slika', normalizovana_slika)
    cv2.waitKey()
    cv2.imwrite('Normalizovana_slika_za_PPGR.jpg', normalizovana_slika)
    print("Slika sacuvana pod imenom 'Normalizovana_slika_za_PPGR.jpg'!")
