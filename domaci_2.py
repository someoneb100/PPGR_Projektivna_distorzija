try: import cv2
except:
    print("Treba instalirati cv2 da bi program radio. Da biste ga instalirali pokrenite sledecu komandu u terminalu:")
    print("python3 -m pip install opencv-python")
    exit(1)
try: import numpy as np
except:
    print("Treba instalirati cv2 da bi program radio. Da biste ga instalirali pokrenite sledecu komandu u terminalu:")
    print("python3 -m pip install numpy")
    exit(1)
from math import sin, cos, sqrt
from sys import argv

#######################################################################################
#
# DEKLARISANJE TIPOVA TACKA I PRESLIKAVANJE I PREOPTERECIVANJE OPERATORA ZA NJIH
#
#######################################################################################

class Tacka:
    def __init__(self, *args):
        x = None
        if(len(args) == 2):
            self.v = np.array(list(args) + [1], dtype = np.float)
        elif(len(args) == 3):
            self.v = np.array(list(args), dtype = np.float)
        else:
            self.v = np.copy(args[0].v if issubclass(args[0].__class__, Tacka) else args[0])

    def __mul__(self, othr):
       if(not issubclass(othr.__class__, Tacka)):
           raise ValueError("Drugi operator mnozenja nije tacka!")
       return Tacka(np.cross(self.v, othr.v))

    def __neg__(self):
        return Tacka(self.v*(-1,-1,1))

    def __call__(self):
        h0, h1, h2 = self.v
        if(h2 == 0):
            return(float('inf'), float('inf'))
        return int(round(h0/h2)), int(round(h1/h2))

    def __str__(self):
        return str(self.v)

    def dist(self, othr):
        if(not issubclass(othr.__class__, Tacka)):
            raise ValueError("Argument za racunanje rastojanja nije tacka!")
        if(self.v[2] == 0 or othr.v[2] == 0):
            return float('inf')
        x1, y1 = self()
        x2, y2 = othr()
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Preslikavanje:
    def __init__(self, *args):
        if(len(args) == 9):
            self.m = np.array([args[:3], args[3:6], args[6:]], subok=False, dtype=np.float, ndmin=2)
        elif(len(args) == 3):
            self.m = np.array(list(args), subok=False, dtype=np.float, ndmin=2)
        else:
            self.m = np.copy(args[0].m if issubclass(args[0].__class__, Preslikavanje) else args[0])

    def __call__(self, t):
        if(not issubclass(t.__class__, Tacka)):
            raise ValueError("Nije prosledjena Tacka, ne moze se preslikati!")
        return Tacka(np.dot(self.m, t.v.reshape(3,1)).transpose()[0])

    def __mul__(self, othr):
        if(not issubclass(othr.__class__, Preslikavanje)):
            raise ValueError("Drugi operator kompozicije nije preslikavanje!")
        return Preslikavanje(np.dot(self.m, othr.m))

    def __invert__(self):
        return Preslikavanje(np.linalg.inv(self.m))

    def __str__(self):
        return str(self.m)

    def scale(self, othr=None, r = 0):
        if(not issubclass(othr.__class__, Preslikavanje) and othr is not None):
            raise ValueError("Skalira se u odnosu na preslikavanje!")
        elif(issubclass(othr.__class__, Preslikavanje)):
            for i, red in enumerate(othr.m):
                for j, val in enumerate(red):
                    if(val != 0 and self.m[i][j] != 0):
                        self.m *= val/self.m[i][j]
                        break
        else:
            self.m /= self.m[0][0]
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


#######################################################################################
#
# DEFINISANJE ALGORITAMA ZA PRONALAZENJE PROJEKTIVNIH PRESLIKAVANJA
#
#######################################################################################


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



#######################################################################################
#
# POMOCNE FUNKCIJE ZA RAD SA FOTOGRAFIJAMA
#
#######################################################################################


def izaberi_tacku(slika):
    img, ime = slika
    cv2.namedWindow(ime, cv2.WINDOW_NORMAL)
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

def ukloni_distorziju(img, originali):
    img_h, img_w = img.shape[:2]
    w = round((tacke[0].dist(tacke[1]) + tacke[2].dist(tacke[3]))/2)
    h = round((tacke[1].dist(tacke[2]) + tacke[3].dist(tacke[0]))/2)
    slike = [Tacka(round((img_w-w)/2), round((img_h-h)/2)), Tacka(round(img_w - w/2), round((img_h-h)/2)),
             Tacka(round(img_w - w/2), round(img_h - h/2)), Tacka(round((img_w-w)/2), round(img_h - h/2))]
    p = naivni(originali, slike)
    return cv2.warpPerspective(img, p.m, (img.shape[1], img.shape[0]))

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



#######################################################################################
#
#           MAIN
#
#######################################################################################

if __name__ == "__main__":
    if(len(argv) == 1):
        print("Program se pokrece sa sledecim argumentima komandne linije:")
        print("\t--test-primer [--unos-tacaka]")
        print("\t\tza pokretanje test primera za prve tri stavke domaceg [opciono da korisnik unese proizvoljne tacke]")
        print("\t--uklanjanje-distorzije <slika>")
        print("\t\tza uklanjanje projektivne distorzije sa slike")
        print("\t--panorama <slika> <slika>")
        print("\t\tza pravljenje panorame od dve slike")
        exit(0)

    if(len(argv) == 3 and argv[1] == "--test-primer" and argv[2] == "--unos-tacaka"):
        input("Unosite tacke redom original pa slika. Unose se samo u obliku x1 x2 x3. Prve 4 unete tacke se koriste u naivnom algoritmu. (ENTER da nastavite dalje)")
        originali, slike = list(), list()
        while(True):
            ulaz = input("Unesite original: ").split(" ")
            originali.append(Tacka(float(ulaz[0]), float(ulaz[1]), float(ulaz[2])))
            ulaz = input("Unesite njegovu sliku: ").split(" ")
            slike.append(Tacka(float(ulaz[0]), float(ulaz[1]), float(ulaz[2])))
            if(len(originali) < 4):
                input("Potrebno je jos tacaka. (ENTER da nastavite dalje)")
            else:
                if(input("Ako zelite da zavrsite unos tacaka, ukucajte 'ne': ") == "ne"): break

        print("Koriscene tacke (originali --> slike):")
        for o, s in zip(originali, slike):
            print(o, "-->", s, sep="\t")

        p_naivno = naivni(originali[:4], slike[:4])
        p_naivno.scale(r = 14)
        print("Matrica preslikavanja naivnim algoritmom (zaokruzeno na 14 decimala):")
        print(p_naivno)

        p_dlt = dlt(originali,slike)
        p_dlt.scale(r = 14)
        print("Matrica preslikavanja DLT algoritmom:")
        print(p_dlt)

        p_dlt_norm = normalizovani_dlt(originali,slike)
        p_dlt_norm.scale(r = 14)
        print("Matrica preslikavanja DLT algoritmom uz normalizaciju:")
        print(p_dlt_norm)

        exit(0)


    if(len(argv) == 2 and argv[1] == "--test-primer"):
        inicijalno_preslikavanje = Preslikavanje(0,3,5,4,0,0,-1,-1,6)
        print("Inicijalno preslikavanje:")
        print(inicijalno_preslikavanje)

        inicijalne_tacke = [(-3, 2, 1), (-2, 5, 2), (1, 0, 3), (-7, 3, 1),
                            (2, 1, 2), (-1, 2, 1), (1, 1, 1)]
        originali = [Tacka(np.array(t, dtype=np.float)) for t in inicijalne_tacke]
        slike = [inicijalno_preslikavanje(t) for t in originali]

        print("Koriscene tacke (originali --> slike --> pokvarene (ako ima)):")
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
        p_dlt_obican_koord_rez.scale(othr=inicijalno_preslikavanje if len(argv) == 2 else None, r=6)
        print(p_dlt_obican_koord_rez)
        print("Matrica DLT normalizovano skalirana sa originalnom zaokruzena na 6 decimala:")
        p_dlt_norm_koord_rez = ~promena_koordinata * p_dlt_norm_koord * promena_koordinata
        p_dlt_norm_koord_rez.scale(othr=inicijalno_preslikavanje if len(argv) == 2 else None, r=6)
        print(p_dlt_norm_koord_rez)

        exit(0)

    if(len(argv) == 3 and argv[1] == "--uklanjanje-distorzije"):
        redosled = {0:"gore levo", 1:"gore desno", 2:"dole desno", 3:"dole levo"}
        slika = cv2.imread(argv[2]), argv[2]
        tacke = list()

        while(len(tacke) < 4):
            input("Izaberite tacku koja zelite da bude " + redosled[len(tacke)] + ". (ENTER da nastavite dalje)")
            tacke.append(izaberi_tacku(slika))

        normalizovana_slika = ukloni_distorziju(slika[0], tacke)
        cv2.namedWindow('Normalizovana slika', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Normalizovana slika', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Normalizovana slika', normalizovana_slika)
        cv2.waitKey()
        cv2.imwrite('Normalizovana_slika_za_PPGR.jpg', normalizovana_slika)
        print("Slika sacuvana pod imenom 'Normalizovana_slika_za_PPGR.jpg'!")
        exit(0)

    if(len(argv) == 4 and argv[1] == "--panorama"):
        slike = tuple((cv2.imread(a), a) for a in argv[2:])
        tacke = dict((i[1], list()) for i in slike)
        input("Slike ce se jedna po jedna pojavljivati, birajte uparene tacke. (ENTER da nastavite dalje)")
        broj_tacaka = 0
        while(True):
            for i in slike:
                tacke[i[1]].append(izaberi_tacku(i))
            broj_tacaka += 1
            if(broj_tacaka >= 4):
                ulaz = input("Zelite li jos tacaka? napisite 'ne' ako ne zelite, u suprotnom pritisnite enter: ")
                if(ulaz) == "ne":
                    break
            else:
                input("Potrebno je jos tacaka! Pritisnite enter da biste uneli jos jedan par.")

        p = normalizovani_dlt(tacke[slike[0][1]], tacke[slike[1][1]])
        panorama = spoji(slike[1][0], slike[0][0], p.m)
        cv2.namedWindow('Panorama', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Panorama', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Panorama', panorama)
        cv2.waitKey()
        cv2.imwrite('Panorama_za_PPGR.jpg', panorama)
        print("Panorama sacuvana pod imenom 'Panorama_za_PPGR.jpg'!")
        exit(0)

    print("Program je pozvan sa losiim opcijama, pozvati bez argumenata za uputstva!")
    exit(1)
