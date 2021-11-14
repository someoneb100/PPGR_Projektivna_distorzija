import numpy as np

class Tacka:
	def __init__(self, x, y = None):
		self.homogeno = np.array((x,y, 1), dtype = np.float) if y != None else x

	def __mul__(self, othr):
		return Tacka(np.cross(self.homogeno, othr.homogeno))

	def nehomogeno(self):
		h0, h1, h2 = self.homogeno
		return int(round(h0/h2)), int(round(h1/h2))

#funlcija za domaci
def nevidljivo(A, B, C, D, A1, B1, C1):
	p = (A*B) * (A1*B1)
	q = (D*A) * (C1*B1)
	return (p*A1) * (q*C1)

if __name__ == "__main__":
	A, B = Tacka(71, 378), Tacka(502, 503)
	C, D = Tacka(758, 179), Tacka(534, 147)
	A1, B1 = Tacka(116, 560), Tacka(497, 713)
	C1 = Tacka(157, 183)
	D1 = nevidljivo(A, B, C, D, A1, B1, C1)

	print("Koordinate:", D1.nehomogeno())
