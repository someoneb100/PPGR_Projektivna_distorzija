import ctypes
import os
import sys
import math

sys.path.append('..')

import pyglet
from pyglet.gl import *

from pywavefront import visualization
from pywavefront import material
from pywavefront import mesh
import pywavefront

import numpy as np
from domaci import *

if(len(sys.argv) > 1 and sys.argv[1] == "--slerp"):
    opcija_slerp = True
    print("SLERP")
else:
    opcija_slerp = False
    print("LERP")
# Create absolute path from this module
file_abspath = os.path.join(os.path.dirname(__file__), '55z27frcahz4-P911GT/Porsche_911_GT2.obj')

a_start, b_start, c_start = 45, 45, -20
a_end, b_end, c_end = 0, 0, 0

def rad2deg(angle):
    return (angle*360)/(2*pi)

def deg2rad(angle):
    return (2*angle*pi)/360

fi_start, teta_start, xi_start = map(deg2rad, [a_start, b_start, c_start])
fi_end, teta_end, xi_end = map(deg2rad, [a_end, b_end, c_end])
p1, angle1 =axisAngle(euler2a(fi_start, teta_start, xi_start))
p2, angle2 = axisAngle(euler2a(fi_end, teta_end, xi_end))

q1 = axisAngle2q(p1, angle1)
q2 = axisAngle2q(p2, angle2)
q = q1

x_translation, y_translation = -5.0, -5.0
meshes = pywavefront.Wavefront(file_abspath, collect_faces=True, create_materials=True)
window = pyglet.window.Window(resizable=True)
lightfv = ctypes.c_float * 4


@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(x_translation, y_translation, -15.0)

    p, angle = q2axisAngle(q)
    angle = (angle*360)/(2*pi)

    glRotatef(angle, p[0], p[1], p[2])

    glEnable(GL_LIGHTING)

    visualization.draw(meshes)


def LERP(q1, q2, t):
    q = (1-t)*q1 + t*q2
    return q / intenzitet(q)

def SLERP(q1, q2, t):
    cosfi = np.dot(q1, q2)/(intenzitet(q1)*intenzitet(q2))
    if cosfi < 0:
        cosfi *= -1
        q1 *= -1
    if cosfi > 0.95:
        return LERP(q1, q2, t)
    fi = acos(cosfi)
    q = (sin(fi*(1-t))/sin(fi)) * q1 + (sin(fi*t)/sin(fi)) * q2
    return q / intenzitet(q)



def update(dt):
    global q1, q2, q
    global x_translation, y_translation
    global opcija_slerp
    if(x_translation <= 5 and y_translation <= 5):
        speed = 3
        x_translation += speed * dt
        y_translation += speed * dt
        t = (x_translation + 5.0) / 10
        if(opcija_slerp):
            q = SLERP(q1, q2, t)
        else:
            q = LERP(q1, q2, t)


pyglet.clock.schedule(update)
pyglet.app.run()
