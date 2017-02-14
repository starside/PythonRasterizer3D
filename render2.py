import pygame
from pygame.locals import *
import math

import numpy as np
import pygame.surfarray as surfarray
import numba
from numba import jit
from numba import jitclass
import cProfile
import math

def surfdemo_show(array_img, zbuffer, name):
    "displays a surface, waits for user to continue"
    screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
    surfarray.blit_array(screen, array_img)
    array_img[:, :] = 0
    pygame.display.flip()
    pygame.display.set_caption(name)
    zbuffer[:,:] = ninf

@jit(nopython=True)
def setpix(x,y, color, img):
    w,h = img.shape[0:2]
    img[x,h-y] = color

@jit(nopython=True)
def setpix2(x,y, color, img, w, h):
    img[x,h-y,:] = color[:]


def line1(pt1, pt2, color,img):
    x1,y1 = pt1
    x2,y2 = pt2
    for t in np.linspace(0,1,100):
        x = round(x1*(1-t) + x2*t)
        y = round(y1*(1-t) + y2*t)
        setpix(x,y,color,img)

def line2(pt1, pt2, color,img):
    x0,y0 = pt1
    x1,y1 = pt2
    for x in np.linspace(x0,x1, np.abs(x1-x0)+1):
        t = 1.0*(x-x0)/(1.0*(x1-x0))
        y = y0*(1-t) + y1*t
        setpix(x,round(y),color,img)

@jit(nopython = True)
def line3(x0,y0,x1,y1, color,img):
    steep = False

    if np.abs(y1-y0) > np.abs(x1 - x0):
        x0,y0 = y0, x0  #Transpose line
        x1, y1,=y1, x1
        steep = True

    if x0 > x1:
        x0,x1 = x1, x0
        y0,y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    derror2 = abs(dy)*2
    error2 = 0

    y = y0

    for x in range(x0,x1+1):
        if steep:
            setpix(y,x,color,img)
        else:
            setpix(x,y,color,img)
        error2 += derror2
        if error2 > dx:
            y += 1 if y1>y0 else -1
            error2 -= dx*2

def loadObj(fname):
    verts = []
    faces = []
    texcoords = []
    texmap = []
    f = open(fname, "r")
    for fl in f:
        pd = fl.split()
        if len(pd) > 1:
            if pd[0] == "v":
                verts.append( [float(x) for x in  pd[1:4]] )
            if pd[0] == "f":
                rf = pd[1:4]
                faces.append( [ int(x.split('/')[0]) for x in rf] )
                texmap.append( [ int(x.split('/')[1]) for x in rf] )
            if pd[0] == "vt":
                texcoords.append( [float(x) for x in  pd[1:4]] )
    f.close()
    carr = np.ones( ( len(verts),4 ), dtype=np.float32) #compact vertex array
    for i,e in enumerate(verts):
        carr[i,0:3] = e
    return carr, np.array(faces, dtype=np.int32), texcoords, texmap

@jit
def drawWireFrame(verts, faces, color, img):
    width, height = img.shape[0:2]
    for face in faces:
        for i in range(3):
            v0 = verts[ face[i] -1 ]
            v1 = verts[ face[(i+1)%3] -1]

            x0 = (v0[0]+1.0)*width/2.0
            y0 = (v0[1]+1.0)*height/2.0 
            x1 = (v1[0]+1.0)*width/2.0 
            y1 = (v1[1]+1.0)*height/2.0 
            line3(int(x0), int(y0), int(x1), int(y1), color, img)

@jit
def drawFilledTris(verts, faces, color, img):
    width, height = img.shape[0:2]
    for face in faces:
        tvl = []
        for i in range(3):
            v0 = verts[ face[i] -1 ]
            x0 = (v0[0]+1.0)*width/2.0
            y0 = (v0[1]+1.0)*height/2.0 
            tvl.append( [x0,y0])
        if color[0] < 0:
            col = np.round(np.random.rand(3)*255)
            triangle( np.array(tvl, dtype=np.float32), width, height, img, col)
        else:
            triangle( np.array(tvl, dtype=np.float32), width, height, img, color)

@jit(nopython=True)
def crossProd(u,v,res):
    res[0] = u[1]*v[2] - u[2]*v[1]
    res[1] = u[2]*v[0] - u[0]*v[2]
    res[2] = u[0]*v[1] - u[1]*v[0]
    res[3] = 0

#@jit(nopython=True)
def calcNormals(verts, faces, resnorm):
    """
    Takes vertecies, faces, and writes normals to resnorm
    :param verts:
    :param faces:
    :param resnorm:
    :return:
    """
    vnorm = np.array([0,0,0,0], dtype=np.float32)
    numFaces = len(faces)
    for j in range(numFaces):
        face = faces[j]
        v1 = verts[face[0]-1]
        v2 = verts[face[1]-1]
        v3 = verts[face[2]-1]
        va = v2 - v1
        vb = v3 - v1
        crossProd(va, vb, vnorm)
        vmag = math.sqrt(vnorm[0]*vnorm[0] + vnorm[1]*vnorm[1] + vnorm[2]*vnorm[2])
        for i in range(3):
            resnorm[j-1, i] = vnorm[i] / vmag;

def calcTexCoords(faces, texture, texcoords, texmap):
    alltexcoords = []
    tw, th, td = np.shape(texture)  # texture width,height,depth
    for j,face in enumerate(faces):
        tcl = [] #taxture coord list
        for i in range(3):
            tcl.append( [texcoords[ texmap[j][i]-1 ][0]*tw, texcoords[ texmap[j][i]-1 ][1]*th] )
        alltexcoords.append(tcl)
    return np.array(alltexcoords, dtype=np.float32)

@jit
def drawLitTris(verts, faces, normals, lightdir, img, zbuff, texture, alltexcoords):
    width, height = img.shape[0:2]
    tvl = np.zeros((3,3), dtype=np.float32)
    for j,face in enumerate(faces):
        for i in range(3):
            tvl[i, :] = verts[face[i] - 1][0:3]
        dotprod = np.dot(lightdir, normals[j-1])
        vdp = round(dotprod*255)
        color = np.array([vdp,vdp,vdp], dtype=np.int32)
        if not (color[0] >= 0 and color[1] >= 0 and color[2] >= 0):
            dotprod = 0.0
        triangleZBuffer(tvl, width, height, img, dotprod, zbuff, texture, alltexcoords[j])


"Takes vector AB, AC, and PA, then returns barycentric coodinates"
@jit('void(f4[:], f4[:], f4[:], f4[:])', nopython=True)
def barycentric(result, ab, ac, pa):
    result[2] = ab[0]*ac[1] - ab[1]*ac[0]
    u = (ac[0]*pa[1] - ac[1]*pa[0])/result[2]
    v = (pa[0]*ab[1] - ab[0]*pa[1])/result[2]
    result[0] = 1.0 -u -v
    result[1] = u
    result[2] = v
    
"""  Takes 2d array of vertices as verts  """
#'void(i4[:,:], i4, i4, i4[:,:], i4[:] )', 
@jit(nopython=True)
def triangle(verts, width, height, img, color):
    #calculate bounding box of triangle
    bboxmin = [width-1, height-1]
    bboxmax = [0 ,0]
    clamp = [width-1, height-1]
    for i in range(3):
        for j in range(2):
            bboxmin[j] = max(0, min(bboxmin[j],verts[i,j]) )
            bboxmax[j] = min(clamp[j], max(bboxmax[j], verts[i,j]) )
    #check if point lies in triangle
    ab = verts[1] - verts[0]
    ac = verts[2] - verts[0]
    pa = np.array([0,0], dtype=np.float32)
    res = np.array([0,0,0], dtype=np.float32)
    for x in range(bboxmin[0],bboxmax[0]+1):
        for y in range(bboxmin[1], bboxmax[1]+1):
            pa[0] = verts[0][0] - x
            pa[1] = verts[0][1] - y
            barycentric(res, ab, ac, pa)
            if res[0] >= 0 and res[1] >= 0 and res[2] >= 0:
                setpix2(x,y, color, img, width, height)


@jit(nopython=True)
def triangleZBuffer(verts, width, height, img, color, zbuffer,texture, texcoord):
    #calculate bounding box of triangle
    bboxmin = [width-1, height-1]
    bboxmax = [0 ,0]
    clamp = [width-1, height-1]
    for i in range(3):
        for j in range(2):
            bboxmin[j] = max(0, min(bboxmin[j],verts[i,j]) )
            bboxmax[j] = min(clamp[j], max(bboxmax[j], verts[i,j]) )
    #check if point lies in triangle
    ab = verts[1] - verts[0]
    ac = verts[2] - verts[0]
    pa = np.array([0,0], dtype=np.float32)
    res = np.array([0,0,0], dtype=np.float32)
    clr = np.array([255,0,0], dtype=np.int32)
    for x in range(bboxmin[0],bboxmax[0]+1):
        for y in range(bboxmin[1], bboxmax[1]+1):
            pa[0] = verts[0][0] - x
            pa[1] = verts[0][1] - y
            barycentric(res, ab, ac, pa)
            if res[0] >= 0 and res[1] >= 0 and res[2] >= 0:
                #test Z buffer
                zv = res[0]*verts[0,2] + res[1]*verts[1,2] + res[2]*verts[2,2]
                if zv > zbuffer[x,y]:
                    zbuffer[x,y] = zv
                    #calculate texture color
                    tx = int(round(res[0]*texcoord[0,0]+res[1]*texcoord[1,0]+res[2]*texcoord[2,0]))
                    ty = int(round(res[0]*texcoord[0,1]+res[1]*texcoord[1,1]+res[2]*texcoord[2,1]))
                    clr[:] = texture[tx,ty]*color
                    setpix2(x,y, clr, img, width, height)

f4 = np.float32
ninf = -1024.0

@jit
def transMat(vertx, matrix, result):
    np.dot(vertx, np.transpose(matrix), result)

@jit(nopython=True)
def dot_py(A, B, C):
    """ C = A*B matrix multiplication
        m, n = A.shape
        p = B.shape[1]
        C = np.zeros((m,p))
    """
    m, n = A.shape
    p = B.shape[1]
    for i in range(m): #Zero out C
        for j in range(p):
            C[i,j] = 0

    for i in range(0,m): #Do mult
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j]

@jit(nopython=True)
def dot_py_transpose(A, B, C):
    """ C = A*B' matrix multiplication
        m, n = A.shape
        p = B.shape[1]
        C = np.zeros((m,p))
    """
    m, n = A.shape
    p = B.shape[0]
    for i in range(m): #Zero out C
        for j in range(p):
            C[j,i] = 0

    for i in range(0,m): #Do mult
        for j in range(0,p):
            for k in range(0,n):
                C[j,i] += A[i,k]*B[j,k]


@jit(nopython=True)
def divbyz(vertx):
    w, h = vertx.shape[0], vertx.shape[1]
    for i in range(w):
        for j in range(h):
            vertx[i,j] = vertx[i,j]/vertx[i,h-1]

def projectionMatrix(cam):
    tranm = np.array(((1, 0, 0, 0),
                      (0, 1, 0, 0),
                      (0, 0, 1, 0),
                      (0, 0, -1.0/cam, 1)), dtype=np.float32)
    return tranm

def viewPortMatrix(x, y, w, h, d):
    tranm = np.array(((w/2.0,       0,      0,      x+w/2.0),
                      (0    ,   h/2.0,      0,      y+h/2.0),
                      (0    ,       0,  d/2.0,      d/2.0)  ,
                      (0    ,       0,      0,      1        )), dtype=np.float32)
    return tranm

def lookAtMatrix(eye, center, up):
    z = (eye-center)
    x = np.cross(up, z)
    y = np.cross(z, x)
    z = z / np.dot(z, z)
    x = x / np.dot(x, x)
    y = y / np.dot(y, y)
    minv = np.eye(4, dtype=np.float32)
    tr = np.eye(4, dtype=np.float32)
    for i in range(3):
        minv[0][i] = x[i]
        minv[1][i] = y[i]
        minv[2][i] = z[i]
        tr[i][3] = -center[i]
    return np.dot(minv, tr)

def rotationMatrix(t):
    s = np.sin(t)
    c = np.cos(t)
    return np.array( ((1,0,0,0),
                      (0,c,s,0),
                      (0,-s,c,0),
                      (0,0,0,1)),dtype=np.float32)

spec = [
    ('faces', numba.int32[:,:]),
    ('verts', numba.float32[:,:]),
    ('normals', numba.float32[:,:]),
]
@jitclass(spec)
class GouradShader(object):
    def __init__(self, faces, verts, normals):
        self.faces = faces
        self.verts = verts
        self.normals = normals

    def vertex(self, iface, nthvert):
        return iface[0]

def main():
    pygame.init()
    striped = np.zeros((800, 800, 3), np.int32)
    zbuffer = np.zeros((800, 800), dtype=np.float32)
    zbuffer[:,:] = ninf
    width, height, depth = np.shape(striped)
    #striped[:] = (255, 0, 0)
    #striped[::3,::3] = (0, 255, 255)
    #Load texture
    ttext = pygame.image.load('african_head.tga')
    texture = np.zeros((ttext.get_width(), ttext.get_height(), 3), np.int32)
    texw,texh, texd = np.shape(texture)
    for x in range(ttext.get_width()):
        for y in range(ttext.get_height()):
            texture[x,texh-1-y,:] = ttext.get_at((x,y))[0:3]

    white = (255,255,255)
    red = (255,0,0)
    green = (0,255,0)

    """  Load the model from file """
    vertx, faces, texcoords, texmap = loadObj('african_head.obj')
    numFaces, _fw = np.shape(faces)

    """  Calculate Matrices """
    viewportmatrix = viewPortMatrix(0,0,width,height,1.0)
    eye = np.array([0,0,1], dtype=np.float32)
    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    normals = np.zeros((numFaces, 4), dtype=np.float32)
    transformedVerts = np.zeros(np.shape(vertx), dtype=np.float32)
    pw, ph = np.shape(transformedVerts)
    """  Calculate model normals """
    calcNormals(vertx, faces, normals)
    dgs = GouradShader(faces, vertx, normals)
    """  Calculate Texture Coordinates """
    alltexcoords = calcTexCoords(faces, texture, texcoords, texmap)
    
    #drawWireFrame(vertx, faces, white, striped)
    #drawFilledTris(vertx, faces, (-1,0,0), striped)
    light = np.array([1, 0, 1, 0], dtype=np.float32)
    angle = 0.0
    tran1 = np.zeros((4,4), dtype=np.float32)
    tranm = np.zeros((4,4), dtype=np.float32)
    while True:
        if angle > 2.0*np.pi:
            angle = 0.0
        eye[0] = 1.2*np.sin(angle)
        eye[2] = 2.0*np.cos(angle)
        eye[1] = 1
        light[0] = np.sin(angle)
        light[2] = np.cos(angle)
        """ Move Camera"""
        modelview = lookAtMatrix(eye, center, up)
        """ Multiply the matrices """
        dist = (eye-center)
        dsc = np.sqrt(np.dot(dist, dist))
        projmatrix = projectionMatrix(dsc)
        dot_py(projmatrix, modelview, tran1)
        dot_py(viewportmatrix, tran1, tranm)
        """ Transform the vertices in vertz, store in transformedVerts """
        dot_py_transpose(tranm, vertx, transformedVerts)
        #transMat(vertx, tranm, transformedVerts)
        """ Divide by the 4th component to do perspective """
        divbyz(transformedVerts)
        """ Render the model """
        drawLitTris(transformedVerts, faces, normals, light, striped, zbuffer, texture, alltexcoords)
        surfdemo_show(striped, zbuffer, 'striped')

        angle += 0.1

if __name__ == "__main__":
    main()

