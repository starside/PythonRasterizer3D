import pygame
from pygame.locals import *
import math

import numpy as np
import pygame.surfarray as surfarray
import numba
from numba import jit
from numba import jitclass
import math

def surfdemo_show(screen, array_img, zbuffer, name, index=None):
    "displays a surface, waits for user to continue"
    surfarray.blit_array(screen, array_img)
    if index is not None:
        pygame.image.save(screen, str(index) + ".png")
    array_img[:, :] = 0
    pygame.display.flip()
    pygame.display.set_caption(name)
    zbuffer[:,:] = ninf

@jit(nopython=True)
def setpix2(x,y, color, img, w, h):
    img[x,h-y,:] = color[:]

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
    return carr, np.array(faces, dtype=np.int32)-1, texcoords, texmap

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
        v1 = verts[face[0]]
        v2 = verts[face[1]]
        v3 = verts[face[2]]
        va = v2 - v1
        vb = v3 - v1
        crossProd(va, vb, vnorm)
        vmag = math.sqrt(vnorm[0]*vnorm[0] + vnorm[1]*vnorm[1] + vnorm[2]*vnorm[2])
        for i in range(3):
            resnorm[j, i] = vnorm[i] / vmag;
        resnorm[3] = 0

def calcGouradNormals(verts, normals, faces):
    #Calculate vertex normals
    rows,cols = np.shape(verts)
    vertnorms = np.zeros((rows, 4), dtype=np.float32)
    for j in range(len(faces)):
        face = faces[j]
        for i in range(3):
            v = face[i]
            vertnorms[v,:] += normals[j]
    for j in range(rows):
        vertnorms[j] = vertnorms[j] / math.sqrt(np.dot(vertnorms[j], vertnorms[j]))
    return vertnorms

def calcTexCoords(faces, texture, texcoords, texmap):
    alltexcoords = []
    tw, th, td = np.shape(texture)  # texture width,height,depth
    for j,face in enumerate(faces):
        tcl = [] #taxture coord list
        for i in range(3):
            tcl.append( [texcoords[ texmap[j][i]-1 ][0]*tw, texcoords[ texmap[j][i]-1 ][1]*th] )
        alltexcoords.append(tcl)
    return np.array(alltexcoords, dtype=np.float32)

"Takes vector AB, AC, and PA, then returns barycentric coodinates"
@jit('void(f4[:], f4[:], f4[:], f4[:])', nopython=True)
def barycentric(result, ab, ac, pa):
    result[2] = ab[0]*ac[1] - ab[1]*ac[0]
    u = (ac[0]*pa[1] - ac[1]*pa[0])/result[2]
    v = (pa[0]*ab[1] - ab[0]*pa[1])/result[2]
    result[0] = 1.0 -u -v
    result[1] = u
    result[2] = v
    
@jit(nopython=True)
def colorTriangleZBuffer(verts, width, height, img, color, zbuffer, shader):
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
                    #clr[:] = color[:]
                    if not shader.fragment(res, clr):
                        #clr[0:3] = zv
                        setpix2(x,y, clr, img, width, height)


f4 = np.float32
ninf = -1024.0

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
def vec_dot_py(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

@jit(nopython=True)
def vec_dot3_py(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@jit(nopython=True)
def dot_py_vecB(A, B, C):
    """ C = A*v matrix multiplication.  B is vector, C is vector
        m, n = A.shape
        p = B.shape[1]
        C = np.zeros((m,p))
    """
    m, n = A.shape
    for i in range(m): #Zero out C
        C[i] = 0

    for i in range(0,m): #Do mult
        for k in range(0,n):
            C[i] += A[i,k]*B[k]

    for i in range(0,m):
        C[i] = C[i] / C[3]

@jit(nopython=True)
def dot_py_vecB_nodiv(A, B, C):
    """ C = A*v matrix multiplication.  B is vector, C is vector
        m, n = A.shape
        p = B.shape[1]
        C = np.zeros((m,p))
    """
    m, n = A.shape
    for i in range(m): #Zero out C
        C[i] = 0

    for i in range(0,m): #Do mult
        for k in range(0,n):
            C[i] += A[i,k]*B[k]

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

@jit(nopython=True)
def normalize3(tn):
    nmag = math.sqrt(tn[0]*tn[0] + tn[1]*tn[1] + tn[2]*tn[2])
    for i in range(3):
        tn[i] = tn[i] / nmag

@jit(nopython=True)
def normalsFromMap(nmap, uv, n):
    """
    Takes a normal map in global coords (nmap), a (u,v) pair as len(uv)=2 array
    and a result stores in n
    :param nmap:
    :param uv:
    :param n:
    :return:
    """
    for i in range(3):
        n[i] = float(nmap[uv[0], uv[1], i]) / 255.0 - 0.5

@jit(nopython=True)
def texelFromMap(tmap, uv, scaler, offset, dest):
    """
    Takes a normal map in global coords (nmap), a (u,v) pair as len(uv)=2 array
    and a result stores in n
    :param nmap:
    :param uv:
    :param n:
    :return:
    """
    for i in range(3):
        dest[i] = min(int(round(tmap[uv[0], uv[1], i]*scaler)), 255)


spec = [
    ('faces', numba.int32[:,:]),
    ('verts', numba.float32[:,:]),
    ('normals', numba.float32[:,:]),
    ('matrix', numba.float32[:,:]),
    ('uniform_M', numba.float32[:,:]),
    ('uniform_MIT', numba.float32[:,:]),
    ('varying_intensity', numba.float32[:]),
    ('varying_uv', numba.float32[:,:]),
    ('lightdir', numba.float32[:]),
    ('uvcoords', numba.float32[:,:,:]),
    ('texture', numba.int32[:,:,:]),
    ('normalmap', numba.int32[:,:,:]),
    ('specmap', numba.uint8[:,:]),
    ('luv', numba.float32[:]),
]
@jitclass(spec)
class GouradShader(object):
    def __init__(self, faces, verts, normals):
        self.faces = faces
        self.verts = verts
        self.normals = normals
        self.varying_intensity = np.ones(3, dtype=np.float32)
        self.varying_uv = np.zeros((2,3), dtype=np.float32)
        self.lightdir = np.zeros(4, dtype=np.float32)
        self.luv = np.zeros(2, dtype=np.float32)

    def setMatrix(self, matrix):
        self.matrix = matrix

    def setNormalMatrix(self, M, MIT):
        """
        This is the inverse of Projection*Modelview
        :param matrix:
        :return:
        """
        self.uniform_M = M
        self.uniform_MIT = MIT

    def setLightdir(self, lightdir):
        self.lightdir = lightdir
        normalize3(self.lightdir)

    def setTextureMap(self, uvcoords):
        self.uvcoords = uvcoords

    def setTexture(self, tex):
        self.texture = tex

    def setNormalmap(self, tex):
        self.normalmap = tex

    def setSpecMap(self, tex):
        self.specmap = tex

    def vertex(self, iface, nthvert, res):
        v = self.verts[self.faces[iface, nthvert]]
        n = self.normals[self.faces[iface, nthvert]]
        self.varying_uv[0:2, nthvert] = self.uvcoords[iface, nthvert,0:2]
        self.varying_intensity[nthvert] = max(0, vec_dot_py(self.lightdir, n))
        dot_py_vecB(self.matrix, v, res)

    def reflected(self, r, n, l):
        c = vec_dot3_py(n, l)
        for i in range(3):
            r[i] = n[i]*2.0*c - l[i]

    def fragment(self, bar, color):
        uv = [0.0, 0.0]
        n = [0.0, 0.0, 0.0, 0.0]
        tn = [0.0, 0.0, 0.0, 0.0]
        r = [0.0, 0.0, 0.0, 0.0]
        intensity = vec_dot3_py(bar, self.varying_intensity)
        dot_py_vecB_nodiv(self.varying_uv, bar, uv)
        luv = [int(round(uv[0])), int(round(uv[1]))]
        normalsFromMap(self.normalmap, luv, n)
        dot_py_vecB_nodiv(self.uniform_MIT, n, tn)
        normalize3(tn)
        self.reflected(r, tn, self.lightdir)
        spec = math.pow(max(0.0, vec_dot3_py(r, [0.0, 0.0, 1.0])), self.specmap[luv[0], luv[1]])    #specular lighting
        diff = max(0,vec_dot3_py(tn, self.lightdir))
        ambient = 5.0
        intensity = diff + spec   #lightdir must be normalized
        texelFromMap(self.texture, luv, intensity, ambient, color)
        return False


specd = [
    ('faces', numba.int32[:,:]),
    ('verts', numba.float32[:,:]),
    ('matrix', numba.float32[:,:]),
    ('varying_verts', numba.float32[:,:]),
    ('lightdir', numba.float32[:]),
]
@jitclass(specd)
class DepthShader(object):
    def __init__(self, faces, verts):
        self.faces = faces
        self.verts = verts
        self.varying_verts = np.zeros((4, 4), dtype=np.float32)
        self.lightdir = np.zeros(4, dtype=np.float32)

    def setMatrix(self, matrix):
        self.matrix = matrix

    def setLightdir(self, lightdir):
        self.lightdir = lightdir
        normalize3(self.lightdir)

    def vertex(self, iface, nthvert, res):
        v = self.verts[self.faces[iface, nthvert]]
        dot_py_vecB(self.matrix, v, res)
        self.varying_verts[0:4, nthvert] = res[0:4]

    def fragment(self, bar, color):
        r = [0.0, 0.0, 0.0, 0.0]
        dot_py_vecB(self.varying_verts, bar, r)
        color[0:3] = int(round(r[2]))   #Copy zbufffer to image buffer
        return False


def main():
    pygame.init()
    screenwidth = 600
    screenheight = 800
    ibuffer = np.zeros((screenwidth, screenheight, 3), np.int32)
    zbuffer = np.zeros((screenwidth, screenheight), dtype=np.float32)
    zbuffer[:,:] = ninf
    width, height, depth = np.shape(ibuffer)
    screen = pygame.display.set_mode(ibuffer.shape[:2], 0, 32)

    #Load texture
    ttext = pygame.image.load('african_head.tga')
    ntext = pygame.image.load('african_head_nm.tga')
    tspecmap = pygame.image.load('african_head_spec.tga')
    texture = np.zeros((ttext.get_width(), ttext.get_height(), 3), np.int32)
    normalm = np.zeros((ttext.get_width(), ttext.get_height(), 3), np.int32)
    specmap = np.zeros((ttext.get_width(), ttext.get_height()), np.int8)
    texw,texh, texd = np.shape(texture)
    for x in range(ttext.get_width()):
        for y in range(ttext.get_height()):
            texture[x,texh-1-y,:] = ttext.get_at((x,y))[0:3]
            normalm[x, texh - 1 - y, :] = ntext.get_at((x, y))[0:3]
            specmap[x, texh -1 - y] = tspecmap.get_at((x,y))[0]

    white = (255,255,255)
    red = (255,0,0)
    green = (0,255,0)

    """  Load the model from file """
    vertx, faces, texcoords, texmap = loadObj('african_head.obj')
    numFaces, _fw = np.shape(faces)

    """  Calculate Matrices """
    viewportmatrix = viewPortMatrix(0,0,width,height,255.0)
    eye = np.array([0,0,1], dtype=np.float32)
    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    normals = np.zeros((numFaces, 4), dtype=np.float32)
    transformedVerts = np.zeros(np.shape(vertx), dtype=np.float32)
    pw, ph = np.shape(transformedVerts)
    """  Calculate model normals """
    calcNormals(vertx, faces, normals)
    gourandNormals = calcGouradNormals(vertx, normals, faces)
    shader = GouradShader(faces, vertx, gourandNormals)  #create shader
    depthshader = DepthShader(faces, vertx)
    """  Calculate Texture Coordinates """
    alltexcoords = calcTexCoords(faces, texture, texcoords, texmap)
    shader.setTextureMap(alltexcoords)
    shader.setTexture(texture)
    shader.setNormalmap(normalm)
    shader.setSpecMap(specmap)
    
    light = np.array([1, 0, 1, 0], dtype=np.float32)
    shader.setLightdir(light)
    angle = 0.0
    tran1 = np.zeros((4,4), dtype=np.float32)
    tranm = np.zeros((4,4), dtype=np.float32)
    frame = None # Set to 0 if you want to dump out images for the gif.
    # gif encoded with ffmpeg -f image2 -i %d.png video.gif
    while True:
        if angle >= 2.0*np.pi:
            angle = 0.0
            frame = None
        eye[0] = 1.5*np.sin(angle)
        eye[1] = 0.0
        eye[2] = 1.5*np.cos(angle)
        light[0] = 0
        light[1] = 2
        light[2] = 1.5
        """ Move Camera"""
        modelview = lookAtMatrix(eye, center, up)
        """ Multiply the matrices """
        dist = (eye-center)
        dsc = np.sqrt(np.dot(dist, dist))
        projmatrix = projectionMatrix(dsc)
        dot_py(projmatrix, modelview, tran1)
        dot_py(viewportmatrix, tran1, tranm)

        """ Render the model """
        normalize3(light)
        shader.setLightdir(light)
        shader.setMatrix(tranm)
        depthshader.setLightdir(light)
        depthshader.setMatrix(tranm)
        MIT = np.linalg.inv(np.transpose(modelview))
        shader.setNormalMatrix(modelview, MIT)
        screenCoords = np.zeros((3,4), dtype=np.float32)
        col1 = np.array([220, 220, 220], dtype=np.int32)

        for i in range(faces.shape[0]):
            for j in range(3):
                shader.vertex(i,j, screenCoords[j])
            colorTriangleZBuffer(screenCoords, width, height, ibuffer,col1,zbuffer, shader)

        if frame is not None:
            frame += 1
        surfdemo_show(screen, ibuffer, zbuffer, 'ibuffer', index=frame)
        angle += 0.1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

if __name__ == "__main__":
    main()

