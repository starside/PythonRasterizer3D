import pygame
from pygame.locals import *

import numpy as np
import pygame.surfarray as surfarray
from numpy import int32
from numba import jit
import cProfile
import math

def surfdemo_show(array_img, zbuffer, name):
    "displays a surface, waits for user to continue"
    screen = pygame.display.set_mode(array_img.shape[:2], 0, 32)
    surfarray.blit_array(screen, array_img)
    pygame.display.flip()
    pygame.display.set_caption(name)
    zbuffer[:,:] = ninf
    while 1:
        e = pygame.event.wait()
        if e.type == MOUSEBUTTONDOWN: break
        elif e.type == KEYDOWN and e.key == K_s:
            #pygame.image.save(screen, name+'.bmp')
            #s = pygame.Surface(screen.get_size(), 0, 32)
            #s = s.convert_alpha()
            #s.fill((0,0,0,255))
            #s.blit(screen, (0,0))
            #s.fill((222,0,0,50), (0,0,40,40))
            #pygame.image.save_extended(s, name+'.png')
            #pygame.image.save(s, name+'.png')
            #pygame.image.save(screen, name+'_screen.png')
            #pygame.image.save(s, name+'.tga')
            pygame.image.save(screen, name+'.png')
        elif e.type == QUIT:
            raise SystemExit()

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
    return carr, faces, texcoords, texmap

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

@jit
def drawLitTris(verts, faces, lightdir, img, zbuff, texture, texcoords, texmap):
    width, height = img.shape[0:2]
    for j,face in enumerate(faces):
        tvl = []
        tcl = [] #taxture coord list
        tw,th,td = np.shape(texture)  #texture width,height,depth
        for i in range(3):
            v0 = verts[ face[i] -1 ]
            x0 = (v0[0]+1.0)*width/2.0
            y0 = (v0[1]+1.0)*height/2.0 
            tvl.append( [x0,y0,v0[2] ])
            tcl.append( [texcoords[ texmap[j][i]-1 ][0]*tw, texcoords[ texmap[j][i]-1 ][1]*th] )
        texcoord = np.array(tcl, dtype=np.float32)
        v1 = np.array(verts[face[0]-1], dtype=np.float32)
        v2 = np.array(verts[face[1]-1], dtype=np.float32)
        v3 = np.array(verts[face[2]-1], dtype=np.float32)
        va = v2 - v1
        vb = v3 - v1
        vnorm = np.cross(va,vb)
        vmag = np.sqrt(np.dot(vnorm, vnorm))
        vnorm = vnorm / vmag
        dotprod = np.dot( lightdir, vnorm)
        vdp = round(dotprod*255)
        color = np.array([vdp,vdp,vdp], dtype=np.int32)
        if color[0] > 0 and color[1] > 0 and color[2] > 0:
            triangleZBuffer( np.array(tvl, dtype=np.float32), width, height, img, dotprod, zbuff,texture, texcoord)


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
    clr = np.array([255,0,0], dtype=int32)
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
def divbyz(vertx, w, h):
    for i in range(w):
        for j in range(h):
            vertx[i,j] = vertx[i,j]/vertx[i,h-1]

def main():
    pygame.init()
    striped = np.zeros((1024, 1024, 3), int32)
    zbuffer = np.zeros((1024, 1024), dtype=np.float32)
    zbuffer[:,:] = ninf
    width, height, depth = np.shape(striped)
    #striped[:] = (255, 0, 0)
    #striped[::3,::3] = (0, 255, 255)
    #Load texture
    ttext = pygame.image.load('african_head.tga')
    texture = np.zeros((ttext.get_width(), ttext.get_height(), 3), int32)
    texw,texh, texd = np.shape(texture)
    for x in range(ttext.get_width()):
        for y in range(ttext.get_height()):
            texture[x,texh-1-y,:] = ttext.get_at((x,y))[0:3]

    white = (255,255,255)
    red = (255,0,0)
    green = (0,255,0)

    vertx, faces, texcoords, texmap = loadObj('african_head.obj')

    proj = np.zeros(np.shape(vertx), dtype=np.float32)
    tranm = np.array( ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,-1,1)), dtype=np.float32)
    cam = 5.0
    tranm[3,2] = -1.0/cam
    transMat(vertx, tranm, proj)
    pw,ph = np.shape(proj)
    divbyz(proj, pw, ph)
    
    #drawWireFrame(vertx, faces, white, striped)
    #drawFilledTris(vertx, faces, (-1,0,0), striped)
    drawLitTris(proj[:,0:3], faces, np.array([0,0,1], dtype=np.float32), striped, zbuffer, texture, texcoords, texmap)
    surfdemo_show(striped, zbuffer, 'striped')

if __name__ == "__main__":
    main()

