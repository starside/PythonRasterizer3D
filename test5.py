# GL interoperability example, by Peter Berrington.
# Draws a rotating teapot, using cuda to invert the RGB value
# each frame

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.GL.ARB.pixel_buffer_object import *


import numpy, sys, time
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule

#this is all munged together from the CUDA SDK postprocessGL example.

initial_size = 1025,768
current_size = initial_size
animate = True
enable_cuda = True
window = None     # Number of the glut window.
time_of_last_draw = 0.0
time_of_last_titleupdate = 0.0
frames_per_second = 0.0
frame_counter = 0
output_texture = None # pointer to offscreen render target
(source_pbo, dest_pbo, cuda_module, invert,
 pycuda_source_pbo, pycuda_dest_pbo) = [None]*6
heading,pitch,bank = [0.0]*3

def create_PBOs(w,h):
    global dest_pbo, pycuda_dest_pbo
    num_texels = w*h
    dest_pbo = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dest_pbo)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, num_texels*4,None, GL_DYNAMIC_COPY)
    pycuda_dest_pbo = cuda_gl.BufferObject(long(dest_pbo))

def destroy_PBOs():
    global dest_pbo, pycuda_dest_pbo
    for pbo in [dest_pbo]:
        glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
        glDeleteBuffers(1, long(pbo));
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    dest_pbo,pycuda_dest_pbo = [None]*4

def create_texture(w,h):
    global output_texture
    output_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, output_texture)
    # set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

def destroy_texture():
    global output_texture
    glDeleteTextures(output_texture);
    output_texture = None

def init_gl():
    Width, Height = current_size
    glClearColor(0.1, 0.1, 0.5, 1.0)
    glDisable(GL_DEPTH_TEST)
    glViewport(0, 0, Width, Height)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, Width/float(Height), 0.1, 10.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glEnable(GL_LIGHT0)
    red   = ( 1.0, 0.1, 0.1, 1.0 )
    white = ( 1.0, 1.0, 1.0, 1.0 )
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  red  )
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white)
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 60.0)

def resize(Width, Height):
    global current_size
    current_size = Width, Height
    glViewport(0, 0, Width, Height)        # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, Width/float(Height), 0.1, 10.0)

def do_tick():
    global time_of_last_titleupdate, frame_counter, frames_per_second
    if ((time.clock () * 1000.0) - time_of_last_titleupdate >= 1000.):
        frames_per_second = frame_counter                   # Save The FPS
        frame_counter = 0  # Reset The FPS Counter
        szTitle = "%d FPS" % (frames_per_second )
        glutSetWindowTitle ( szTitle )
        time_of_last_titleupdate = time.clock () * 1000.0
    frame_counter += 1

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global animate, enable_cuda
    # If escape is pressed, kill everything.
    if args[0] == '\033':
        print('Closing..')
        destroy_PBOs()
        destroy_texture()
        exit()
    elif args[0] == 'a':
        print('toggling animation')
        animate = not animate
    elif args[0] == 'e':
        print('toggling cuda')
        enable_cuda = not enable_cuda

def idle():
    global heading, pitch, bank
    if animate:
        heading += 0.2
        pitch   += 0.6
        bank    += 1.0

    glutPostRedisplay()

def display():
    try:
        render_scene()
        if enable_cuda:
            process_image()
            display_image()
        glutSwapBuffers()
    except:
        from traceback import print_exc
        print_exc()
        from os import _exit
        _exit(0)

def process(width, height):
    """ Use PyCuda """
    bsize = 32
    w = 200
    h = 100
    grid_dimensions   = ((w+bsize-1)//bsize, (h+bsize-1)//bsize)

    dest_mapping   = pycuda_dest_pbo.map()

    verts = numpy.array([200,200,300,200,200,900], dtype=numpy.float32)
    for i in range(1000):
        triangle(verts, width, height, dest_mapping, invert)


    cuda_driver.Context.synchronize()
    dest_mapping.unmap()

def process_image():
    """ copy image and process using CUDA """
    global pycuda_source_pbo,source_pbo,current_size, dest_pbo
    image_width, image_height = current_size
    # tell cuda we are going to get into these buffers

    # run the Cuda kernel
    process(image_width, image_height)
    # blit convolved texture onto the screen
    # download texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, long(dest_pbo))
    glBindTexture(GL_TEXTURE_2D, output_texture)

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image_width, image_height,
                    GL_BGRA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

def display_image():
    """ render a screen sized quad """
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glMatrixMode( GL_MODELVIEW)
    glLoadIdentity()
    glViewport(0, 0, current_size[0], current_size[1])
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, 0.5)
    glEnd()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glDisable(GL_TEXTURE_2D)
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)


def render_scene():
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)# Clear Screen And Depth Buffer
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity ()      # Reset The Modelview Matrix
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(heading, 1.0, 0.0, 0.0)
    glRotatef(pitch  , 0.0, 1.0, 0.0)
    glRotatef(bank   , 0.0, 0.0, 1.0)
    glViewport(0, 0, current_size[0],current_size[1])
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glutSolidTeapot(1.0)
    do_tick()#just for fps display..
    return True

def triangle(vertsp, width, height, dest_mapping, invert):
    #calculate bounding box of triangle
    verts = vertsp.reshape(3,2)
    bboxmin = [width-1, height-1]
    bboxmax = [0 ,0]
    clamp = [width-1, height-1]
    for i in range(3):
        for j in range(2):
            bboxmin[j] = max(0, min(bboxmin[j],verts[i,j]) )
            bboxmax[j] = min(clamp[j], max(bboxmax[j], verts[i,j]) )

    bsize = 32
    w = int(round(bboxmax[0] - bboxmin[0]))
    h = int(round(bboxmax[1] - bboxmin[1]))
    x0 = int(round(bboxmin[0]))
    y0 = int(round(bboxmin[1]))
    grid_dimensions = ((w + bsize - 1) // bsize, (h + bsize - 1) // bsize)
    cuda_driver.memcpy_htod(vertex_local_gpu, vertsp)

    invert.prepared_call(grid_dimensions, (bsize, bsize, 1),
                         dest_mapping.device_ptr(), vertex_local_gpu, w * h, width, x0, y0, w, h)


def main():
    global window, cuda_module, cuda_gl, cuda_driver, invert
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(*initial_size)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow(b"PyCuda GL Interop Example")
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyPressed)
    glutSpecialFunc(keyPressed)
    init_gl()

    # create texture for blitting to screen
    create_texture(*initial_size)

    #setup pycuda gl interop
    import pycuda.gl.autoinit
    import pycuda.gl
    cuda_gl = pycuda.gl
    cuda_driver = pycuda.driver

    cuda_module = SourceModule("""
    #include <stdio.h>

    __device__ void barycentric(float *result, float *ab, float *ac, float *pa){
        result[2] = ab[0]*ac[1] - ab[1]*ac[0];
        float u = (ac[0]*pa[1] - ac[1]*pa[0])/result[2];
        float v = (pa[0]*ab[1] - ab[0]*pa[1])/result[2];
        result[0] = 1.0 -u -v;
        result[1] = u;
        result[2] = v;
    }

    __global__ void invert(unsigned char *dest, float verts[3][2],  unsigned int n, unsigned int screenwidth, unsigned int x0, unsigned int y0, unsigned int w, unsigned int h)
    {
      int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
      int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
      int threads_in_block = blockDim.x * blockDim.y;
      //Since the image is RGBA we multiply the index 4.
      //We'll only use the first 3 (RGB) channels though
      int idx              = 4 * (threads_in_block * block_num + thread_num);
      int y = (idx/4)/w + y0;
      int x = (idx/4)%w + x0;

      //y = blockIdx.y*blockDim.y + threadIdx.y + 100;
      int idx2 = 4*(screenwidth*y + x);

      //ab = verts[1] - verts[0]
      //  ac = verts[2] - verts[0]
      float ab[2] = {verts[1][0] - verts[0][0] ,verts[1][1] - verts[0][1]};
      float ac[2] = {verts[2][0] - verts[0][0] ,verts[2][1] - verts[0][1]};
      float pa[2] = {0, 0};
      float res[3];

      if(idx < n*4) {
        pa[0] = verts[0][0] - x;
        pa[1] = verts[0][1] - y;
        barycentric(res, ab, ac, pa);
        if( res[0] >= 0 && res[1] >= 0 && res[2] >= 0) {
            dest[idx2  ] = 0;
            dest[idx2+1] = 0;
            dest[idx2+2] = 0;
        }
      }

    }
    """)

    global vertex_local_gpu
    vertex_local = numpy.zeros(3*2, dtype=numpy.float32)
    vertex_local_gpu = cuda_driver.mem_alloc(vertex_local.nbytes)

    invert = cuda_module.get_function("invert")
    # The argument "PP" indicates that the invert function will take two PBOs as arguments
    invert.prepare("PPiiiiii")

    # create source and destination pixel buffer objects for processing
    create_PBOs(*initial_size)

    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    print("Hit ESC key to quit, 'a' to toggle animation, and 'e' to toggle cuda")
    main()