This is a simple 3D rasterizer written in Python and Numba.  It is based on the tinyrenderer tutorials [link].  Normal mapping is supported, shadows are yet to come.  Numba is used for performance, however performance was the main goal.  The software emulates a shader like pipeline, which is not optimal, but it is fun.

The code is also simply terrible.  One of these days I will refactor it, but it is low on my list of things I want to do.  If I ever were to write any sort of 3D application, it would not be with a python rasterizer.  It is however fun for experimenting.  Numba has limitations, for example, it is difficult to profile jitted code.
