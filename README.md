# SSSSS

This repository is an implementation of screen-space subsurface scattering(SSSSS). The repo is written in *Python* and
is based on libraries as is described in the *requirements.txt*.

Notice that the kernel used in this repository, which determines how translucent the rendered model looks, is fitted
using another program, which basically runs a Monte-Carlo simulation over a small square patch of area.

The result is supposed to look as follows.

![SSSSS Result](.\result.png)

Notice that the library *luisa* cannot yet be installed using command `pip install`, so if you run into any problem when
trying to run this project, you may wish to download the library from
its [GitHub site](https://github.com/LuisaGroup/LuisaCompute).
