Conway's Game of Life (nVidia CUDA on Linux Framebuffer)
========================================================

Conway's Game of Life demonstration for nVidia GPUs on Linux systems.
This program generates a random initial state and then uses the GPU threads to update the playing field. For each generation the output is mapped to Linux framebuffer (`/dev/fb0`).
Since this is only a quick demonstration of GPU programming, the code isn't particularly clean and does the bare minimum.

For more informations about Conway's Game of Life (and for performance comparisons with the same algorithm run on CPU), see the slow C# version: <https://github.com/HashakGik/Life-C-sharp>.

Compilation
-----------

- Install the `nvcc` compiler (in Slackware Linux is included in the [`cudatoolkit`](https://slackbuilds.org/repository/14.2/development/cudatoolkit) Slackbuild) and all its dependencies
- A simple Makefile is provided, so simply run `make` on a terminal

Execution
---------

In a real terminal (i.e. **outside** the X Window System, e.g. after hitting `Ctrl+Alt+F2`) run `./Life`.