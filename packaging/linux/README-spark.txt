Kera for the NVIDIA DGX Spark (Linux ARM64, CUDA)
=================================================

For the DGX Spark and the other GB10 machines (Dell, HP, ASUS, Acer, Gigabyte, Lenovo), with
DGX OS or another Linux with NVIDIA's CUDA 13 driver and libraries.

1. Unpack this folder anywhere on the internal SSD (models load fastest from it).
2. Run ./kera in it.

A page opens in your browser and Kera measures this computer by itself: how fast the SSD,
the memory and the graphics chip are. It takes about a minute.

On a Spark without a screen: ./kera --no-browser prints the address (http://127.0.0.1:8341/
unless that port is taken). From your own computer, forward that same port number and open it:
    ssh -L 8341:localhost:8341 you@your-spark
then open http://localhost:8341/ in your browser. Use the same number on both sides: the page
only answers at the port Kera printed.
Press Ctrl+C in the terminal, or "Quit Kera" on the page, to stop it.

The graphics chip is used through CUDA, with the CUDA libraries DGX OS comes with (cuBLAS 13).
If Kera says it found no graphics chip, check that they're installed (nvidia-smi works, and
ls /usr/local/cuda/lib64/libcublas.so.13 finds the library) and run ./kera setup again.

On a Spark, Kera:
- treats the 128 GB as shared by the CPU and the graphics chip, and lets a model use about
  110 GB of it;
- reads a model into memory in whole rather than memory-mapping it (faster on the Spark);
- says what would free more memory (no desktop, vm.swappiness=1, DGX OS's KHO update) but
  changes no system settings itself.

Pictures (./kera image), music (./kera music), video (./kera video), 3D models (./kera 3d)
and sound effects (./kera sfx) use the graphics chip too. Speech: ./kera speak. Their models
are your own download: each command's --help says which files, where they come from, and
their licences.

Kera keeps its files in ~/.local/share/kera (or $KERA_HOME if set).

To teach a model your own examples (fine-tuning), run ./kera finetune --help. It needs Python and
the packages in finetune/requirements.txt.

Kera runs only on this computer. It doesn't send anything anywhere.
