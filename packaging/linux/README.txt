Kera for Linux (x64)
====================

1. Unpack this folder anywhere.
2. Run ./kera in it.

A page opens in your browser and Kera measures this computer by itself: how fast the
SSD, the memory and the graphics chip are. It takes about a minute. Press Ctrl+C in the
terminal, or "Quit Kera" on the page, to stop it.

Graphics chips are used through Vulkan, so the Vulkan driver for your graphics chip must
be installed (Mesa's RADV for AMD, the vendor driver for NVIDIA).

Besides chatting, Kera makes pictures (./kera image), music (./kera music), speech (./kera speak),
short videos (./kera video), 3D models from a picture or a description (./kera 3d) and sound
effects (./kera sfx). Their models are your own download: each command's --help says which
files, where they come from, and their licences.

Kera keeps its files in ~/.local/share/kera (or $KERA_HOME if set).

To teach a model your own examples (fine-tuning), run ./kera finetune --help. It needs Python and
the packages in finetune/requirements.txt.

Kera runs only on this computer. It doesn't send anything anywhere.
