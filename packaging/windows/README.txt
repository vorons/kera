Kera for Windows
================

1. Unzip this folder anywhere, for example into your Documents.
2. Double-click kera.exe.

A page opens in your browser and Kera measures this computer by itself: how fast the
SSD, the memory and the graphics chip are. It takes about a minute and you don't need
to do anything. A black window also opens: that's Kera running. Close it to stop Kera.

If Windows says "Windows protected your PC": Kera isn't signed yet. Click "More info",
then "Run anyway".

Besides chatting, Kera makes pictures (kera image), music (kera music), speech (kera speak),
short videos (kera video), 3D models from a picture or a description (kera 3d) and sound
effects (kera sfx), in a terminal in this folder. Their models are your own download: each
command's --help says which files, where they come from, and their licences.

Kera keeps its files in %LOCALAPPDATA%\Kera. To keep them somewhere else, set the
environment variable KERA_HOME to that folder before starting Kera.

To teach a model your own examples (fine-tuning), run "kera finetune --help" in a terminal in
this folder. It needs Python (python.org) and the packages in finetune\requirements.txt.

Kera runs only on this computer. It doesn't send anything anywhere.
