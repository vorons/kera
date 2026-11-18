Kera for Mac (Apple Silicon)
============================

1. Move Kera.app to your Applications folder.
2. Open it.

A page opens in your browser and Kera measures this Mac by itself: how fast the SSD,
the memory and the graphics chip are. It takes about a minute and you don't need to do
anything. Kera has no window of its own: to stop it, press "Quit Kera" on its page.

Kera isn't signed by Apple yet, so the first time macOS will refuse to open it. Then:
open System Settings, Privacy & Security, scroll down, and click "Open Anyway" next to
Kera. You only do this once.

Besides chatting, Kera makes pictures (kera image), music (kera music), speech (kera speak),
short videos (kera video), 3D models from a picture or a description (kera 3d) and sound
effects (kera sfx), run in Terminal as Kera.app/Contents/MacOS/kera image and so on. Their
models are your own download: each command's --help says which files, where they come
from, and their licences.

Kera keeps its files in ~/Library/Application Support/Kera.

To teach a model your own examples (fine-tuning), run Kera.app/Contents/MacOS/kera finetune --help in
Terminal. It needs Python and MLX-LM (pip install mlx-lm).

Kera runs only on this Mac. It doesn't send anything anywhere.
