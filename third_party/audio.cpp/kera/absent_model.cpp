// Added by Kera (docs/UPSTREAM_PATCHES.md, "audio.cpp"); not part of upstream audio.cpp.
//
// Kera's copy of audio.cpp keeps only the sound-effect models' sources (and the two voice-activity models every build
// links). Each model left out still gets its build target, made from this empty file, so the build's other lines that
// name it keep working; the models Kera builds (AUDIOCPP_MODELS) never use it.
