// Finding the model files already on this computer (spec 4a step 5), so the app can show each one's
// predicted speed, and use models where they are (the owner asked: never move or copy them).
#pragma once

#include <string>
#include <vector>

namespace kera {

// Folders to look in: the Kera folder's models/, folders the user added (model_folders.json in the
// Kera folder), and where common apps keep GGUF files (LM Studio, llama.cpp, Hugging Face).
std::vector<std::string> model_folders();

// Add a folder to model_folders.json. Returns false if it isn't a folder.
bool add_model_folder(const std::string & folder);

// Every .gguf model in those folders (and their subfolders): split models once (their first part),
// and no vision projectors (mmproj files). Paths are absolute.
std::vector<std::string> find_models(const std::vector<std::string> & folders);

} // namespace kera
