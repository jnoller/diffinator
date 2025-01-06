# llamalog

A simple tool to analyze and correlate changes between two releases of Llama.cpp.

## Features

- Fetches release information using GitHub's API
- Compares changes between two specified versions
- Organizes changes into logical groups (server, ggml, etc.)
- Highlights important file changes (e.g., gguf-py/pyproject.toml)
- Provides a clear summary of differences between releases

## Description
A simple tool to look at at correlate the changes between two release of Llama.cpp. 

Using the llama.cpp github release API, given version A and version B, it will look at the release notes and build 
a list of changes that were made between the two versions and organize the changes into similiar groups (such as 
`server`, `ggml`, etc).

Additionally, it examines the files changes between the two highlighting changes to key files (such as 
gguf-py/pyproject.toml). If the file was changes, it will be highlighted in the output.



