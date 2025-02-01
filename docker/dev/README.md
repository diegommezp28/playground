# Development Environment

This is a development environment for the project. It is based on Docker and Docker Compose and we use VSCode devcontainers to work with it.

It is meant to standardize everything a developer needs in order to start working in this project. 
Things like usefull extensions, linters, formatters, Poetry, Python, Jupyter, CMake, etc will all be unified in this environment.

We will use Poetry to manage the Python Development dependencies and Docker to manage the environment.

The default Python virtual env will be created by Poetry inisde this folder with the name `venv`. 

It will include an editable install of the project packages, and stuff like the jupyter kernel for notebook development.

Ideally, notebooks should all refer to this `venv` kernel, so that we can have a consistent environment for development.

In the rare case when a notebook needs to use a different kernel, it should also be managed by poetry and included in its separate `venv` folder, where the said notebook can reference its kernel.