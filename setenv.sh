#!/bin/bash

export LD_LIBRARY_PATH="$(find $PWD/.venv/ -type d -regex '.*nvidia.*lib$' -printf '%p:')"

