# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# Created by 2LazyDevs
# Helper for Package Resolver System for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.
# helper.py

import os
import sys
from pathlib import Path

# Get absolute path to the SAPL root directory
SAPL_ROOT = os.path.abspath(os.path.dirname(__file__))
SYS_LIB_PATH = os.path.join(SAPL_ROOT, "libs")

# Add SAPL libs dir to sys.path if not already present
if SYS_LIB_PATH not in sys.path:
    sys.path.insert(0, SYS_LIB_PATH)

def resolve_include(pkg_name: str) -> str:
    return str(Path("SAPL", "include", pkg_name))

def resolve_import(pkg_name: str) -> str:
    return str(Path(os.getcwd(), pkg_name))
