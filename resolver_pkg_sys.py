
# SAPL (Simply Advanced Programming Language)
# This file is part of the SAPL project, which is a simple yet powerful programming language.
# This is licensed under the 2LazyDevs OpenSource License.
# A copy of the license can be found at https://github.com/2-LazyDevs/LICENSE/LICENSE
# resolver_pkg_sys.py
# Created by 2LazyDevs
# Package Resolver System for SAPL (Simply Advanced Programming Language)
# For now, this is written in Python but later will be written in SAPL itself.

import os
from typing import Dict
from helper import resolve_import, resolve_include

class PackageResolver:
    def __init__(self, working_dir: str):
        self.working_dir = working_dir  # Where custom packages are loaded from
        self.cache: Dict[str, str] = {}

    def resolve(self, pkg_name: str, is_include: bool = False) -> str:
        """
        Resolves and returns the contents of a package file.
        If `is_include` is True, resolves a system package.
        If False, resolves a user-defined package in working directory.
        """

        key = f"{'include' if is_include else 'import'}:{pkg_name}"
        if key in self.cache:
            return self.cache[key]

        try:
            if is_include:
                path = resolve_include(pkg_name)
            else:
                path = resolve_import(pkg_name, self.working_dir)

            with open(path, "r", encoding="utf-8") as f:
                contents = f.read()

            self.cache[key] = contents
            return contents

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[SAPL Resolver] {e}")