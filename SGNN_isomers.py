# -*- coding: utf-8 -*-

import os
import shutil
import sys
import subprocess
import shutil
import time
import re
from openbabel import pybel


def SetupIsomers(Isomers, settings):
    for iso in Isomers:
        atoms = ReadIsomer(iso.BaseName, settings)
        iso.Atoms = atoms
    return Isomers


def ReadIsomer(name, settings):
    AN_to_atom = {
        1 = 'H',
        6 = 'C'
    }
    isomer = next(pybel.readfile("sdf", name + ".sdf"))
    atoms_list = []
    for atom in pybel.Molecule(isomer).atoms:
        if atom.atomicnum in AN_to_atom:
            atoms_list.append(AN_to_atom[atom.atomicnum])
        else:
            atoms_list.append('_')

    return atoms_list