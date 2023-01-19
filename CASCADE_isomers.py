# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import sys
import time

from openbabel import pybel
from rdkit.Chem import MolFromMolFile, rdmolops


def SetupIsomers(Isomers, settings):
    for iso in Isomers:
        atoms = ReadIsomer(iso.BaseName, settings)
        iso.Atoms = atoms
    return Isomers


def ReadIsomer(name, settings):
    atoms_list = []
    mol = MolFromMolFile(name + '.sdf')
    mol = rdmolops.AddHs(mol)
    for atom in mol.GetAtoms():
        atoms_list.append(atom.GetSymbol())
    return atoms_list