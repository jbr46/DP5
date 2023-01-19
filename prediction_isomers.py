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
        atoms = ReadIsomer(iso.BaseName)
        iso.Atoms = atoms
    return Isomers


def ReadIsomer(name):
    atoms_list = []
    mol = MolFromMolFile(name + '.sdf')
    mol = rdmolops.AddHs(mol)
    for atom in mol.GetAtoms():
        atoms_list.append(atom.GetSymbol())

    return atoms_list

def CheckComposition(Isomers):
    iso = Isomers[0]
    name = iso.BaseName
    allowed = set(['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl'])
    mol = MolFromMolFile(name + '.sdf')
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return False
    return True