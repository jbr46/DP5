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
    # Using openbabel
    # AN_to_atom = {
    #     1: 'H',
    #     6: 'C'
    # }
    # isomer = next(pybel.readfile("sdf", name + ".sdf"))
    # for atom in pybel.Molecule(isomer).atoms:
    #     if atom.atomicnum in AN_to_atom:
    #         atoms_list.append(AN_to_atom[atom.atomicnum])
    #     else:
    #         atoms_list.append('_')

    # Using rdkit
    mol = MolFromMolFile(name + '.sdf')
    mol = rdmolops.AddHs(mol)
    for atom in mol.GetAtoms():
        atoms_list.append(atom.GetSymbol())

    return atoms_list