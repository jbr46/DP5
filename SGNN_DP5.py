#!/usr/bin/env python

# -*- coding: utf-8 -*-

import subprocess
import os
import time
import glob
import shutil

def SetupNMRPred(Isomers, settings):

    jobdir = os.getcwd()

    if not os.path.exists('nmr'):
        os.mkdir('nmr')
    os.chdir('nmr')

    if os.path.exists('SGNN_inputs.csv'):
        os.remove('SGNN_inputs.csv')
    with open('SGNN_inputs.csv', 'w') as inputfile:
        inputfile.write('number,sample-id\n')
        for num, iso in enumerate(Isomers):
            if os.path.exists(iso.BaseName + '.sout'):
                iso.SGNNOutputFiles.append(iso.BaseName + '.sout')
            else:
                inputfile.write(str(num) + ',' + iso.BaseName + '\n')

    os.chdir(jobdir)

    return Isomers


def RunNMRPred(Isomers, settings):

    print('\nRunning SGNN NMR prediction locally...')

    from SGNN import prediction
    
    save_folder = os.getcwd() + '/nmr'
    path_csv = save_folder + '/SGNN_inputs.csv'
    prediction(save_folder, path_csv)

    return Isomers


def ReadPred(Isomers):

    jobdir = os.getcwd()
    os.chdir('nmr')

    for iso in Isomers:
        output_file = iso.BaseName + '.sout'
        with open(output_file, 'r') as sgnnfile:
            for shift in sgnnfile:
                iso.PredShifts.append(shift)

    os.chdir(jobdir)

    return Isomers



def GetAtomSymbol(AtomNum):
    Lookup = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
              'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
              'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
              'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
              'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
              'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
              'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

    if AtomNum > 0 and AtomNum < len(Lookup):
        return Lookup[AtomNum-1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0
