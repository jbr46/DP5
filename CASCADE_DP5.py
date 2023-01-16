#!/usr/bin/env python

# -*- coding: utf-8 -*-
 
import subprocess
import os
import time
import glob
import shutil

def SetupCNMRPred(Isomers, settings):

    jobdir = os.getcwd()

    if not os.path.exists('nmr'):
        os.mkdir('nmr')
    os.chdir('nmr')

    if os.path.exists('CASCADE_inputs.csv'):
        os.remove('CASCADE_inputs.csv')
    with open('CASCADE_inputs.csv', 'w') as inputfile:
        inputfile.write('number,id\n')
        for num, iso in enumerate(Isomers):
            if os.path.exists(iso.BaseName + '.cout'):
                os.remove(iso.BaseName + '.cout')
            inputfile.write(str(num) + ',' + iso.BaseName + '\n')

    os.chdir(jobdir)

    return Isomers


def RunCNMRPred(Isomers, settings):

    print('\nRunning CASCADE NMR prediction locally...')

    from CASCADE import prediction
    
    save_folder = os.getcwd() + '/nmr'
    path_csv = save_folder + '/CASCADE_inputs.csv'
    prediction(save_folder, path_csv)

    return Isomers


def ReadCPred(Isomers):

    jobdir = os.getcwd()
    os.chdir('nmr')

    for iso in Isomers:
        output_file = iso.BaseName + '.cout'
        with open(output_file, 'r') as cascadefile:
            for shift in cascadefile:
                iso.PredShifts.append(float(shift.rstrip('\n')))

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
