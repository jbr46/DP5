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

    for iso in Isomers:
        if iso.ExtCharge > -10:
            charge = iso.ExtCharge
        else:
            charge = iso.MMCharge

        # if iso.DFTConformers == []:
        #     conformers = iso.Conformers
        # else:
        #     conformers = iso.DFTConformers

        filename = iso.BaseName + 'sinp'

        if os.path.exists(filename + '.sout'):
            if IsSGNNCompleted(filename + '.sout'):
                iso.SGNNOutputFiles.append(filename + '.sout')
            else:
                os.remove(filename + '.sout')
            
            WriteSGNNFile(filename, iso.Atoms, charge, settings, 'nmr')
            iso.SGNNInputFiles.append(filename + '.scom')

        # for num in range(0, len(conformers)):
        #     filename = iso.BaseName + 'ginp' + str(num + 1).zfill(3)

        #     if os.path.exists(filename + '.out'):
        #         if IsGausCompleted(filename + '.out'):
        #             iso.NMROutputFiles.append(filename + '.out')
        #             continue
        #         else:
        #             os.remove(filename + '.out')

        #     WriteGausFile(filename, conformers[num], iso.Atoms, charge, settings, 'nmr')
        #     iso.NMRInputFiles.append(filename + '.com')

    os.chdir(jobdir)

    return Isomers


def RunNMRPred(Isomers, settings):

    print('\nRunning SGNN NMR prediction locally...')

    jobdir = os.getcwd()
    os.chdir('nmr')

    SGNNJobs = []

    for iso in Isomers:
        SGNNJobs.extend([x for x in iso.NMRInputFiles if (x[:-4] + '.sout') not in iso.NMROutputFiles])

    Completed = RunCalcs(GausJobs, settings)

    for iso in Isomers:
        iso.NMROutputFiles.extend([x[:-4] + '.sout' for x in iso.NMRInputFiles if (x[:-4] + '.sout') in Completed])

    os.chdir(jobdir)

    return Isomers


def GetPrerunNMRPred(Isomers):

    print('\nLooking for prerun SGNN NMR prediction files...')

    jobdir = os.getcwd()
    os.chdir('nmr')

    for iso in Isomers:
        iso.NMRInputFiles = glob.glob(iso.BaseName + 'sinp*scom')
        iso.NMROutputFiles.extend([x[:-4] + '.sout' for x in iso.NMRInputFiles if IsGausCompleted(x[:-4] + '.sout')])

    print('NMR prediction files:')
    print(', '.join([', '.join(x.NMROutputFiles) for x in Isomers]))

    os.chdir(jobdir)

    return Isomers


def RunCalcs(SGNNJobs, settings):

    NCompleted = 0
    Completed = []

    if len(SGNNJobs) == 0:
        print("There were no jobs to run.")
        return Completed

    if ('GAUS_EXEDIR' in os.environ):
        gausdir = os.environ['GAUSS_EXEDIR']
        if shutil.which(os.path.join(gausdir, 'g09')) is None:
            GausPrefix = os.path.join(gausdir, "g16")
        else:
            GausPrefix = os.path.join(gausdir, "g09")
    else:
        GausPrefix = settings.GausPath

    if shutil.which(GausPrefix) is None:
        print('Gaussian.py, RunCalcs:\n  Could not find Gaussian executable at ' + GausPrefix)
        quit()

    for f in GausJobs:
        time.sleep(3)
        print(GausPrefix + " < " + f + ' > ' + f[:-3] + 'out')
        outp = subprocess.check_output(GausPrefix + " < "  + f + ' > ' + f[:-3] + 'out', shell=True,timeout= 86400)

        NCompleted += 1
        if IsGausCompleted(f[:-4] + '.out'):
            Completed.append(f[:-4] + '.out')
            print("Gaussian job " + str(NCompleted) + " of " + str(len(GausJobs)) + \
                  " completed.")
        else:
            print("Gaussian job terminated with an error. Continuing.")

    if NCompleted > 0:
        print(str(NCompleted) + " Gaussian jobs completed successfully.")

    return Completed


def WriteSGNNFile(SGNNinp, atoms, charge, settings, type):

    f = open(SGNNinp + '.scom', 'w')
    if(settings.nProc > 1):
        f.write('%nprocshared=' + str(settings.nProc) + '\n')
    if settings.DFT == 'g':
        f.write('%mem=2000MB\n%chk='+Gausinp + '.chk\n')
    elif settings.DFT == 'z':
        f.write('%mem=1000MB\n%chk=' + Gausinp + '.chk\n')
    else:
        f.write('%mem=6000MB\n%chk='+Gausinp + '.chk\n')

    if type == 'nmr':
        f.write(NMRRoute(settings))
    elif type == 'e':
        f.write(ERoute(settings))
    elif type == 'opt':
        f.write(OptRoute(settings))

    f.write('\n'+Gausinp+'\n\n')
    f.write(str(charge) + ' 1\n')

    natom = 0

    for atom in conformer:
        f.write(atoms[natom] + '  ' + atom[0] + '  ' + atom[1] + '  ' +
                atom[2] + '\n')
        natom = natom + 1
    f.write('\n')

    f.close()


def IsGausCompleted(f):
    Gfile = open(f, 'r')
    outp = Gfile.readlines()
    Gfile.close()
    if len(outp) < 10:
        return False
    if ("Normal termination" in outp[-1]) or (('termination' in '\n'.join(outp[-3:])) and ('l9999.exe' in '\n'.join(outp[-3:]))):
        return True
    else:
        return False


def ReadPred(Isomers):

    jobdir = os.getcwd()
    os.chdir('nmr')

    for iso in Isomers:

        if len(iso.SGNNOutputFiles) < 1:
            print("SGNN.py, ReadPred: No NMR SGNN output" +
                  " files found, NMR data could not be read. Quitting.")
            quit()

        for OutpFile in iso.NMROutputFiles:

            with open(OutpFile, 'r') as sgnnfile:
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
