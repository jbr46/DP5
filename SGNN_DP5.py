#!/usr/bin/env python

# -*- coding: utf-8 -*-
 
import os
import sys

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
                os.remove(iso.BaseName + '.sout')
            inputfile.write(str(num) + ',' + iso.BaseName + '\n')

    os.chdir(jobdir)

    return Isomers


def RunNMRPred(Isomers, settings):

    print('\nRunning SGNN NMR prediction locally...')

    sys.path.insert(0, settings.SGNN_path)

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
                iso.PredShifts.append(float(shift.rstrip('\n')))

    os.chdir(jobdir)

    return Isomers