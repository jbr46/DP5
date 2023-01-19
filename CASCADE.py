import pandas as pd

from keras.models import load_model

from nfp.layers.layers import (Squeeze, ReduceBondToAtom, GatherAtomToBond, ReduceAtomToPro)
from nfp.models.models import GraphModel
from cascade.apply import predict_NMR_C,predict_NMR_H

import os

def prediction(save_folder, path_csv):

    # modelpath_C = os.path.join('cascade', 'trained_model', 'best_model.hdf5')
    # modelpath_H = os.path.join('cascade', 'trained_model', 'best_model_H_DFTNN.hdf5')
    modelpath_C = '/users/benji/dp5/cascade/trained_model/best_model.hdf5'
    modelpath_H = '/users/benji/dp5/cascade/trained_model/best_model_H_DFTNN.hdf5'

    # Load the C and H NMR prediction models
    NMR_model_C = load_model(modelpath_C, custom_objects={'GraphModel': GraphModel,
                                                'ReduceAtomToPro': ReduceAtomToPro,
                                                'Squeeze': Squeeze,
                                                'GatherAtomToBond': GatherAtomToBond,
                                                'ReduceBondToAtom': ReduceBondToAtom})
    NMR_model_H = load_model(modelpath_H, custom_objects={'GraphModel': GraphModel,
                                                'ReduceAtomToPro': ReduceAtomToPro,
                                                'Squeeze': Squeeze,
                                                'GatherAtomToBond': GatherAtomToBond,
                                                'ReduceBondToAtom': ReduceBondToAtom})

    # Load all the candidate structures
    data = pd.read_csv(path_csv)
    data.columns

    # Loop over all the candidate structures
    for ID in data.id:
        # C predictions
        pred_data_C = pd.DataFrame()
        mols, weightedPrediction, spreadShift = predict_NMR_C(ID, NMR_model_C)
        weightedPrediction['ID'] = ID
        pred_data_C = pd.concat([pred_data_C,weightedPrediction])

        # H predictions
        pred_data_H = pd.DataFrame()
        mols, weightedPrediction, spreadShift = predict_NMR_H(ID, NMR_model_H)
        weightedPrediction['ID'] = ID
        pred_data_H = pd.concat([pred_data_H, weightedPrediction])

        predictions = pd.concat([pred_data_C, pred_data_H])

        # Save the results of the NMR prediction into output files, ordered by the atom index. 
        # Fill in zeros for atoms other than C and H.
        name = str(ID) + ".cout"
        output_file = os.path.join(save_folder, name)
        with open(output_file, "w",  encoding='utf-8', errors='ignore') as output:
            idx = 1
            for label, shift in zip(predictions.atom_index, predictions.Shift):
                while idx < label:
                    output.write(str(0))
                    output.write('\n')
                    idx += 1
                output.write(str(shift))
                output.write('\n')
                idx += 1