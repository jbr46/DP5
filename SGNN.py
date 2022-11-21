## RUN Calculations
# For some reason I get an error when running main_execute(data_df, save_folder)
## but here it works fine

import pandas as pd
from tqdm import tqdm
import glob
from rdkit.Chem.rdmolfiles import MolFromSmiles
from execution import *
# load_std_mean, save_as_npy_for_13C, save_as_npy_for_1H, \
#   save_results_sdf_file, load_model
from helpers import *

def prediction(save_folder, path_csv):
  data_df = pd.read_csv(path_csv)


  graph_representation = "sparsified"
  target = "13C" 
  train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
  target = "1H" 
  train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)

  SMILES_list = data_df["SMILES"]
  sample_id = data_df["sample-id"]

  flist = glob.glob(save_folder +"/*")
  done = False
  failed_molecules = []
  x_h_connectivity_dict = {}

  for smiles, ID in tqdm(zip(SMILES_list[:],sample_id[:])):
      done = False
      for i in flist:
        if ID in i:
          done = True
          break
      if done == False:
          # try:
            # print(smi, ID)
            mol = MolFromSmiles(smiles)
            ############## For 13C ##############
            target = "13C" 
            save_path_C = save_as_npy_for_13C(mol)
            net_C = load_model(target, save_path_C)
            test_y_pred_C = inference_C(target, net_C, save_path_C, train_y_mean_C, train_y_std_C)

            ############## For 1H ##############
            target = "1H" 
            save_path_H, x_h_connectivity_dict = save_as_npy_for_1H(mol)
            net_H = load_model(target, save_path_H)
            test_y_pred_H = inference_H(target, net_H, save_path_H, train_y_mean_H, train_y_std_H)

            ############## Reconstruction ##############

            mol = Chem.AddHs(mol)
            final_list = create_shift_list(mol, x_h_connectivity_dict, test_y_pred_C, test_y_pred_H)

            output_sdf = save_results(mol, save_folder, ID, final_list)
            # print(output_sdf)
            # break
          # except:
          #   failed_molecules.append(smi)
      # break
  # Delete the normal SDF files
  flist_filter = [f for f in flist if not "SGNN_*.sout" in f]
  for i in flist_filter:
    os.remove(i)