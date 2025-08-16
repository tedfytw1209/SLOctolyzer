import os
import sys
SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
if os.name == 'posix': 
    MODULE_PATH = "/".join(SCRIPT_PATH.split('/')[:-1])
    PACKAGE_PATH = "/".join(SCRIPT_PATH.split('/')[:-2])
elif os.name == 'nt':
    MODULE_PATH = "\\".join(SCRIPT_PATH.split('\\')[:-1])
    PACKAGE_PATH = "\\".join(SCRIPT_PATH.split('\\')[:-2])
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import shutil
import argparse
import time
import pprint
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path, PosixPath, WindowsPath
import analyse
from PIL import Image, ImageOps
import utils
from skimage import segmentation, morphology
import matplotlib.pyplot as plt
from segment import slo_inference, avo_inference, fov_inference
from measure import slo_measurement
import utils


# Outer function to run analyse script through directory of .vol files
def run(args):
    '''
    Outer function to analyse .vol files from analysis_directory.
    '''

    # analysis directory
    analysis_fname = args["analysis_csv"]
    analysis_df = pd.read_csv(analysis_fname)

    # Detect .vol files from analysis_directory
    oct_paths = analysis_df['Path'].tolist()
    enface_paths = analysis_df['Enface Path'].tolist()
    
    # output directory
    save_directory = args["output_directory"]
    if not os.path.exists(save_directory):
        print(f"""Cannot find directory
{save_directory} to store results. Creating folder.""")
        os.mkdir(save_directory)

    # This is a particularly helpful parameter when running large batches, and ignored
    # any unexpected errors from a particular file. Setting it as 0 will throw up errors
    # for debugging
    robust_run = args["robust_run"]
    save_ind_results = True
    save_ind_images = args["save_individual_segmentations"]
    collate_segs = True
    
    # create segmentations directory if specified
    segmentation_directory = os.path.join(save_directory, "segmentations")
    if collate_segs and not os.path.exists(segmentation_directory):
        os.mkdir(segmentation_directory)

    # Instantiate SLO binary/AVOD/Fovea segmentation model
    print(f"\nRunning En-face vessel SLO analysis.\n")
    slosegmenter = slo_inference.SLOSegmenter()
    avosegmenter = avo_inference.AVOSegmenter()
    fovsegmenter = fov_inference.FOVSegmenter()

    # Detect and load resolution file
    print("\n\nWARNING: Cannot find resolution file (in .csv or .xlsx format).")
    print("Falling back to default state of measuring in pixels instead of microns for any detected image files.")
    print("Place a file 'fname_resolution_location_eye.csv' with filenames, scaling and location to prevent mis-measurement.\n\n")
    res_df = pd.DataFrame(columns=["Filename", "Scale", 'Location', 'Eye'])


    # Loop through .img files, segment, measure and save out in analyse()
    st = time.time()
    result_dict = {}
    for (oct_path, enface_path) in tqdm(zip(oct_paths, enface_paths), total=len(oct_paths)):

        if isinstance(oct_path, PosixPath):
            fname_type = str(oct_path).split('/')[-1]
        elif isinstance(oct_path, WindowsPath):
            fname_type = str(oct_path).split("\\")[-1]
        else:
            fname_type = str(oct_path).split("/")[-1]
        ftype = fname_type.rsplit('.',1)[1]
        fname = fname_type.rsplit(".",1)[0]
        fname_path = os.path.join(save_directory, fname)
        output_fname = os.path.join(fname_path, f"{fname}_output.xlsx")
        
        if os.path.exists(output_fname):
            msg = f"\n\nPreviously analysed {fname}. Attempting to load measurements and segmentations."
            print(msg)
            logging_list = [msg]
            # Load in previously analysed results (if saved out)
            try:
                ind_df = pd.read_excel(output_fname, sheet_name="metadata")
                slo_dfs = []
                if ind_df.location.iloc[0] == "Optic disc":
                    for r in ["B", "C", "whole"]:
                        df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_{r}")
                        slo_dfs.append(df)
                else:
                    df = pd.read_excel(output_fname, sheet_name=f"slo_measurements_whole")
                    slo_dfs.append(df)
                msg = "Successfully loaded in measurement file!"
                print(msg)
                logging_list.append(msg)
            except:
                msg = "Cannot find measurement file to append to final batch measurements."
                print(msg)
                logging_list.append(msg)
                ind_df = None
                slo_dfs = None

            # Check if manual annotation has occurred
            binary_nii_path = os.path.join(fname_path, f"{fname}_slo_binary_map.nii.gz")
            avod_nii_path = os.path.join(fname_path, f"{fname}_slo_avod_map.nii.gz")
            new_avodmap = False
            new_binary = False
            if os.path.exists(avod_nii_path):
                new_avodmap = True
            if os.path.exists(binary_nii_path):
                new_binary = True
                
            # Compute new metrics if new manual annotations have been created
            if new_avodmap or new_binary:    
                msg = f"\n\nAnalysing {fname}."
                logging_list.append(msg)
                print(msg)
                msg = "Detected manual annotation of"
            
                # Load in annotations if exists. If not, load in already saved segmentation
                segmentation_dict = {}
                if new_avodmap:
                    msg += " artery-vein-optic disc mask"
                    if new_binary:
                        msg += " and"
                    new_avodmap = utils.load_annotation(avod_nii_path)
                else:
                    avodmap = np.array(Image.open(os.path.join(fname_path, f"{fname}_slo_avod_map.png")))
                    new_avodmap = np.concatenate([avodmap[...,np.newaxis]==191,
                                                 avodmap[...,np.newaxis]==255,
                                                 avodmap[...,np.newaxis]==127,
                                                 avodmap[...,np.newaxis]>0], axis=-1)
                if new_binary:
                    msg += " binary vessel mask"
                    new_binary = utils.load_annotation(binary_nii_path, binary=True)
                else:
                    new_binary = np.array(ImageOps.grayscale(Image.open(os.path.join(fname_path, f"{fname}_slo_binary_map.png"))))/255
                msg += ". Recomputing metrics..."
                print(msg)
                logging_list.append(msg)
                    
                # Collect new segmentations and metadata (eye, scale and location). If unspecified, set to None
                segmentation_dict['avod_map'] = new_avodmap.astype(int)
                segmentation_dict['binary_map'] = new_binary.astype(int)
                meta_dict = dict(ind_df.iloc[0])
                segmentation_dict['metadata'] = meta_dict

                # Extract scale, location and eye from metadata
                k_params = []
                for k in ['scale', 'location', 'eye']:
                    if k in meta_dict:
                        kval = meta_dict[k]
                        if pd.isna(kval):
                            kval = None
                    else:
                        kval = None
                    k_params.append(kval)
                scale, location, eye = k_params
                
                # Run analysis again, recomputing metrics for vessels which have been manually edited
                # inside try-exception block
                try:
                    output = analyse.analyse(oct_path, 
                                            save_directory, 
                                            scale,
                                            location,
                                            eye,
                                            save_results=True,
                                            save_images=True,
                                            collate_segmentations=True,
                                            compute_metrics=True,
                                            verbose=True,
                                            segmentation_dict=segmentation_dict,
                                            enface_path=enface_path)
                    _, slo_dfs, _, _, _ = output
                    msg = "Done!"
                    print(msg)
                    logging_list.append(msg)
                except Exception as e:
                    
                    # print and log error
                    logging_list = utils.print_error(fname, e)
                    slo_dfs = None
                    ind_df = e.args[0]

            output = (ind_df, slo_dfs, logging_list)
            result_dict[fname_type] = output
            
        else:
            scale = None
            location = None
            eye = None
            if ftype not in ['vol', 'dcm']:
                fname_res_loc = res_df[res_df.Filename == fname_type]
                if fname_res_loc.shape[0] == 0:
                    print(f"\n\nCan't find information on {fname}. If fname_resolution_location.xlsx specified, check filename spelling?")
                    print(f"Falling back to default state of not using a scale and automatically detecting centering.")
                else:
                    # extract scale and location. If unspecified, set to None
                    scale = fname_res_loc.iloc[0].Scale
                    if pd.isna(scale):
                        scale = None
                    location = fname_res_loc.iloc[0].Location
                    if pd.isna(location):
                        location = None
                    eye = fname_res_loc.iloc[0].Eye
                    if pd.isna(eye):
                        eye = None

            # For robust run, i.e. unexpected errors do not hault run and instead moved onto next image.
            if robust_run:
                try:
                    output = analyse.analyse(oct_path, 
                                    save_directory, 
                                    scale,
                                    location,
                                    eye,
                                    slosegmenter, 
                                    avosegmenter,
                                    fovsegmenter,
                                    save_ind_results,
                                    save_ind_images,
                                    collate_segs,
                                    enface_path=enface_path,
                                    compute_metrics=True)
                    ind_df, slo_dfs, _, _, logging_list = output
                    result_dict[fname_type] = ind_df, slo_dfs, logging_list
                    
                except Exception as e:
                    logging_list = utils.print_error(fname, e)
                    result_dict[fname_type] = e.args[0], None, logging_list
 
            else:
                output = analyse.analyse(oct_path, 
                                save_directory, 
                                scale,
                                location,
                                eye,
                                slosegmenter, 
                                avosegmenter,
                                fovsegmenter,
                                save_ind_results,
                                save_ind_images,
                                collate_segs,
                                compute_metrics=True,
                                enface_path=enface_path)
                ind_df, slo_dfs, _, _, logging_list = output
                result_dict[fname_type] = ind_df, slo_dfs, logging_list

    # collate all results into a single dataframe
    print(f"\n\nCollecting all results together into one output file.")
    all_logging_list = []
    all_df = pd.DataFrame()
    for fname_type, output in result_dict.items():

        fname = fname_type.rsplit(".", 1)[0]
        fname_path = os.path.join(save_directory, fname)
        os.makedirs(fname_path, exist_ok=True)
        all_logging_list.append(f"\n\nANALYSIS LOG OF {fname_type}")

        # Create default dataframe for failed individuals
        if isinstance(output[0], str):
            ind_df = pd.DataFrame({"Filename":fname_type}, index=[0])
            log = output[-1]
            all_logging_list.extend(log)    
            with open(os.path.join(fname_path, f"{fname}_log.txt"), "w") as f:
                for line in log:
                    f.write(line+"\n")
        
        # Otherwise, collate measurements and save out segmentations if specified
        else:    
            ind_df, slo_dfs, logging_list = output

            # If cannot find measurements/metadata, create default dataframe and bypass
            # the segmentation visualisation
            if ind_df is None or slo_dfs is None:
                ind_df = pd.DataFrame({"Filename":fname_type}, index=[0])
                continue
            all_logging_list.extend(logging_list)

            # process measurement DFs and save out global dataframe
            flat_df = pd.DataFrame()
            rtypes = []
            for df in slo_dfs:
                # flatten
                dfarr = df.values.flatten()

                # Collect all columns in flattened DF
                dfcols = list(df.columns)
                rtype = df.zone.iloc[0]
                rtypes.append(rtype)
                vtypes = df.vessel_map.values
                dfcols = [col+f"_{vtype}_{rtype}" for vtype in vtypes for col in dfcols]
                df_flatten = pd.DataFrame(dfarr.reshape(1,-1), columns = dfcols, index=[0])

                # Remove indexing columns and concatenate with different zones
                cols_to_drop = df_flatten.columns[df_flatten.columns.str.contains("vessel_map|zone")]
                df_flatten = df_flatten.drop(cols_to_drop, axis=1, inplace=False)
                flat_df = pd.concat([flat_df, df_flatten], axis=1)    

            # Order feature columns by importance in literature
            order_str = ["fractal_dimension", "vessel_density", 'average_global_calibre', 'average_local_calibre', "tortuosity_density",
                         'CRAE_Knudtson', 'CRVE_Knudtson', 'AVR']
            order_str_rv = [col+f"_{vtype}_{rtype}" for rtype in rtypes[::-1] for vtype in vtypes for col in order_str]
            flat_df = flat_df[order_str_rv]
            flat_df = flat_df.loc[:, ~(flat_df == -1).any()]
            flat_df = flat_df.rename({f"AVR_artery-vein_{rtype}":f"AVR_{rtype}" for rtype in rtypes}, inplace=False, axis=1)

            # Concatenate measurements with metadata
            ind_df = pd.concat([ind_df, flat_df], axis=1)

        # Concatenate to create global dataframe of results
        if len(all_df) > 0:
            all_df = pd.concat([all_df, ind_df], axis=0)
        else:
            all_df = ind_df.copy()
        

    # save out measurement
    all_df = all_df.reset_index(drop=True)
    all_df.to_csv(os.path.join(save_directory, "analysis_output.csv"), index=False)

    # save out log
    with open(os.path.join(save_directory, f"analysis_log.txt"), "w") as f:
        for line in all_logging_list:
            f.write(line+"\n")
    print("Done!")
    
    elapsed = time.time() - st
    print(f"\n\nCompleted analysis in {elapsed:.03f} seconds.\n\n")




# Once called from terminal
if __name__ == "__main__":
    
    print("Checking configuration file for valid inputs...")

    # Load in configuration from file
    config_path = os.path.join(MODULE_PATH, 'config.txt')
    with Path(config_path).open('r') as f:
        lines = f.readlines()
    inputs = [l.strip() for l in lines if (":" in str(l)) and ("#" not in str(l))]
    params = {p.split(": ")[0]:p.split(": ")[1] for p in inputs}
    
    # Make sure inputs are correct format before constructing args dict
    for key, param in params.items():
        
        # Checks for directory
        if "directory" in key:
            check_param = param.replace(" ", "")
            if "image" in key:
                try:
                    assert os.path.exists(param), f"The specified path:\n{param}\ndoes not exist. Check spelling or location. Exiting analysis."
                except AssertionError as msg:
                    sys.exit(msg)
            continue

        # Check numerical value inputs
        param = param.replace(" ", "")
        if param == "":
            msg = f"No value entered for {key}. Please check config.txt. Exiting analysis"
            sys.exit(msg)

        # All remaining inputs should be either 0 or 1 
        try:
            assert param in ["0", "1"], f"{key} flag must be either 0 or 1, not {param}. Exiting analysis."
        except AssertionError as msg:
            sys.exit(msg)

    # Construct args dict and run
    args = {key:val if ("directory" in key) else int(val) for (key,val) in params.items()}

    # run analysis
    run(args)