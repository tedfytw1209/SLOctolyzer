import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import cv2
import scipy
import copy
from PIL import Image, ImageOps
from eyepy.core import utils as eyepy_utils
from eyepy.io.he import vol_reader
from skimage import segmentation, morphology
from pathlib import Path, WindowsPath, PosixPath
from sloctolyzer import utils
from sloctolyzer.measure import slo_measurement
from sloctolyzer.segment import slo_inference, avo_inference, fov_inference

def analyse(path, 
            save_path=None, 
            scale=None,
            location=None,
            eye=None,
            slo_model=None, 
            avo_model=None, 
            fov_model=None,
            save_results=True,
            save_images=False,
            collate_segmentations=True,
            compute_metrics=True,
            verbose=True,
            segmentation_dict={},
            demo_return=False,
            enface_path=None,
            ):
    """
    Inner function to analyse an individual IR-SLO img, given options for scaling/location/eye.

    Inputs:
    -------------------
    path (str) : Path to image/vol file.

    save_path (str) : Path to save results to. Will be created if doesn't exist

    scale (str) : Microns-per-pixel conversion factor from pixel space to physical space.
    
    location (str) : Either 'Macular' or 'Optic disc' centred. 
    
    Eye (str) : Either 'Right' or 'Left'.

    slo_model, avo_model, fov_model : binary, artery-vein-optic disc, and fovea detection models.

    save_results (bool) : Flag to save log and xlsx output file with feature measurements and metadata.

    save_images (bool) : Flag to save segmentation masks and segmentations superimposed onto SLO.

    collate_segmentations (bool) : Flag to save the superimposed segmentation image to a global directory, 
                                   assuming batch analysis.

    compute_metrics (bool) : Flag to compute feature measurements of the retinal vessels.

    segmentation_dict (dict) : Dictionaries with new segmentation to recompute measurements. By default left empty UNLESS
                               correcting manual segmentations.
    """
    # Initialise list of messages to save
    logging_list = []
    metadata = {}

    # load image, check to make sure is path
    if isinstance(path, (str, WindowsPath, PosixPath)):

        # check if vol file, otherwise is regular image file
        ftype = str(path).rsplit('.', 1)[-1]
        if ftype.lower() == 'vol':
            slo, meta, log = utils.load_volfile(path, verbose=verbose, logging=[])
            eye = meta['eye']
            scale = meta['scale']
            location = meta['location']
            metadata = copy.deepcopy(meta)
            logging_list.extend(log)
        elif ftype.lower() == 'dcm':
            slo, meta, log = utils.load_dcmfile(path, enface_path=enface_path, verbose=verbose, logging=[])
            eye = meta['eye']
            scale = meta['scale']
            location = meta['location']
            metadata = copy.deepcopy(meta)
            logging_list.extend(log)
        else:
            slo = np.array(ImageOps.grayscale(Image.open(path)))

    elif isinstance(path, np.ndarray):
        slo = path.copy()
        path = save_path   
    else:
        msg = "Unknown filetype, must be either string/filepath/numpy array."
        logging_list.append(msg)
        if verbose:
            print(msg)
        return
    img_shape = slo.shape

    # accuounting for image files saved from HEYEX with info bar at bottom of image which is 100 rows.
    if slo.shape[0]==1636 or slo.shape[0]==868:
        slo = slo[:img_shape[0]-100]
        img_shape = slo.shape
    _, N = img_shape

    # Collect metadata for recomputing measurements when a manual annotation is provided
    segmented_already = False
    if 'metadata' in segmentation_dict:
        segmented_already = True
        metadata = copy.deepcopy(segmentation_dict['metadata'])
        fovea = np.array([metadata['fovea_x'], metadata['fovea_y']]).astype(int)
        location = metadata['location']
        eye = metadata['eye']
        
        # Extract segmentation masks, recompute OD centre and OD radius
        slo_avimout = segmentation_dict['avod_map']
        slo_vbinmap = segmentation_dict['binary_map']
        od_mask = slo_avimout[...,1]
        od_centre = avo_inference._get_od_centre(od_mask)
        if location == 'Optic disc':
            od_radius, od_boundary = utils._process_opticdisc(od_mask)
            metadata['optic_disc_x'] = od_centre[0]
            metadata['optic_disc_y'] = od_centre[1]
            metadata['optic_disc_radius_px'] = od_radius
        else:
            od_centre = None

    # compatability with mac, linux and windows
    dirpath = save_path
    if isinstance(Path(path), PosixPath):
        split_path = str(path).split('/')
    elif isinstance(Path(path), WindowsPath):
        split_path = str(path).split('\\')
    fname_type = split_path[-1]
        
    # Get filename and log to user SLO is being analysed
    fname = fname_type.split(".")[0]
    metadata['Filename'] = fname_type
    msg = f"\nAnalysing {fname}."
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Error handle scale
    if scale is not None:
        if not isinstance(scale, (float, int)):
            msg = f"Pixel lengthscale {scale} should be a float or integer. Ignoring scale and measuring in pixels."
            logging_list.append(msg)
            if verbose:
                print(msg)
            scale = None
        if (scale > 20) or (scale < 3):
            msg = f"Pixel lengthscale {scale} should be in [3,20] microns-per-pixel. Is your scale in mm-per-pixel?. Ignoring scale and measuring in pixels."
            logging_list.append(msg)
            if verbose:
                print(msg)
            scale = None

    # Error handle save_path
    if (save_results+save_images)>0:
        if save_path is None:
            msg = f"Path {save_path} is not specified, but option to save is flagged. Creating directory 'output' in current working directory."
            logging_list.append(msg)
            if verbose:
                print(msg)
            save_path = "output"
            os.mkdir(save_path)
              
        else: 
            if not os.path.exists(save_path):
                msg = f"Path {save_path} does not exist. Creating directory."
                logging_list.append(msg)
                if verbose:
                    print(msg)
                os.mkdir(save_path)

        # Create fname directory
        save_path = os.path.join(save_path, fname)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
                
    # Save out SLO image
    if slo.max() == 1:
        slo_save = (255*slo).astype(np.uint8)
    else:
        slo_save = slo.copy()
    if save_images:
        cv2.imwrite(os.path.join(save_path,f"{fname}_slo.png"), slo_save)

    # SEGMENTING
    if not segmented_already:
        msg = "\nSEGMENTING..."
        
        logging_list.append(msg)
        if verbose:
            print(msg)
    
        # Forcing model instantiation if unspecified
        # SLO segmentation models
        if slo_model is None or type(slo_model) != slo_inference.SLOSegmenter:
            msg = "Loading models..."
            logging_list.append(msg)
            if verbose:
                print(msg)
            slo_model = slo_inference.SLOSegmenter()
         # SLO segmentation models
        if fov_model is None or type(fov_model) != fov_inference.FOVSegmenter:
            fov_model = fov_inference.FOVSegmenter()
        # AVO segmentation models
        if avo_model is None or type(avo_model) != avo_inference.AVOSegmenter:
            avo_model = avo_inference.AVOSegmenter()

        # binary vessel detection
        msg = "    Segmenting binary vessels from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        slo_vbinmap = slo_model.predict_img(slo)

        # fovea detection
        msg = "    Segmenting fovea from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        fmask, fovea = fov_model.predict_img(slo)
        if save_images:
            cv2.imwrite(os.path.join(save_path,f"{fname}_slo_fovea_map.png"), 
                        (255*fmask).astype(np.uint8))

        # artery-vein-optic disc detection, using binary vessel detector as original reference
        # We also reassigh the binary vessel map as artery+vein maps
        msg = "    Segmenting artery-vein vessels and optic disc from SLO image."
        logging_list.append(msg)
        if verbose:
            print(msg)
        slo_avimout, od_centre = avo_model.predict_img(slo, location=location)#, slo_vbinmap)
        if od_centre is None:
            msg = 'WARNING: Optic disc not detected. Please check image.'
            logging_list.append(msg)
            if verbose:
                print(msg)
        od_mask = slo_avimout[...,1]

    # Attempt to resolve location if not inputted
    msg = "\n\nInferring image metadata..."
    logging_list.append(msg)
    if verbose:
        print(msg)
    metadata["location"] = location
    if location is None:
        
        # We check whether x-location of optic disc centre is in the main
        # area of the image, i.e. in [0.1N, 0.9N] where N is image width
        location = "Macula"
        if od_centre is not None:
            if 0.1*img_shape[1] < od_centre[0] < 0.9*img_shape[1]:
                location = "Optic disc"
        msg = f"    No location specified. Detected SLO image to be {location.lower()}-centred."
        logging_list.append(msg)
        if verbose:
            print(msg)
        metadata["location"] = location
    else:
        msg = f"    Location is specified as {location.lower()}-centred."
        logging_list.append(msg)
        if verbose:
            print(msg)
            
    # If eye unspecified, try to infer
    if eye is None:
        # If fovea known, detect which eye based on position of OD and fovea as first port of call
        if fovea.sum() != 0 and od_centre is not None:
            if fovea[0] < od_centre[0]:
                eye = "Right"
            else:
                eye = "Left"
            msg = f"    Using the position of the fovea and optic disc, we infer it is the {eye} eye."
              
        # if the fovea is not detected (predictions at origin), we must only use optic disc centre
        # positioning wrt image We check whether the optic disc x-location is less than half the image width
        elif od_centre is not None and fovea.sum() == 0:
            ratio = 0.05
            if od_centre[0] < (0.5-ratio)*img_shape[1]:
                eye = "Left"
                msg = f"    The optic disc is nearer the left of the image, and so the SLO image is assumed to be the {eye} eye. Please check."
            elif od_centre[0] > (0.5+ratio)*img_shape[1]:
                eye = "Right"
                msg = f"    The optic disc is nearer the right of the image, and so the SLO image is assumed to be the {eye} eye. Please check."
            elif (0.5-ratio)*img_shape[1] < od_centre[0] < (0.5+ratio)*img_shape[1]:
                msg = "    The optic disc is near the centre of the image, "
                if fovea.sum() == 0 and od_centre[0] < 0.5*img_shape[1]:
                    eye = "Left"
                    msg += f" and no fovea is detected. The SLO image is assumed as the {eye} eye. Please check."
                elif fovea.sum() == 0 and od_centre[0] > 0.5*img_shape[1]:
                    eye = "Right"
                    msg += f" and no fovea is detected. The SLO image is assumed as the {eye} eye. Please check."   
        
        # If all else fails, we check where the largest amount of vasculature is - i.e. should be around the disc which is
        # typically off-centre which tells us which eye it might be. Last chance saloon.
        elif od_centre is None:
            msg = '    Detecting eye based on which half of image has highest proportion of vessel pixels.'
            if slo_vbinmap[:, :N//2].sum() >= slo_vbinmap[:, N//2:].sum():
                eye = 'Left'
            else:
                eye = 'Right'
            msg += f" Thus, the SLO image is assumed as the {eye} eye. Please check."
        logging_list.append(msg)
        if verbose:
            print(msg)   

    # Eye provided in the metadata
    else:
        msg = f"    Eye type is specified as the {eye} eye."
        logging_list.append(msg)
        if verbose:
            print(msg)
    metadata["eye"] = eye
    metadata['manual_annotation'] = segmented_already

    # catch any potential fovea detections at origin, i.e. model did not predict fovea anywhere
    fovea_missing = False
    if fovea.sum() == 0:
        # alert user that fovea is missing
        fovea_missing = True
        msg = "Fovea was not detected. Please double-check image."
        logging_list.append(msg)
        if verbose:
            print(msg)  
    metadata["fovea_x"] = fovea[0]
    metadata["fovea_y"] = fovea[1]
    metadata["missing_fovea"] = fovea_missing

    # store optic disc centre,
    # The latter is only stored for for an optic disc-centred scan
    # macular-centred SLO do not show the optic disc entirel
    od_radius, od_boundary = utils._process_opticdisc(od_mask)
    if location == "Optic disc":
        metadata["optic_disc_x"] = od_centre[0]
        metadata["optic_disc_y"] = od_centre[1]
        metadata["optic_disc_radius_px"] = od_radius  
    else:
        od_radius = None
    if scale is None:
        metadata["measurement_units"] = "px"
    else:
        metadata["scale"] = scale
        metadata["measurement_units"] = "microns"
        metadata["scale_units"] = "microns-per-pixel"
    msg = f"Measurements which have units are in {metadata['measurement_units']} units. Otherwise they are non-dimensional."
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Binary vessels are:
    #        pixels detected by the binary vessel detector
    #        - any detected in the optic disc by the AVOD-model
    #        + any missed pixels identified by the AVOD-map.
    # We purposely choose not to fill in any missing AV-pixels using the binary vessel detector as this
    # can lead to many missclassified pixels due to the AV-model's uncertainty.
    # Therefore, the binary vessel map will ALWAYS contain more pixels detected.
    slo_vbinmap = (((slo_vbinmap + (slo_avimout[...,[0,2]].sum(axis=-1) > 0)).astype(bool)) * (1-od_mask)).astype(int)

    # option to save out segmentation masks
    if save_images:
        avoimout_save = 191*slo_avimout[...,0] + 127*slo_avimout[...,2] + 255*slo_avimout[...,1]
        Image.fromarray((255*slo_vbinmap).astype(np.uint8)).save(os.path.join(save_path,f"{fname}_slo_binary_map.png"))
        Image.fromarray((avoimout_save).astype(np.uint8)).save(os.path.join(save_path,f"{fname}_slo_avod_map.png"))

    # FEATURE MEASUREMENTS
    # - macula-centred SLO: 8mm square ROI if scale specified, otherwise whole image
    # - optic disc-centred SLO: Zone B and C (0.5-1, 0.5-2 annulus) around optic disc, and whole image
    msg = "\n\nFEATURE MEASUREMENT..."
    logging_list.append(msg)
    if verbose:
        print(msg)
    if compute_metrics:
        slo_dict = {}
        slo_keys = ["binary", "artery", "vein"]

        # specifying params for measurements
        if location == 'Macula':
            rois = ["whole"]
            macula_r = [0]
        elif location == 'Optic disc':
            rois = ["B", "C"]
            macula_r = [-1, -1]
            rois.extend(["whole"])
            macula_r.extend([0])

        # Logging
        msg = f"\nMeasuring en-face vessel metrics on {location.lower()}-centred SLO image"
        
        if location == "Macula":
            postfix_msg = " using a fovea-centred region of interest"
            slo_roi_center = fovea.copy()
        elif location == "Optic disc":
            postfix_msg = " using an optic disc-centred region of interest"
            slo_roi_center =  od_centre.copy()
            
        # if scale is None:
        msg += " using the whole image. This may lead to non-standardised measurements across a population."
        # else:
           # msg += postfix_msg + f" using a {macula_r[-1]}mm, {rois[-1]}-shaped ROI."
        logging_list.append(msg)
        if verbose:
            print(msg)

        if location == 'Optic disc':
            msg = "We will also measure Zones B (0.5-1 OD diameter) and C (2 OD diameter) from optic disc margin."
            logging_list.append(msg)
            if verbose:
                print(msg)

        # Loop over vessel maps to measure
        artery_vbinmap, vein_vbinmap = slo_avimout[...,0], slo_avimout[...,2]
        for v_map, v_type in zip([slo_vbinmap, artery_vbinmap, vein_vbinmap], slo_keys):
                
            # log to user 
            slo_dict[v_type] = {}
            msg = f"    Measuring {v_type} SLO vessel map"
            logging_list.append(msg)
            if verbose:
                print(msg)
            masks = []
            #mask_rois = []

            # Loop over zones and measure vessels
            for (grid, r) in zip(rois, macula_r):

                if location == 'Optic disc':
                    msg = f"        Using zone {grid} ROI" if grid in ["B", "C"] else f"        Using {grid} ROI"
                    logging_list.append(msg)
                    if verbose:
                        print(msg)

                # For debugging
                if demo_return and grid == 'C':
                    return slo_vbinmap, fovea, od_centre, od_radius, scale, r, grid

                # Compute features 
                output = slo_measurement.measure_sloroi(v_map, 
                                                       fovea,
                                                       od_centre,
                                                       od_radius,
                                                       scale, 
                                                       img_shape,
                                                       v_type, 
                                                       distance=r, 
                                                       roi_type=grid,
                                                       method='fast',
                                                       verbose=verbose)
                slo_dict[v_type][grid], logging, mask, mask_roi = output
                masks.append(mask)
                #mask_rois.append(mask_roi)
                logging_list.extend(logging)

        # Plot the segmentations superimposed onto the SLO
        if save_images or collate_segmentations:
            
            # binary vessel mask - purple
            slo_vcmap = utils.generate_imgmask(slo_vbinmap, None, 1)
            stacked_img = np.hstack(3*[slo/255])

            # Stacks the colour maps together, binary, then artery-vein-optic disc
            slo_av_cmap = slo_avimout.copy()
            slo_av_cmap[slo_av_cmap[...,1]>0,-1] = 0
            slo_av_cmap[...,1] = 0
            stacked_cmap = np.hstack([np.zeros_like(slo_vcmap), slo_vcmap, slo_av_cmap])
            if od_mask.sum() != 0:
                od_coords = avo_inference._fit_ellipse((255*od_mask).astype(np.uint8), get_contours=True)[:,0]
                od_coords = od_coords[(od_coords[:,0] > 0) & (od_coords[:,0] < N-1)]
                od_coords = od_coords[(od_coords[:,1] > 0) & (od_coords[:,1] < N-1)]
            # od_cmap = utils.generate_imgmask(np.hstack(2*[np.zeros_like(od_mask)]+[od_boundary]), None, 1)
            # od_cmap = utils.generate_imgmask(np.hstack(2*[np.zeros_like(od_mask)]+[od_mask]), None, 1)
        
            fig, ax = plt.subplots(1,1,figsize=(18,6))
            ax.imshow(stacked_img, cmap="gray")
            ax.imshow(stacked_cmap, alpha=0.5)
            for i in [N, 2*N]:
                ax.scatter(fovea[0]+i, fovea[1], marker="X", s=100, edgecolors=(0,0,0), c="r")
                if i == 2*N:  
                    if od_mask.sum() != 0:
                        ax.plot(od_coords[:,0]+i, od_coords[:,1], color='lime', linestyle='--', linewidth=3, zorder=4)
                        if location == "Optic disc":
                            ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="lime", zorder=4)
                else:
                    if od_mask.sum() != 0:
                        ax.plot(od_coords[:,0]+i, od_coords[:,1], color='blue', linestyle='--', linewidth=3, zorder=4)
                        if location == "Optic disc":
                            ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="blue", zorder=4)
                
                    if location == "Optic disc":
                        # Work out line between OD and fovea
                        # od_intersection, fov_od_line, intersection_idx = plotinfo
                        # x_grid, y_grid = fov_od_line
                        # Plot regions of interest
                        # int_idxs = []
                        for mask, colour, z in zip(masks[:2], [0,2], [3,2]):
                            mask_bnds = segmentation.find_boundaries(mask)
                            # mask_int_idx = np.argwhere(mask_bnds[(y_grid, x_grid)] == 1)[0][0]
                            # int_idxs.append(mask_int_idx)
                            mask = np.hstack(2*[np.zeros_like(mask)]+[mask])
                            cmap = utils.generate_imgmask(mask, None, colour)
                            mask_bnds = np.hstack(2*[np.zeros_like(mask_bnds)]+[mask_bnds])
                            mask_bnds = morphology.dilation(mask_bnds, footprint=morphology.disk(radius=2))
                            cmap_bnds = utils.generate_imgmask(mask_bnds, None, colour)
                            ax.imshow(cmap, alpha=0.25, zorder=z)
                            ax.imshow(cmap_bnds, alpha=0.75, zorder=z)
                        # Plot OD radius, etc.
                        # ax.plot(fov_od_line[0]+i, fov_od_line[1], "k--", linewidth=2, zorder=3)
                        # ax.plot(fov_od_line[0][intersection_idx[0]:]+i, fov_od_line[1][intersection_idx[0]:], "darkgreen", linewidth=2, zorder=3)
                        # for idx, c in zip(int_idxs[::-1], ["b","r"]):
                        #     ax.plot(fov_od_line[0][idx:intersection_idx[0]]+i, fov_od_line[1][idx:intersection_idx[0]], f"{c}--", linewidth=2, zorder=3)
                        # ax.scatter(od_intersection[0,0]+i, od_intersection[0,1], marker="X", s=100, edgecolors=(0,0,0), c="b", zorder=4)
                
            # ax.imshow(od_cmap, alpha=0.5)
            ax.set_axis_off()
            fig.tight_layout(pad = 0)
            if save_images:
                fig.savefig(os.path.join(save_path, f"{fname}_superimposed.png"), bbox_inches="tight")
            if collate_segmentations:
                segmentation_directory = os.path.join(dirpath, "segmentations")
                if not os.path.exists(segmentation_directory):
                    os.mkdir(segmentation_directory)
                fig.savefig(os.path.join(segmentation_directory, f"{fname}.png"), bbox_inches="tight")
            plt.close()

        # Organise measurements of SLO into dataframe
        slo_df = utils.nested_dict_to_df(slo_dict).reset_index()
        slo_df = slo_df.rename({"level_0":"vessel_map", "level_1":"zone"}, axis=1, inplace=False)
        reorder_cols = ["vessel_map", "zone", "fractal_dimension", "vessel_density", "average_global_calibre", 
                        "average_local_calibre", "tortuosity_density", "CRAE_Knudtson", "CRVE_Knudtson"]
        slo_df = slo_df[reorder_cols]

        # Compute AVR
        slo_df["AVR"] = -1
        all_grids = np.array(list(slo_df.zone.drop_duplicates()))
        avrs = []
        for z in all_grids:
            crae = slo_df[(slo_df.zone==z) & (slo_df.vessel_map=="artery")].iloc[0].CRAE_Knudtson
            crve = slo_df[(slo_df.zone==z) & (slo_df.vessel_map=="vein")].iloc[0].CRVE_Knudtson
            if crae==-1 or crve ==-1:
                avrs.append(-1)
            else:
                avrs.append(crae/crve)
        avrs = np.array(avrs)

        # Outputting warning to user if AVR exceeds 1
        warning_zones = all_grids[avrs > 1]
        if warning_zones.shape[0] > 0:
            if location == "Macula":
                msg = f"WARNING: AVR value exceeds 1, please check artery-vein segmentation."
            elif location == "Optic disc":
                msg = f"WARNING: AVR value exceeds 1 for zones "
                for z in warning_zones[:-1]:
                    msg += f"{z}, "
                msg += f"and {warning_zones[-1]}. Please check artery-vein segmentation."
            if verbose:
                print(msg)
        logging_list.append(msg)

        # add AVR to measurement dataframe
        null_dict = {key:len(all_grids)*[-1] for key in reorder_cols[2:]}
        avr_dict = {**{"vessel_map":len(all_grids)*["artery-vein"], "zone":all_grids},**null_dict, **{"AVR":avrs}}
        avr_df = pd.DataFrame(avr_dict)
        slo_df = pd.concat([slo_df, avr_df], axis=0).reset_index(drop=True)

        # Collect dataframes per zone
        slo_df.loc[slo_df.zone.isin(["B", "C"]), ["fractal_dimension", "vessel_density", "average_global_calibre"]] = -1
        slo_dfs = []
        for z in all_grids:
            df = slo_df[slo_df.zone == z].reset_index(drop=True)
            df = df.iloc[[1,0,2,3]].reset_index(drop=True)
            slo_dfs.append(df)
        
    else:
        msg = f"\n\nSkipping metric calculation."
        if verbose:
            print(msg)
        logging_list.append(msg)
        slo_dfs = None

    # Save out measurements and segmentations
    meta_df = pd.DataFrame(metadata, index=[0])
    if save_results:
        with pd.ExcelWriter(os.path.join(save_path, f'{fname}_output.xlsx')) as writer:
            # write metadata
            meta_df.to_excel(writer, sheet_name='metadata', index=False)
    
            # write SLO measurements
            if slo_dfs is not None: 
                for (df, z) in zip(slo_dfs, all_grids):
                    df.to_excel(writer, sheet_name=f'slo_measurements_{z}', index=False)
    
        msg = f"\n\nSaved out metadata, measurements and segmentations."
        logging_list.append(msg)
        if verbose:
            print(msg)
    
        with open(os.path.join(save_path, f"{fname}_log.txt"), "w") as f:
            for line in logging_list:
                f.write(line+"\n")

    # final log
    msg = f"\n\nCompleted analysis of {fname}.\n\n"
    logging_list.append(msg)
    if verbose:
        print(msg)

    # Return metadata, SLO image, measurements, segmentations and logging
    segmentations = [slo_avimout, slo_vbinmap]
    return meta_df, slo_dfs, slo, segmentations, logging_list
