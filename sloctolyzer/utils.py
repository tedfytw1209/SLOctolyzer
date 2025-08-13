import numpy as np
import skimage as sk
import os
import pandas as pd
import pydicom
import pickle
import matplotlib.pyplot as plt
import SimpleITK as sitk
import eyepy
from datetime import datetime

from PIL import Image, ImageOps
from skimage import segmentation, measure, exposure
from sklearn.linear_model import LinearRegression
from sloctolyzer.measure import tortuosity_measures, slo_measurement
from eyepy.core import utils as eyepy_utils
from eyepy.io.he import vol_reader



def load_volfile(vol_path, logging=[], verbose=False):
    """
    Load and extract pixel and meta data from .vol file

    Returns OCT B-scan data, all relevant metadata and three
    versions of the corresponding IR-SLO image: A pain version,
    one with the fovea-centred B-scan acquisition location superimposed,
    and another with all B-scan acquisition locations superimposed.
    """
    fname_type = os.path.split(vol_path)[1]
    pat_id = fname_type.split('.')[0]
    msg = f"Reading file {fname_type}..."
    logging.append(msg)
    if verbose: 
        print(msg)
    
    # Catch whether .vol file is a peripapillary or macular scan. Other locations, i.e. radial "star-shaped" scans
    # currently not supported.
    try: 
        voldata = eyepy.import_heyex_vol(vol_path)
        scan_type = "Macula"

        # pixel data
        bscan_data = voldata.data / 255
        N_scans, M, N = bscan_data.shape
        fovea_slice_num = N_scans // 2 + 1

    except ValueError as msg:
        if len(msg.args) > 0 and msg.args[0] == "The EyeVolume object does not support scan pattern 2 (one Circular B-scan).":
            voldata = vol_reader.HeVolReader(vol_path)
            scan_type = "Optic disc"
        else:
            logging.append(msg)
            raise msg

    # slo data and metadata
    slo = voldata.localizer.data.astype(float)
    slo_N = slo.shape[0]
    slo_metadict = voldata.localizer.meta.as_dict()
    slo_metadict["pixel_resolution"] = slo_N
    slo_metadict["field_of_view_mm"] = slo_metadict["scale_x"] * slo_N
    
    # bscan metadata
    vol_metadata = voldata.meta.as_dict()
    eye = vol_metadata["laterality"]
    if eye == 'OD':
        eye = 'Right'
    elif eye == 'OS':
        eye = 'Left'
    else:
        eye = 'Unknown'
    scale_x, scale_y, scale_z = vol_metadata["scale_x"], vol_metadata["scale_y"], vol_metadata["scale_z"]
    bscan_meta = vol_metadata["bscan_meta"]
    
    # Detect type of scan
    if scan_type == "Optic disc":
        type = scan_type
        msg = f"Loaded a peripapillary (circular) B-scan and Optic disc IR-SLO."
    elif scan_type == "Macula" and scale_z != 0:
        type = "Ppole"
        msg = f"Loaded a posterior pole scan with {N_scans} B-scans and Macular IR-SLO."
    else:
        stp = bscan_meta[0]["start_pos"][0]
        enp = bscan_meta[0]["end_pos"][1]
        if np.allclose(stp,0,atol=1e-3):
            type = "H-line"
        elif np.allclose(enp,0,atol=1e-3):
            type = "V-line"
        else:
            type = "AV-line"
        msg = f"Loaded a single {type} B-scan."
    logging.append(msg)
    if verbose:
        print(msg)

    # Construct slo-acquisition image and extract quality of B-scan    
    msg = "Accessing IR-SLO and organising metadata..."
    logging.append(msg)
    if verbose:
        print(msg)
    all_mm_points = []
    all_quality = []
    for m in bscan_meta:
        all_quality.append(m["quality"])
        st = m["start_pos"]
        en = m["end_pos"]
        point = np.array([st, en])
        all_mm_points.append(point)

    # Only relevant for Ppole data
    quality_mu = np.mean(all_quality)
    quality_sig = np.std(all_quality)
    
    # Convert start and end B-scan locations from mm to pixel
    all_px_points = []
    for point in all_mm_points:
        all_px_points.append(slo_N * point / slo_metadict["field_of_view_mm"])

    # Draw the acquisition locations onto the SLO
    slo_at_fovea = np.concatenate(3*[slo[...,np.newaxis]], axis=-1)
    slo_acq = slo_at_fovea.copy()

    # For peripapillary scans, we draw a circular ROI
    if scan_type == "Optic disc":
        peripapillary_coords = all_px_points[0].astype(int)
        
        if eye == "Right":
            OD_center, OD_edge = peripapillary_coords[peripapillary_coords[:,0].argsort()]
        elif eye == "Left":
            OD_edge, OD_center = peripapillary_coords[peripapillary_coords[:,0].argsort()]

        circular_radius = np.abs(OD_center[0] - OD_edge[0])
        circular_mask,_ = slo_measurement._create_circular_mask(img_shape=(slo_N,slo_N), 
                                                             center=OD_center, 
                                                             radius=circular_radius)
        circular_bnd_mask = segmentation.find_boundaries(circular_mask)
        slo_acq[circular_bnd_mask,:] = 0
        slo_acq[circular_bnd_mask,1] = 1
        slo_at_fovea = slo_acq.copy()
        #slo_metadict["stxy_coord"] = f"{OD_edge[0]},{OD_edge[1]}"
        slo_metadict["od_radius_px"] = circular_radius
        slo_metadict["od_center_x"] = OD_center[0]
        slo_metadict["od_center_y"] = OD_center[1]
        #slo_metadict["roi_radius_mm"] = np.round(circular_radius*slo_metadict["scale_x"],2)

    else:
        # For macular scans, we generate line for each B-scan location and 
        # superimpose acquisition line onto copied SLO. Create one with all
        #  acquisition lines, and one with only
        # the fovea. slo_at_fov only used when N_scans > 1
        for idx, point in enumerate(all_px_points):
            x_idx, y_idx = [[1,0], [0,1]][type != "V-line"]
            X, y = point[:,x_idx].reshape(-1,1), point[:,y_idx]
            linmod = LinearRegression().fit(X, y)
            x_grid = np.linspace(X[0,0], X[1,0], 800).astype(int)
            x_grid = x_grid[(x_grid < slo_N) & (x_grid >= 0)]
            y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
            x_grid = x_grid[y_grid < slo_N]
            y_grid = y_grid[y_grid < slo_N]
            for (x,y) in zip(x_grid, y_grid):
                x_idx, y_idx = [[y,x], [x,y]][type != "V-line"]
                slo_acq[y_idx, x_idx, :] = 0
                slo_acq[y_idx, x_idx, 1] = 1
                if (idx+1) == fovea_slice_num:
                    slo_at_fovea[y_idx, x_idx, :] = 0
                    slo_at_fovea[y_idx, x_idx, 1] = 1
    # Create DataFrame of metadata
    bscan_metadict = {}
    bscan_metadict["Filename"] = fname_type
    bscan_metadict["eye"] = eye
    bscan_metadict["scale_units"] = "microns_per_pixel"
    bscan_metadict["avg_quality"] = quality_mu

    # Remove duplicates: store scales as microns-per-pixel, laterality=eye
    slo_metadict["scale"] = 1e3*slo_metadict["scale_x"]
    for key in ["laterality", "scale_x", "scale_y", "scale_unit"]:
        del slo_metadict[key]
    slo_metadict["location"] = scan_type
    slo_metadict["field_size_degrees"] = slo_metadict.pop("field_size")
    #slo_metadict["slo_modality"] = slo_metadict.pop("modality")
        
    # Combine metadata and return with data
    metadata = {**bscan_metadict, **slo_metadict}
    msg = "Done!"
    logging.append(msg)
    if verbose:
        print(msg)
        
    return slo, metadata, logging

def load_dcmfile(dcm_oct_path, dcm_slo_path, logging=[], verbose=False):
    """
    Load and extract pixel and meta data from .dcm file

    Returns OCT B-scan data, all relevant metadata and three
    versions of the corresponding IR-SLO image: A pain version,
    one with the fovea-centred B-scan acquisition location superimposed,
    and another with all B-scan acquisition locations superimposed.
    """
    fname_type = os.path.split(dcm_oct_path)[1]
    pat_id = fname_type.split('.')[0]
    msg = f"Reading file {fname_type}..."
    logging.append(msg)
    if verbose: 
        print(msg)

    # Catch whether .dcm file is a peripapillary or macular scan. Other locations, i.e. radial "star-shaped" scans
    # currently not supported.
    scan_type = "Macula"
    try: 
        #voldata = eyepy.import_heyex_vol(vol_path)
        #(25, 496, 512)
        voldata = pydicom.dcmread(dcm_oct_path)
        # pixel data
        bscan_data = voldata.pixel_array / 255
        N_scans, M, N = bscan_data.shape
        fovea_slice_num = N_scans // 2
        
    except ValueError as msg:
        logging.append(msg)
        raise msg

    # slo data and metadata
    slo_voldata = pydicom.dcmread(dcm_slo_path)
    slo = slo_voldata.pixel_array.astype(float) / 255 #TODO:need check
    slo_N = slo.shape[0]
    slo_metadict = {
        "modality": slo_voldata.Modality,
        "sop_class": str(slo_voldata.SOPClassUID),
        "num_frames": slo_voldata.NumberOfFrames or 1,
        "rows": slo_voldata.Rows, "cols": slo_voldata.Columns,
        "scale_x": slo_voldata.PixelSpacing[0], #row spacing
        "scale_y": slo_voldata.PixelSpacing[1], #column spacing
        "laterality": slo_voldata.ImageLaterality,
        "manufacturer": slo_voldata.get("Manufacturer", None),
        "field_size": slo_voldata.get("HorizontalFieldOfView", None),
    }
    slo_metadict["slo_resolution_px"] = slo_N #e.g. 768
    slo_metadict["field_of_view_mm"] = slo_metadict["scale_x"] * slo_N #e.g. 768*0.011811=9.07
    
    # Extract dates
    try:
        visit_date = datetime.strptime(voldata.StudyDate, "%Y%m%d").isoformat()
    except:
        visit_date = "Unknown"
    try:
        exam_time = datetime.strptime(voldata.StudyDate + voldata.ContentTime.split(".")[0], "%Y%m%d%H%M%S").isoformat()
    except:
        exam_time = "Unknown"
    # bscan metadata
    pixel_spacing = voldata['SharedFunctionalGroupsSequence'][0]['PixelMeasuresSequence'][0]['PixelSpacing'].value
    slice_thickness = float(voldata['SharedFunctionalGroupsSequence'][0]['PixelMeasuresSequence'][0]['SliceThickness'].value)
    vol_metadata = {
        'Filename': fname_type,
        "modality": voldata.Modality,
        "sop_class": str(voldata.SOPClassUID),
        "sop_instance_uid": str(voldata.SOPInstanceUID),
        "study_instance_uid": str(voldata.StudyInstanceUID),
        "series_instance_uid": str(voldata.SeriesInstanceUID),
        "num_frames": int(voldata.NumberOfFrames) or 1,
        "rows": voldata.Rows,
        "cols": voldata.Columns,
        "scale_y": pixel_spacing[0], # e.g. 0.003872, row spacing
        "scale_x": pixel_spacing[1], # e.g. 0.011811, column spacing
        "scale_z": slice_thickness, # e.g. 0.251631
        "slice_thickness_mm": slice_thickness, # e.g. 0.251631
        "laterality": voldata.ImageLaterality,
        "manufacturer": voldata.get("Manufacturer", None),
        'scale_units': 'microns_per_pixel',
        'retinal_layers_N': 2,   # placeholder
        'scan_focus': -1.77,     # placeholder
        'visit_date': visit_date,
        'exam_time': exam_time
    }
    eye = vol_metadata["laterality"]
    if eye == 'OD':
        eye = 'Right'
    elif eye == 'OS':
        eye = 'Left'
    else:
        eye = 'Unknown'
    scale_z, scale_x, scale_y = vol_metadata["scale_z"], vol_metadata["scale_x"], vol_metadata["scale_y"]
    bscan_meta = voldata['PerFrameFunctionalGroupsSequence']
    
    # Detect type of scan
    if scan_type == "Optic disc":
        type = scan_type
        msg = f"Loaded a peripapillary (circular) B-scan and Optic disc IR-SLO."
    elif scan_type == "Macula" and scale_z != 0:
        type = "Ppole"
        msg = f"Loaded a posterior pole scan with {N_scans} B-scans and Macular IR-SLO."
    else:
        stp = bscan_meta[0]["start_pos"][0]
        enp = bscan_meta[0]["end_pos"][1]
        if np.allclose(stp,0,atol=1e-3):
            type = "H-line"
        elif np.allclose(enp,0,atol=1e-3):
            type = "V-line"
        else:
            type = "AV-line"
        msg = f"Loaded a single {type} B-scan."
    logging.append(msg)
    if verbose:
        print(msg)

    # Construct slo-acquisition image and extract quality of B-scan    
    msg = "Accessing IR-SLO and organising metadata..."
    logging.append(msg)
    if verbose:
        print(msg)
    all_mm_points = []
    all_quality = [0,0,0]
    for m in bscan_meta:
        img_position = m["PlanePositionSequence"][0]["ImagePositionPatient"].value
        st = (img_position[0], img_position[2])
        en = (img_position[0] + scale_x*N, img_position[2])
        point = np.array([st, en])
        all_mm_points.append(point)

    # Only relevant for Ppole data
    quality_mu = np.mean(all_quality)
    quality_sig = np.std(all_quality)
    
    # Convert start and end B-scan locations from mm to pixel
    all_px_points = []
    for point in all_mm_points:
        all_px_points.append(slo_N * point / slo_metadict["field_of_view_mm"])
    all_px_points = np.array(all_px_points)
    
    # Draw the acquisition locations onto the SLO
    slo_at_fovea = np.concatenate(3*[slo[...,np.newaxis]], axis=-1)
    slo_acq = slo_at_fovea.copy()

    # For peripapillary scans, we draw a circular ROI
    if scan_type == "Optic disc":
        peripapillary_coords = all_px_points[0].astype(int)
        
        if eye == "Right":
            OD_center, OD_edge = peripapillary_coords[peripapillary_coords[:,0].argsort()]
        elif eye == "Left":
            OD_edge, OD_center = peripapillary_coords[peripapillary_coords[:,0].argsort()]

        circular_radius = np.abs(OD_center[0] - OD_edge[0])
        circular_mask,_ = slo_measurement._create_circular_mask(img_shape=(slo_N,slo_N), 
                                                             center=OD_center, 
                                                             radius=circular_radius)
        circular_bnd_mask = segmentation.find_boundaries(circular_mask)
        slo_acq[circular_bnd_mask,:] = 0
        slo_acq[circular_bnd_mask,1] = 1
        slo_at_fovea = slo_acq.copy()
        #slo_metadict["stxy_coord"] = f"{OD_edge[0]},{OD_edge[1]}"
        slo_metadict["od_radius_px"] = circular_radius
        slo_metadict["od_center_x"] = OD_center[0]
        slo_metadict["od_center_y"] = OD_center[1]
        #slo_metadict["roi_radius_mm"] = np.round(circular_radius*slo_metadict["scale_x"],2)
    else:
        # For macular scans, we generate line for each B-scan location and 
        # superimpose acquisition line onto copied SLO. Create one with all
        #  acquisition lines, and one with only
        # the fovea. slo_at_fov only used when N_scans > 1
        for idx, point in enumerate(all_px_points):
            x_idx, y_idx = [[1,0], [0,1]][type != "V-line"]
            X, y = point[:,x_idx].reshape(-1,1), point[:,y_idx]
            linmod = LinearRegression().fit(X, y)
            x_grid = np.linspace(X[0,0], X[1,0], 800).astype(int)
            x_grid = x_grid[(x_grid < slo_N) & (x_grid >= 0)]
            y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
            x_grid = x_grid[y_grid < slo_N]
            y_grid = y_grid[y_grid < slo_N]
            for (x,y) in zip(x_grid, y_grid):
                x_idx, y_idx = [[y,x], [x,y]][type != "V-line"]
                slo_acq[y_idx, x_idx, :] = 0
                slo_acq[y_idx, x_idx, 1] = 1
                if (idx+1) == fovea_slice_num:
                    slo_at_fovea[y_idx, x_idx, :] = 0
                    slo_at_fovea[y_idx, x_idx, 1] = 1
    # Create DataFrame of metadata
    bscan_metadict = {}
    bscan_metadict["Filename"] = fname_type
    bscan_metadict["eye"] = eye
    bscan_metadict["scale_units"] = "microns_per_pixel"
    bscan_metadict["avg_quality"] = quality_mu

    # Remove duplicates: store scales as microns-per-pixel, laterality=eye
    slo_metadict["scale"] = 1e3*slo_metadict["scale_x"]
    for key in ["laterality", "scale_x", "scale_y", "scale_unit"]:
        del slo_metadict[key]
    slo_metadict["location"] = scan_type
    slo_metadict["field_size_degrees"] = slo_metadict.pop("field_size")
    #slo_metadict["slo_modality"] = slo_metadict.pop("modality")
        
    # Combine metadata and return with data
    metadata = {**bscan_metadict, **slo_metadict}
    msg = "Done!"
    logging.append(msg)
    if verbose:
        print(msg)
        
    return slo, metadata, logging

def load_img(path, ycutoff=0, xcutoff=0, pad=False, pad_factor=32,
             dtype=np.uint8, normalise=False, grayscale=True):
    '''
    Helper function to load in image and crop
    '''
    img = Image.open(path)
    if grayscale:
        img = ImageOps.grayscale(img)
    img = np.array(img)[ycutoff:, xcutoff:]
    if normalise:
        img -= img.min()
        img /= img.max()
    img = img.astype(dtype)
    
    # If padding to dimensions divisible bvy 32
    if pad:
        ndim = img.ndim
        M, N = img.shape[:2]
        pad_M = (pad_factor - M%pad_factor) % pad_factor
        pad_N = (pad_factor - N%pad_factor) % pad_factor
        if ndim == 2:
            return np.pad(img, ((0, pad_M), (0, pad_N)))
        else: 
            return np.pad(img, ((0, pad_M), (0, pad_N), (0,0)))

    return img



def plot_img(img_data, traces=None, cmap=None, fovea=None, save_path=None, 
             fname=None, sidebyside=False, rnfl=False):
    '''
    Helper function to plot the result - plot the image, traces, colourmap, etc.
    '''
    img = img_data.copy().astype(np.float64)
    img -= img.min()
    img /= img.max()
    M, N = img.shape
    
    if rnfl:
        figsize=(15,6)
    else:
        figsize=(6,6)

    if sidebyside:
        figsize = (2*figsize[0], figsize[1])
    
    if sidebyside:
        fig, (ax0, ax) = plt.subplots(1,2,figsize=figsize)
        ax0.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
    fontsize=16
    if traces is not None:
        if len(traces) == 2:
            for tr in traces:
                 ax.plot(tr[:,0], tr[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)
        else:
            ax.plot(traces[:,0], traces[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)

    if cmap is not None:
        cmap_data = cmap.copy().astype(np.float64)
        cmap_data -= cmap_data.min()
        cmap_data /= cmap_data.max()
        ax.imshow(cmap_data, alpha=0.5, zorder=2)
    if fname is not None:
        ax.set_title(fname, fontsize=15)

    if fovea is not None:
        ax.scatter(fovea[0], fovea[1], color="r", edgecolors=(0,0,0), marker="X", s=50, linewidth=1)
            
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    if save_path is not None and fname is not None:
        ax.set_title(None)
        fig.savefig(os.path.join(save_path, f"{fname}.png"), bbox_inches="tight")



def generate_imgmask(mask, thresh=None, cmap=0):
    '''
    Given a prediction mask Returns a plottable mask
    '''
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot
    

def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = sk.measure.label(binmask)                       
    regions = sk.measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


def compute_opticdisc_radius(fovea, od_centre, od_mask):
    """
    Work out optic disc radius in pixels, according to it's position relative to the fovea.
    """
    # Extract Optic disc mask and acquisition line boundary,
    od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    #od_mask_radius = od_mask_props.axis_minor_length/2 # naive radius, without account for orientation with fovea
    #od_mask_radius = od_mask_props.axis_major_length/2
    od_boundary = segmentation.find_boundaries(od_mask)

    # Work out reference line between fovea and  optic disc center. 
    Xy =  np.concatenate([od_centre[np.newaxis], fovea[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])
    x_grid = np.arange(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0])).astype(int)
    y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
    
    # Intersection of reference line and optic disc boundary
    intersection_idx = np.argwhere(od_boundary[(y_grid, x_grid)] == 1)[0]
    od_intersection = np.array((x_grid[intersection_idx], 
                                y_grid[intersection_idx])).reshape(1,-1)

    # Now we can work out the optic disc radius, according to it's position with the fovea
    od_bounds = np.concatenate([od_centre.reshape(1,-1), od_intersection], axis=0)
    od_radius = tortuosity_measures._curve_length(od_bounds[:,0], od_bounds[:,1])

    plot_info = (od_intersection, (x_grid, y_grid), intersection_idx)

    return np.round(od_radius).astype(int), plot_info


def _process_opticdisc(od_mask):
    """
    Work out optic disc radius in pixels, according to it's position relative to the fovea.
    """
    # Extract Optic disc radius and OD boundary if detected
    try:
        od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    except:
        return None, np.zeros_like(od_mask)
    od_radius = int((od_mask_props.axis_minor_length + od_mask_props.axis_major_length)/4)
    od_boundary = segmentation.find_boundaries(od_mask)

    return od_radius, od_boundary


def normalise(img, 
              minmax_val=(0,1), 
              astyp=np.float64):
    '''
    Normalise image between minmax_val.

    INPUTS:
    ----------------
        img (np.array, dtype=?) : Input image of some data type.

        minmax_val (tuple) : Tuple storing minimum and maximum value to normalise image with.

        astyp (data type) : What data type to store normalised image.
    
    RETURNS:
    ----------------
        img (np.array, dtype=astyp) : Normalised image in minmax_val.
    '''
    # Extract minimum and maximum values
    min_val, max_val = minmax_val

    # Convert to float type to perform [0, 1] normalisation
    img = img.astype(np.float64)

    # Normalise to [0, 1]
    img -= img.min()
    img /= img.max()

    # Rescale to max_val and output as specified data type
    img *= (max_val - min_val)
    img += min_val

    return img.astype(astyp)


def flatten_dict(nested_dict):
    '''
    Recursive flattening of a dictionary of dictionaries.
    '''
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    '''
    Nested dictionary is flattened and converted into an index-wise, multi-level Pandas DataFrame
    '''
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def load_annotation(path, key=None, raw=False, binary=False):
    """Load in .nii.gz file and output region and vessel masks"""
    
    # Read the .nii image containing thevsegmentations
    sitk_t1 = sitk.ReadImage(path)

    if key is None:
        a_i = 191
        od_i = 255
        v_i = 127
    else:
        a_i, v_i, od_i = key
        
    # and access the numpy array, saved as (1, N, N)
    segmentations = sitk.GetArrayFromImage(sitk_t1)[0]

    # returning raw segmentation map
    if raw:
        return segmentations

    # if binary, only return vessels with label 255
    if binary:
        return (segmentations == od_i).astype(int)

    # for artery-vein-optic disc map
    artery = (segmentations == a_i)
    OD = (segmentations == od_i)
    vein = (segmentations == v_i)
    mask = (segmentations > 0)
    cmap = np.concatenate([artery[...,np.newaxis], 
                           OD[...,np.newaxis],
                           vein[...,np.newaxis], 
                           mask[...,np.newaxis]], axis=-1).astype(int)
    
    
    return cmap



def print_error(fname, e):
    '''
    If robust_run is 1 and an unexpected error occurs, this will be printed out and also saved to the log.

    A detailed explanation of the error found.
    '''
    user_fail = f"\nFailed to analyse {fname}."
    message = f"\nAn exception of type {type(e).__name__} occurred. Error description:\n{str(e)}\n"
    print(user_fail)
    print(message)
    trace = ["Full traceback:\n"]
    print(trace[0])
    tb = e.__traceback__
    tb_i = 1
    while tb is not None:
        tb_fname = tb.tb_frame.f_code.co_filename
        tb_func = tb.tb_frame.f_code.co_name
        tb_lineno = tb.tb_lineno
        tb_str = f"Traceback {tb_i} to filename\n{tb_fname}\nfor function {tb_func}(...) at line {tb_lineno}.\n"
        print(tb_str)
        trace.append(tb_str)
        tb = tb.tb_next
        tb_i += 1

    skip = "Skipping and moving to next file.\nCheck data input and/or set robust_run to 0 to debug code.\n"
    print(skip)
    logging_list = [user_fail, message] + trace + [skip]
    return logging_list