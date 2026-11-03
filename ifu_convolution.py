"""
ifu_convolution.py

Utilities for generating JWST PSFs, creating wavelength-dependent convolution
kernels, stacking kernel cubes, and convolving IFU science cubes.

Notes
-----
- Much of this code was written by Tom Williams and Ryan Chown, specifically the kernel creation, convolution, and regridding codes
- Their code was adapted to include IFU PSF generation and to convolve IFU cubes by Hannah Koziol
- The Aniano kernel class and its image-processing helpers are included below.
- STPSF instrument configuration can depend on the observing mode and installed
  STPSF version. Check the relevant instrument settings before large runs.
"""

# =============================================================================
# Imports
# =============================================================================

import csv
import glob
import os
import copy
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.convolution import (
    convolve, convolve_fft, Gaussian2DKernel, interpolate_replace_nans,
)
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom, rotate
from photutils.centroids import centroid_com

import stpsf

from reproject import reproject_interp, reproject_exact, reproject_adaptive


ALLOWED_REPROJECT_FUNCS = ("interp", "exact", "adaptive")


# =============================================================================
# Aniano-kernel image-processing helpers
# =============================================================================

def interp_nans(data, x_stddev=2):
    """Interpolate over NaNs using an Astropy Gaussian kernel."""
    data = np.asarray(data, dtype=float)
    if not np.isnan(data).any():
        return data.copy()
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    return interpolate_replace_nans(data, kernel)


def resample(data, source_pixscale, target_pixscale, interp_order=3):
    """Resample a 2D image to a new pixel scale and force odd dimensions."""
    if source_pixscale <= 0 or target_pixscale <= 0:
        raise ValueError("Pixel scales must be positive.")
    data_resample = zoom(
        np.asarray(data), source_pixscale / target_pixscale, order=interp_order
    )
    if data_resample.shape[0] % 2 == 0:
        data_resample = data_resample[:-1, :]
    if data_resample.shape[1] % 2 == 0:
        data_resample = data_resample[:, :-1]
    return data_resample


def centroid(data):
    """Shift a 2D image so its center of mass is at the central pixel."""
    data = np.asarray(data).copy()
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2
    # centroid_com expects finite, non-negative weights for reliable results.
    weights = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.nansum(weights) == 0:
        return data
    j_centroid, i_centroid = centroid_com(weights)
    i_shift = int(np.round(i_cen - i_centroid))
    j_shift = int(np.round(j_cen - j_centroid))
    return np.roll(np.roll(data, i_shift, axis=0), j_shift, axis=1)


def circularize(data, rotations=14):
    """Approximate rotational symmetry by repeatedly averaging rotated copies."""
    data = np.asarray(data, dtype=float).copy()
    for n in range(rotations, 0, -1):
        data_rotate = rotate(data, 360 / (2 ** n), order=3, reshape=False)
        data = 0.5 * (data + data_rotate)

    radius = min(data.shape) / 2
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2
    jj, ii = np.meshgrid(np.arange(data.shape[1]) - j_cen,
                          np.arange(data.shape[0]) - i_cen)
    data[jj**2 + ii**2 > radius**2] = 0
    data[np.abs(data) < np.finfo(float).eps] = 0
    return data


def _center_crop(data, shape):
    """Return the centered crop of a 2D array with the requested shape."""
    y, x = data.shape
    ny, nx = int(shape[0]), int(shape[1])
    y0, x0 = max(0, (y - ny) // 2), max(0, (x - nx) // 2)
    return data[y0:y0 + min(y, ny), x0:x0 + min(x, nx)]


def _center_pad(data, shape):
    """Zero-pad a 2D array symmetrically to the requested shape."""
    ny, nx = int(shape[0]), int(shape[1])
    if ny < data.shape[0] or nx < data.shape[1]:
        raise ValueError("Padding target must be at least as large as input.")
    out = np.zeros((ny, nx), dtype=data.dtype)
    y0 = (ny - data.shape[0]) // 2
    x0 = (nx - data.shape[1]) // 2
    out[y0:y0 + data.shape[0], x0:x0 + data.shape[1]] = data
    return out


def resize(data, pixscale, grid_size_arcsec=None):
    """Center-crop or zero-pad a 2D image to a target angular grid size."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("resize expects a 2D array.")
    if pixscale <= 0:
        raise ValueError("pixscale must be positive.")
    if grid_size_arcsec is None:
        grid_size_arcsec = np.array([729.0, 729.0])
    grid_size_pix = np.maximum(1, np.asarray(grid_size_arcsec, dtype=float) / pixscale).astype(int)
    # Ensure both target dimensions are odd, as required for centered kernels.
    grid_size_pix += (grid_size_pix % 2 == 0)
    if np.all(data.shape > grid_size_pix):
        return _center_crop(data, grid_size_pix)
    if np.all(data.shape < grid_size_pix):
        return _center_pad(data, grid_size_pix)
    if np.all(data.shape == grid_size_pix):
        return data.copy()
    # Handle mixed crop/pad dimensions, rather than failing for rectangular arrays.
    cropped = _center_crop(data, np.minimum(data.shape, grid_size_pix))
    return _center_pad(cropped, grid_size_pix)


def high_pass_filter(data, fwhm, pixscale=0.1):
    """Apply the Aniano-style high-pass filter in Fourier space."""
    k_b = 8 * np.pi / (fwhm * pixscale)
    k_a = 0.9 * k_b
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2
    jj, ii = np.meshgrid(np.arange(data.shape[1]) - j_cen,
                          np.arange(data.shape[0]) - i_cen)
    ri = np.sqrt(jj**2 + ii**2) * pixscale
    filt = np.zeros(data.shape, dtype=float)
    filt[ri <= k_a] = 1
    idx = (k_a < ri) & (ri <= k_b)
    filt[idx] = np.exp(-(1.8249 * (ri[idx] - k_a) / (k_b - k_a)) ** 4)
    return filt * data


def low_pass_filter(data, pixscale=0.1):
    """Create the Aniano-style low-pass filter for a Fourier-domain image."""
    i_range, j_range = data.shape
    data_slice = data[int((i_range - 1) / 2):, int((j_range - 1) / 2)]
    data_slice_max = np.nanmax(data_slice)
    below = np.where(data_slice < 0.005 * data_slice_max)[0]
    if len(below):
        k_h = below[0] * pixscale
    else:
        print("k_h too large. Something is probably up with kernel generation...")
        k_h = len(data_slice) * pixscale
    k_l = 0.7 * k_h
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2
    jj, ii = np.meshgrid(np.arange(data.shape[1]) - j_cen,
                          np.arange(data.shape[0]) - i_cen)
    ri = np.sqrt(jj**2 + ii**2) * pixscale
    filt = np.zeros(data.shape, dtype=float)
    if k_h > k_l:
        idx = (k_l < ri) & (ri <= k_h)
        filt[idx] = 0.5 * (1 + np.cos(np.pi * (ri[idx] - k_l) / (k_h - k_l)))
    filt[ri <= k_l] = 1
    return filt


def trim_kernel_energy(kernel, energy_tol=0.999):
    """Trim a centered kernel to the smallest radius enclosing energy_tol."""
    kernel = np.asarray(kernel)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("trim_kernel_energy expects a square 2D kernel.")
    if not 0 < energy_tol <= 1:
        raise ValueError("energy_tol must be in (0, 1].")
    kernel_radius = int((kernel.shape[0] - 1) / 2)
    i_cen = (kernel.shape[0] - 1) / 2
    j_cen = (kernel.shape[1] - 1) / 2
    jj, ii = np.meshgrid(np.arange(kernel.shape[1]) - j_cen,
                          np.arange(kernel.shape[0]) - i_cen)
    ri = np.sqrt(jj**2 + ii**2)
    total = np.nansum(np.abs(kernel[ri <= kernel_radius]))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Kernel has no finite enclosed energy.")
    radius = kernel_radius
    for candidate in range(kernel_radius + 1):
        enclosed = np.nansum(np.abs(kernel[ri <= candidate]))
        if enclosed / total >= energy_tol:
            radius = candidate
            break
    size = 2 * radius + 1
    return _center_crop(kernel, (size, size))

# =============================================================================
# General helpers
# =============================================================================

PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1', 'PIXELSCL']
def get_pixscale(hdu):
    """Get pixel scale from header.

    Checks HDU header for pixel scale keywords, and returns a pixel scale in arcsec. If no suitable keyword is found,
    will throw up an error.

    Args:
        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:
        pix_scale (float): Pixel scale in arcsec.

    Raises:
        Warning: If no suitable pixel scale keyword is found in header.

    """
    if isinstance(hdu, (fits.hdu.image.PrimaryHDU, fits.hdu.image.ImageHDU) ):
        header =hdu.header
    elif isinstance(hdu, fits.header.Header ):
        header =hdu
    
    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


def get_cube_wavelengths(cube_hdu, unit=u.micron):
    """
    Return the wavelength at each spectral plane using the SCI WCS.

    Parameters
    ----------
    cube_hdu : HDUList or HDU
        A JWST-style FITS file with a SCI extension, or the SCI HDU itself.
    unit : astropy.units.Unit
        Desired wavelength unit. Defaults to microns.

    Returns
    -------
    astropy.units.Quantity
        One wavelength per spectral plane.
    """
    sci = cube_hdu["SCI"] if hasattr(cube_hdu, "__getitem__") and "SCI" in cube_hdu else cube_hdu
    data = sci.data
    if data is None or data.ndim != 3:
        raise ValueError("Expected a 3D SCI cube with shape (nwave, ny, nx).")

    wcs = WCS(sci.header)
    pix = np.arange(data.shape[0])
    _, _, spectral_world = wcs.all_pix2world(
        np.zeros(data.shape[0]),
        np.zeros(data.shape[0]),
        pix,
        0,
    )

    # FITS WCS may express the spectral axis in metres, microns, or another unit.
    spectral_unit = wcs.wcs.cunit[2]
    if not spectral_unit:
        spectral_unit = u.m
    else:
        spectral_unit = u.Unit(spectral_unit)

    return (spectral_world * spectral_unit).to(unit)


# =============================================================================
# PSF generation
# =============================================================================

def _make_stpsf_instrument(instrument):
    """Construct the requested STPSF instrument object."""
    instrument = instrument.strip().lower()

    if instrument == "nircam":
        return stpsf.NIRCam()
    if instrument == "nirspec":
        return stpsf.NIRSpec()
    if instrument == "miri":
        return stpsf.MIRI()

    raise ValueError("instrument must be 'NIRCam', 'NIRSpec', or 'MIRI'.")


def generate_psf(
    instrument,
    resolution,
    output_file,
    cube_file=None,
    fov_arcsec=None,
    oversample=5,
    pixel_scale=None,
    filter_name=None,
    disperser=None,
    mode=None,
    wavelengths=None,
    overwrite=False,
):
    """
    Generate a PSF or PSF cube using STPSF.

    Parameters
    ----------
    instrument : {'NIRCam', 'NIRSpec', 'MIRI'}
        STPSF instrument to use.
    resolution : str
        Resolution label. Labels beginning with 'F' are treated as photometric
        filters; other labels are treated as IFU/grating configurations.
    output_file : str
        Destination FITS path.
    cube_file : str, optional
        Science cube used to derive wavelength sampling and pixel scale for IFU
        PSF cubes. Expected to contain a SCI extension.
    fov_arcsec : float, optional
        Requested IFU PSF field of view in arcsec. If omitted, STPSF's instrument default
        is used; instrument/channel-specific defaults should be verified for
        the intended observing mode.
    oversample : int
        STPSF oversampling factor.
    pixel_scale : float, optional
        Optional detector/PSF pixel scale in arcsec/pixel. Applied when
        explicitly provided and supported by the selected instrument.
    filter_name : str, optional
        Explicit STPSF filter override. Defaults to `resolution` for filters
        and can be set separately for IFU configurations (e.g. F100LP).
    disperser : str, optional
        Spectral disperser/grating (e.g. G140M).
    mode : str, optional
        Instrument mode override. For NIRSpec IFU this is normally 'IFU'.
    wavelengths : Quantity or array, optional
        Explicit wavelength sampling. If omitted for an IFU cube, derive it from
        the SCI WCS. Numeric values are assumed to be metres for STPSF.
    overwrite : bool
        Whether to overwrite an existing output file.

    Returns
    -------
    str
        Path to the written PSF FITS file.

    Notes
    -----
    This function uses calc_psf for a photometric filter and calc_datacube for
    an IFU wavelength sequence. Some STPSF instrument/mode combinations expose
    different settings; confirm the instrument-specific filter, disperser, and
    FOV requirements for your installed STPSF version.
    """
    if os.path.exists(output_file) and not overwrite:
        print(f"PSF already exists; skipping: {output_file}")
        return output_file

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    is_photometric_filter = str(resolution).upper().startswith("F")
    inst = _make_stpsf_instrument(instrument)

    if filter_name is not None:
        inst.filter = filter_name
    elif is_photometric_filter:
        inst.filter = resolution

    if mode is not None:
        inst.mode = mode
    elif not is_photometric_filter and instrument.strip().lower() == "nirspec":
        inst.mode = "IFU"

    if disperser is not None:
        inst.disperser = disperser
    elif not is_photometric_filter and hasattr(inst, "disperser"):
        # Use the resolution label as the disperser if no explicit value supplied.
        inst.disperser = resolution

    if pixel_scale is not None:
        inst.pixelscale = float(pixel_scale)

    # Do not set fov_arcsec for photometric filters.
    if not is_photometric_filter and fov_arcsec is not None:
        inst.fov_arcsec = float(fov_arcsec)

    if is_photometric_filter:
        psf_hdul = inst.calc_psf(oversample=oversample)
        psf_hdul.writeto(output_file, overwrite=overwrite)
        psf_hdul.close()
        print(f"Wrote photometric PSF: {output_file}")
        return output_file

    if wavelengths is None:
        if cube_file is None:
            raise ValueError(
                "For an IFU PSF cube, supply either `wavelengths` or `cube_file`."
            )
        with fits.open(cube_file) as cube:
            wavelengths = get_cube_wavelengths(cube, unit=u.m)

    if not isinstance(wavelengths, u.Quantity):
        # Matches the metre convention in the supplied workflow.
        wavelengths = np.asarray(wavelengths) * u.m

    psf_cube = inst.calc_datacube(wavelengths=wavelengths, oversample=oversample)
    psf_cube.writeto(output_file, overwrite=overwrite)
    psf_cube.close()
    print(f"Wrote IFU PSF cube: {output_file}")
    return output_file


# =============================================================================
# Kernel generation
# =============================================================================

def make_kernel_cube(
    source_cube,
    source_wavelengths,
    target_psf,
    target_fwhm,
    source_name,
    output_dir,
    source_pixscale=0.01,
    target_pixscale=0.35,
    target_name="target",
    skip_existing=True,
):
    """
    Make a convolution kernel for every slice of a source PSF cube.

    Requires `MakeConvolutionKernel` and `evaluate_kernel` to be defined in or
    imported into this module.
    """
    if "MakeConvolutionKernel" not in globals():
        raise ImportError(
            "MakeConvolutionKernel is not defined. Add its implementation to "
            "this file or import it from your kernel-generation module."
        )
    if "evaluate_kernel" not in globals():
        raise ImportError(
            "evaluate_kernel is not defined. Add its implementation to this "
            "file or import it from your kernel-generation module."
        )

    source_cube = np.asarray(source_cube)
    source_wavelengths = np.asarray(
        source_wavelengths.to_value(u.micron)
        if isinstance(source_wavelengths, u.Quantity)
        else source_wavelengths
    )

    if source_cube.ndim != 3:
        raise ValueError("source_cube must have shape (nwave, ny, nx).")
    if len(source_wavelengths) != source_cube.shape[0]:
        raise ValueError("source_wavelengths must have one value per PSF slice.")

    os.makedirs(output_dir, exist_ok=True)
    diagnostics_file = os.path.join(output_dir, "diagnostics.csv")

    if not os.path.exists(diagnostics_file):
        with open(diagnostics_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "slice", "wavelength", "source_fwhm", "target_fwhm",
                "Wminus", "D", "kernel_file"
            ])

    grid_size = np.array(target_psf.shape, dtype=float) * target_pixscale

    for i, wave in enumerate(source_wavelengths):
        outfile = os.path.join(output_dir, f"kernel_{i:04d}.fits")

        if skip_existing and os.path.exists(outfile):
            print(f"Skipping existing kernel {i}: {outfile}")
            continue

        source_psf = source_cube[i]
        source_fwhm = fit_2d_gaussian(source_psf, pixscale=source_pixscale)

        kernel = MakeConvolutionKernel(
            source_psf=source_psf,
            source_pixscale=source_pixscale,
            source_name=f"{source_name}_{i:04d}",
            target_psf=target_psf,
            target_pixscale=target_pixscale,
            target_name=target_name,
            source_fwhm=source_fwhm,
            target_fwhm=target_fwhm,
            common_pixscale=source_pixscale,
            grid_size_arcsec=grid_size,
            verbose=True,
        )
        kernel.make_convolution_kernel()

        fits.PrimaryHDU(np.asarray(kernel.kernel)).writeto(
            outfile, overwrite=True
        )
        d_value, wminus = evaluate_kernel(kernel)

        with open(diagnostics_file, "a", newline="") as f:
            csv.writer(f).writerow([
                i, wave, source_fwhm, target_fwhm, wminus, d_value,
                os.path.basename(outfile)
            ])

        print(f"Finished kernel {i}: wavelength={wave}")

    return output_dir


def pad_kernel(kernel, target_shape):
    """Center-pad a 2D kernel with zeros to the requested shape."""
    kernel = np.asarray(kernel)
    if kernel.ndim != 2:
        raise ValueError("Each kernel must be a 2D array.")
    if kernel.shape[0] > target_shape[0] or kernel.shape[1] > target_shape[1]:
        raise ValueError("target_shape must be at least as large as the kernel.")

    padded = np.zeros(target_shape, dtype=kernel.dtype)
    yoff = (target_shape[0] - kernel.shape[0]) // 2
    xoff = (target_shape[1] - kernel.shape[1]) // 2
    padded[yoff:yoff + kernel.shape[0], xoff:xoff + kernel.shape[1]] = kernel
    return padded


def stack_kernel_cube(
    kernel_dir,
    output_file,
    pixscale,
    wavelengths=None,
    overwrite=True,
):
    """
    Stack kernel_*.fits files into a single cube, padding kernels to a common
    shape. Files are sorted lexicographically, matching kernel_0000.fits naming.
    """
    kernel_files = sorted(glob.glob(os.path.join(kernel_dir, "kernel_*.fits")))
    if not kernel_files:
        raise RuntimeError(f"No kernel_*.fits files found in {kernel_dir}")

    kernels = [fits.getdata(path) for path in kernel_files]
    if any(kernel.ndim != 2 for kernel in kernels):
        raise ValueError("Every kernel file must contain a 2D image.")

    ny = max(k.shape[0] for k in kernels)
    nx = max(k.shape[1] for k in kernels)
    print(f"Padding all kernels to {ny} x {nx}")

    cube = np.empty((len(kernels), ny, nx), dtype=float)
    for i, kernel in enumerate(kernels):
        cube[i] = pad_kernel(kernel, (ny, nx))

    header = fits.Header()
    header["NKERNEL"] = len(kernels)
    header["PIXELSCL"] = float(pixscale)

    if wavelengths is not None:
        if len(wavelengths) != len(kernels):
            raise ValueError("wavelengths must have one value per kernel.")
        wave_values = (
            wavelengths.to_value(u.micron)
            if isinstance(wavelengths, u.Quantity)
            else wavelengths
        )
        header["WAVEUNIT"] = "um"
        for i, wave in enumerate(wave_values):
            header[f"WAVE{i:04d}"] = float(wave)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    fits.PrimaryHDU(cube, header).writeto(output_file, overwrite=overwrite)
    print(f"Wrote kernel cube: {output_file}")
    return cube


# =============================================================================
# Science-cube convolution
# =============================================================================

def do_jwst_convolution_3d_astropy(
        file_in,
        file_out,
        file_kernel,
        blank_zeros=True,
        output_grid=None,
        reproject_func="interp",
        pad_edges=False,
        pad_width=None,
):
    """
    Convolves input image with an input kernel, and writes to disk.

    Will also process errors and do reprojection, if specified.
    Error processing is skipped if no ERR extension exists.
    """

    if reproject_func == "interp":
        r_func = reproject_interp
    elif reproject_func == "exact":
        r_func = reproject_exact
    elif reproject_func == "adaptive":
        r_func = reproject_adaptive
    else:
        raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

    with fits.open(file_kernel) as kernel_hdu:
        kernel_pix_scale = get_pixscale(kernel_hdu[0])
        kernel_data = kernel_hdu[0].data
        nwave = kernel_data.shape[0]
        kernel_hdu_length = kernel_data.shape[1]
        original_central_pixel = (kernel_hdu_length - 1) / 2
        original_grid = (
            np.arange(kernel_hdu_length) - original_central_pixel
        ) * kernel_pix_scale

    with fits.open(file_in) as image_hdu:

        # Check whether an ERR extension exists
        has_err = "ERR" in image_hdu

        if blank_zeros:
            if has_err:
                image_hdu["ERR"].data[
                    image_hdu["SCI"].data == 0
                ] = np.nan

            image_hdu["SCI"].data[
                image_hdu["SCI"].data == 0
            ] = np.nan

        image_pix_scale = get_pixscale(image_hdu["SCI"])

        interpolate_kernel_size = int(
            np.floor(kernel_hdu_length * kernel_pix_scale / image_pix_scale) - 2
        )

        if interpolate_kernel_size % 2 == 0:
            interpolate_kernel_size -= 1

        new_central_pixel = (interpolate_kernel_size - 1) / 2
        new_grid = (
            np.arange(interpolate_kernel_size) - new_central_pixel
        ) * image_pix_scale
        x_coords_new, y_coords_new = np.meshgrid(new_grid, new_grid)

        kernel_interp_cube = np.empty(
            (nwave, interpolate_kernel_size, interpolate_kernel_size),
            dtype=float,
        )

        for i in range(nwave):

            grid_interpolated = RegularGridInterpolator(
                (original_grid, original_grid),
                kernel_data[i],
                bounds_error=False,
                fill_value=0.0,
            )

            kernel_interp = grid_interpolated(
                (x_coords_new.flatten(), y_coords_new.flatten())
            )

            kernel_interp = kernel_interp.reshape(x_coords_new.shape)
            kernel_interp /= np.nansum(kernel_interp)

            kernel_interp_cube[i] = kernel_interp

        sci_cube = image_hdu["SCI"].data

        if has_err:
            err_cube = image_hdu["ERR"].data

        # --------------------------------------------------
        # Optional zero-padding around spatial edges
        # --------------------------------------------------
        if pad_edges:

            if pad_width is None:
                pad_width = interpolate_kernel_size // 2

            print(f"Padding spatial edges by {pad_width} pixels")

            sci_cube_padded = np.pad(
                sci_cube,
                ((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
                mode="constant",
                constant_values=0.0,
            )

            if has_err:
                err_cube_padded = np.pad(
                    err_cube,
                    ((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
                    mode="constant",
                    constant_values=0.0,
                )

        else:

            sci_cube_padded = sci_cube

            if has_err:
                err_cube_padded = err_cube

        conv_im_padded = np.empty_like(sci_cube_padded)

        if has_err:
            conv_err_padded = np.empty_like(err_cube_padded)

        # --------------------------------------------------
        # Convolve SCI
        # --------------------------------------------------
        for i in range(nwave):

            conv_im_padded[i] = convolve_fft(
                sci_cube_padded[i],
                kernel_interp_cube[i],
                allow_huge=True,
                preserve_nan=True,
                fill_value=np.nan,
            )

        if pad_edges:

            conv_im = conv_im_padded[
                :,
                pad_width:-pad_width,
                pad_width:-pad_width,
            ]

        else:

            conv_im = conv_im_padded

        # --------------------------------------------------
        # Convolve ERR, if available
        # --------------------------------------------------
        if has_err:

            for i in range(nwave):

                conv_err_padded[i] = np.sqrt(
                    convolve_fft(
                        err_cube_padded[i]**2,
                        kernel_interp_cube[i]**2,
                        preserve_nan=True,
                        allow_huge=True,
                        normalize_kernel=False,
                    )
                )

            if pad_edges:

                conv_err = conv_err_padded[
                    :,
                    pad_width:-pad_width,
                    pad_width:-pad_width,
                ]

            else:

                conv_err = conv_err_padded

        image_hdu["SCI"].data = conv_im

        if has_err:
            image_hdu["ERR"].data = conv_err

        if output_grid is None:
            image_hdu.writeto(file_out, overwrite=True)

        else:
            # Reprojection to target spatial WCS/grid
            target_wcs, target_shape = output_grid

            input_wcs = WCS(image_hdu["SCI"].header).celestial

            repr_data = np.empty(
                (conv_im.shape[0], target_shape[0], target_shape[1]),
                dtype=conv_im.dtype,
            )

            if has_err:
                repr_err = np.empty_like(repr_data)

            # Reproject each wavelength slice independently
            for i in range(conv_im.shape[0]):

                repr_data[i], fp = r_func(
                    (conv_im[i], input_wcs),
                    output_projection=target_wcs,
                    shape_out=target_shape,
                )

                fp = fp.astype(bool)
                repr_data[i][~fp] = np.nan

                if has_err:
                    repr_err[i] = r_func(
                        (conv_err[i], input_wcs),
                        output_projection=target_wcs,
                        shape_out=target_shape,
                        return_footprint=False,
                    )

                    repr_err[i][~fp] = np.nan

            # --------------------------------------------------
            # Construct output 3D WCS
            # --------------------------------------------------

            input_wcs_3d = WCS(image_hdu["SCI"].header)
            target_wcs_2d = target_wcs

            output_wcs = WCS(naxis=3)

            # Spatial WCS
            target_header = target_wcs_2d.to_header()

            output_wcs.wcs.ctype[0] = target_wcs_2d.wcs.ctype[0]
            output_wcs.wcs.ctype[1] = target_wcs_2d.wcs.ctype[1]

            output_wcs.wcs.crval[0] = target_wcs_2d.wcs.crval[0]
            output_wcs.wcs.crval[1] = target_wcs_2d.wcs.crval[1]

            output_wcs.wcs.crpix[0] = target_wcs_2d.wcs.crpix[0]
            output_wcs.wcs.crpix[1] = target_wcs_2d.wcs.crpix[1]

            output_wcs.wcs.cdelt[0] = target_wcs_2d.wcs.cdelt[0]
            output_wcs.wcs.cdelt[1] = target_wcs_2d.wcs.cdelt[1]

            output_wcs.wcs.cunit[0] = target_wcs_2d.wcs.cunit[0]
            output_wcs.wcs.cunit[1] = target_wcs_2d.wcs.cunit[1]

            output_wcs.wcs.pc[0, 0] = target_wcs_2d.wcs.pc[0, 0]
            output_wcs.wcs.pc[0, 1] = target_wcs_2d.wcs.pc[0, 1]
            output_wcs.wcs.pc[1, 0] = target_wcs_2d.wcs.pc[1, 0]
            output_wcs.wcs.pc[1, 1] = target_wcs_2d.wcs.pc[1, 1]

            # Spectral axis
            input_header = image_hdu["SCI"].header

            output_wcs.wcs.ctype[2] = "WAVE"
            output_wcs.wcs.crpix[2] = input_header["CRPIX3"]
            output_wcs.wcs.crval[2] = input_header["CRVAL3"]
            output_wcs.wcs.cdelt[2] = input_header["CDELT3"]
            output_wcs.wcs.cunit[2] = "um"

            output_wcs.wcs.pc[2, :] = [0.0, 0.0, 1.0]
            output_wcs.wcs.pc[:, 2] = [0.0, 0.0, 1.0]

            # Create output header
            sci_header = output_wcs.to_header()

            sci_header["NAXIS"] = 3
            sci_header["NAXIS1"] = target_shape[1]
            sci_header["NAXIS2"] = target_shape[0]
            sci_header["NAXIS3"] = conv_im.shape[0]

            # Create output file
            hdulist_out = fits.HDUList([
                fits.PrimaryHDU(header=image_hdu[0].header)
            ])

            hdulist_out.append(
                fits.ImageHDU(
                    data=repr_data,
                    header=sci_header,
                    name="SCI",
                )
            )

            if has_err:
                err_header = sci_header.copy()

                hdulist_out.append(
                    fits.ImageHDU(
                        data=repr_err,
                        header=err_header,
                        name="ERR",
                    )
                )

            hdulist_out.writeto(file_out, overwrite=True)



# =============================================================================
# Aniano-style convolution kernel class and PSF measurements
# =============================================================================
#
# The class below uses helper functions from the original kernel implementation:
# interp_nans, resample, centroid, circularize, resize, high_pass_filter,
# low_pass_filter, and trim_kernel_energy. Add those helper implementations to
# this file (or import them from their original module) for kernel generation
# to run. Their implementations were not included in the supplied code, so they
# are deliberately not replaced with approximations here.


class MakeConvolutionKernel:
    """Generate convolution kernels following the Aniano et al. (2011) method."""

    def __init__(
        self,
        source_psf=None,
        source_fwhm=None,
        source_name=None,
        source_pixscale=1,
        target_psf=None,
        target_fwhm=None,
        target_name=None,
        target_pixscale=1,
        common_pixscale=None,
        grid_size_arcsec=None,
        verbose=False,
    ):
        if source_psf is None:
            raise ValueError("source_psf must be provided")

        if isinstance(source_psf, (fits.PrimaryHDU, fits.ImageHDU)):
            self.source_psf = copy.deepcopy(source_psf.data)
            self.source_pixscale = get_pixscale(source_psf)
        elif isinstance(source_psf, np.ndarray):
            self.source_psf = copy.deepcopy(source_psf)
            self.source_pixscale = float(source_pixscale)
        else:
            raise TypeError("source_psf must be an ndarray or FITS image HDU")

        self.target_psf = None
        self.target_pixscale = float(target_pixscale)
        self.target_fwhm = None

        if target_psf is not None:
            if isinstance(target_psf, (fits.PrimaryHDU, fits.ImageHDU)):
                self.target_psf = copy.deepcopy(target_psf.data)
                self.target_pixscale = get_pixscale(target_psf)
            elif isinstance(target_psf, np.ndarray):
                self.target_psf = copy.deepcopy(target_psf)
            else:
                raise TypeError("target_psf must be an ndarray or FITS image HDU")

            if target_fwhm is None:
                target_fwhm = fit_2d_gaussian(
                    data=self.target_psf.copy(), pixscale=self.target_pixscale
                )
            self.target_fwhm = copy.deepcopy(target_fwhm)

        if source_fwhm is None:
            source_fwhm = fit_2d_gaussian(
                data=self.source_psf.copy(), pixscale=self.source_pixscale
            )

        if self.target_fwhm is not None and source_fwhm >= self.target_fwhm:
            raise ValueError(
                "Cannot create a kernel from a source PSF whose FWHM is "
                "equal to or broader than the target PSF "
                f"(source={source_fwhm}, target={self.target_fwhm} arcsec)."
            )

        self.source_fwhm = copy.deepcopy(source_fwhm)
        self.source_name = source_name if source_name is not None else "source"
        self.target_name = target_name if target_name is not None else "target"
        self.common_pixscale = copy.deepcopy(common_pixscale)
        self.grid_size_arcsec = (
            copy.deepcopy(grid_size_arcsec) if grid_size_arcsec is not None else None
        )

        self.source_fourier = None
        self.target_fourier = None
        self.kernel_fourier = None
        self.kernel = None
        self._source_processed = False
        self._target_processed = False
        self.verbose = verbose

    def _process_psf(
        self, psf_data, pixscale, common_pixscale, grid_size_arcsec,
        do_circularize=True,
    ):
        """Interpolate, resample, centroid, optionally circularize, resize, normalize."""
        psf_data = interp_nans(psf_data)

        if not np.isclose(pixscale, common_pixscale):
            if self.verbose:
                print("Resampling")
            psf_data = resample(psf_data, pixscale, common_pixscale)
        else:
            if self.verbose:
                print("Skipping resample (pixel scales match)")
            if psf_data.shape[0] % 2 == 0:
                psf_data = psf_data[:-1, :]
            if psf_data.shape[1] % 2 == 0:
                psf_data = psf_data[:, :-1]

        if self.verbose:
            print("Centroiding")
        psf_data = centroid(psf_data)

        if do_circularize:
            if self.verbose:
                print("Circularizing")
            psf_data = circularize(psf_data)

        if self.verbose:
            print("Resizing")
        psf_data = resize(
            psf_data, common_pixscale, grid_size_arcsec=grid_size_arcsec
        )

        total = np.nansum(psf_data)
        if not np.isfinite(total) or np.isclose(total, 0):
            raise ValueError("Processed PSF has a zero or invalid sum.")
        psf_data = psf_data / total

        if self.verbose:
            print(f"  final shape: {psf_data.shape}")
        return psf_data

    def process_source_psf(self, do_circularize=True, do_fourier_filter=True):
        """Process only the source PSF, optionally applying Fourier filtering."""
        if self._source_processed:
            raise RuntimeError(
                "Source PSF has already been processed. Create a new instance to reprocess."
            )

        if self.common_pixscale is None:
            self.common_pixscale = self.source_pixscale
        if self.grid_size_arcsec is None:
            self.grid_size_arcsec = (
                np.array(self.source_psf.shape, dtype=float) * self.source_pixscale
            )

        if self.verbose:
            print(f"Processing source PSF ({self.source_name})")

        self.source_psf = self._process_psf(
            self.source_psf, self.source_pixscale, self.common_pixscale,
            self.grid_size_arcsec, do_circularize=do_circularize,
        )

        if do_fourier_filter:
            if self.verbose:
                print("Fourier-domain filtering")
            source_fourier = np.real(
                np.fft.fft2(np.fft.ifftshift(self.source_psf))
            )
            source_fourier = np.fft.fftshift(source_fourier)
            source_fourier = circularize(source_fourier)
            source_fourier = high_pass_filter(
                source_fourier, self.source_fwhm, self.common_pixscale / 2
            )
            self.source_psf = np.fft.fftshift(
                np.real(np.fft.ifft2(np.fft.ifftshift(source_fourier)))
            )
            self.source_psf[np.abs(self.source_psf) <= np.finfo(float).eps] = 0
            total = np.nansum(self.source_psf)
            if not np.isfinite(total) or np.isclose(total, 0):
                raise ValueError("Filtered source PSF has a zero or invalid sum.")
            self.source_psf /= total

        self._source_processed = True

    def make_convolution_kernel(self):
        """Process both PSFs and create the convolution kernel."""
        if self.target_psf is None:
            raise ValueError("target_psf is required to create a convolution kernel.")
        if self._source_processed or self._target_processed:
            raise RuntimeError(
                "PSFs have already been processed. Create a new instance to reprocess."
            )

        if self.common_pixscale is None:
            self.common_pixscale = self.source_pixscale
        if self.grid_size_arcsec is None:
            self.grid_size_arcsec = (
                np.array(self.source_psf.shape, dtype=float) * self.source_pixscale
            )

        if self.verbose:
            print("Processing source and target PSFs")

        self.source_psf = self._process_psf(
            self.source_psf, self.source_pixscale, self.common_pixscale,
            self.grid_size_arcsec,
        )
        self.target_psf = self._process_psf(
            self.target_psf, self.target_pixscale, self.common_pixscale,
            self.grid_size_arcsec,
        )
        self._source_processed = True
        self._target_processed = True
        self._fourier_kernel_pipeline()

    def _fourier_kernel_pipeline(self):
        """Construct the Fourier-domain kernel from spatially processed PSFs."""
        if self.target_psf is None:
            raise ValueError("target_psf is required to build a kernel.")
        if self.target_fwhm is None:
            raise ValueError("target_fwhm is required to build a kernel.")

        self.source_fourier = np.fft.fftshift(
            np.real(np.fft.fft2(np.fft.ifftshift(self.source_psf)))
        )
        self.target_fourier = np.fft.fftshift(
            np.real(np.fft.fft2(np.fft.ifftshift(self.target_psf)))
        )

        self.source_fourier = circularize(self.source_fourier)
        self.target_fourier = circularize(self.target_fourier)

        source_high = high_pass_filter(
            self.source_fourier, self.source_fwhm, self.common_pixscale / 2
        )
        target_high = high_pass_filter(
            self.target_fourier, self.target_fwhm, self.common_pixscale / 2
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source_inverse = source_high ** -1
            source_inverse[~np.isfinite(source_inverse)] = 0

        source_low = low_pass_filter(self.source_fourier, self.common_pixscale / 2)
        self.kernel_fourier = np.fft.ifftshift(
            target_high * (source_low * source_inverse)
        )
        self.kernel = np.fft.fftshift(
            np.real(np.fft.ifft2(self.kernel_fourier))
        )
        self.kernel[np.abs(self.kernel) <= np.finfo(float).eps] = 0
        self.kernel = centroid(self.kernel)
        self.kernel = trim_kernel_energy(self.kernel)

        peak = np.nanmax(self.kernel)
        if not np.isfinite(peak) or np.isclose(peak, 0):
            raise ValueError("Generated kernel has a zero or invalid peak.")
        self.kernel = circularize(self.kernel)
        self.kernel /= peak

    def make_convolution_kernel_from_processed(self):
        """Build a kernel when both PSFs have already been processed."""
        if self.target_psf is None:
            raise ValueError("target_psf is required.")
        if self._source_processed or self._target_processed:
            raise RuntimeError("PSFs are already marked as processed.")
        if self.source_psf.shape != self.target_psf.shape:
            raise ValueError("Processed source and target PSFs must have equal shapes.")
        if not np.isclose(self.source_pixscale, self.target_pixscale):
            raise ValueError("Processed source and target pixel scales must match.")

        if self.common_pixscale is None:
            self.common_pixscale = self.source_pixscale
        elif not np.isclose(self.common_pixscale, self.source_pixscale):
            raise ValueError("common_pixscale must match the processed PSF pixel scale.")

        self._source_processed = True
        self._target_processed = True
        self._fourier_kernel_pipeline()

    def save_processed_psf(
        self, outdir, which="source", filename_suffix="aniano_circ_filt"
    ):
        """Save processed source PSF, target PSF, or both to FITS files."""
        if which not in ("source", "target", "both"):
            raise ValueError("which must be 'source', 'target', or 'both'.")

        os.makedirs(outdir, exist_ok=True)
        saved = []
        psfs_to_save = []
        if which in ("source", "both"):
            psfs_to_save.append((self.source_psf, self.source_name))
        if which in ("target", "both"):
            psfs_to_save.append((self.target_psf, self.target_name))

        for psf_data, name in psfs_to_save:
            if psf_data is None:
                raise ValueError(f"No PSF data available for {name!r}.")
            hdu = fits.PrimaryHDU(data=np.asarray(psf_data, dtype=np.float32))
            hdu.header["PSFNAME"] = (str(name), "PSF identifier")
            hdu.header["PIXELSCL"] = (self.common_pixscale, "arcsec/pixel")
            hdu.header["CRPIX1"] = (psf_data.shape[1] + 1) / 2
            hdu.header["CRPIX2"] = (psf_data.shape[0] + 1) / 2
            hdu.header["CRVAL1"] = 0.0
            hdu.header["CRVAL2"] = 0.0
            hdu.header["CDELT1"] = -self.common_pixscale / 3600
            hdu.header["CDELT2"] = self.common_pixscale / 3600
            filename = os.path.join(
                outdir, f"{name}_{filename_suffix}.fits"
            )
            hdu.writeto(filename, overwrite=True)
            saved.append(filename)
        return saved

    def write_out_kernel(
        self, outdir=None, add_keys=None, naming_convention="PHANGS",
        print_name=False,
    ):
        """Write the current kernel to a FITS file."""
        if self.kernel is None:
            raise ValueError("No kernel exists yet; call make_convolution_kernel first.")
        if outdir is None:
            raise ValueError("outdir must be provided.")
        os.makedirs(outdir, exist_ok=True)

        if naming_convention == "PHANGS":
            target_name = self.target_name.replace(".", "p").lower()
            source_name = self.source_name.lower()
        else:
            target_name = self.target_name
            source_name = self.source_name

        filename = os.path.join(outdir, f"{source_name}_to_{target_name}.fits")
        hdu = fits.PrimaryHDU(data=np.asarray(self.kernel, dtype=np.float32))
        hdu.header["CRPIX1"] = (self.kernel.shape[1] + 1) / 2
        hdu.header["CRPIX2"] = (self.kernel.shape[0] + 1) / 2
        hdu.header["CRVAL1"] = 0.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CDELT1"] = -self.common_pixscale / 3600
        hdu.header["CDELT2"] = self.common_pixscale / 3600
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"

        if isinstance(add_keys, dict):
            for key, value in add_keys.items():
                hdu.header[key] = value

        hdu.writeto(filename, overwrite=True)
        if print_name:
            print(f"Saved: {filename}")
        return filename


def fit_2d_gaussian(data, pixscale=None):
    """Fit a 2D Gaussian to a PSF and return its FWHM in the requested units."""
    # Work on a copy: normalizing the PSF should not mutate the caller's array.
    data = np.array(data, dtype=float, copy=True)
    peak = np.nanmax(data)
    if not np.isfinite(peak) or np.isclose(peak, 0):
        raise ValueError("Cannot fit a Gaussian to an empty or zero-valued image.")
    data /= peak

    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2
    ji, ii = np.meshgrid(
        np.arange(data.shape[1]) - j_cen,
        np.arange(data.shape[0]) - i_cen,
    )
    if pixscale is not None:
        ji = ji * pixscale
        ii = ii * pixscale

    center_col = int(i_cen)
    column = data[:, center_col]
    column_sum = np.nansum(column)
    if not np.isfinite(column_sum) or np.isclose(column_sum, 0):
        raise ValueError("Cannot estimate Gaussian width from the central column.")
    guess_sigma = np.sqrt(
        np.nansum(ii[:, center_col] ** 2 * column) / column_sum
    )
    if not np.isfinite(guess_sigma) or guess_sigma <= 0:
        guess_sigma = (pixscale or 1.0)

    model = models.Gaussian2D(
        amplitude=1, x_mean=0, y_mean=0,
        x_stddev=guess_sigma, y_stddev=guess_sigma, theta=0,
    )
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(model, ji, ii, data)

    return 2.355 * (fit.x_stddev.value + fit.y_stddev.value) / 2


def evaluate_kernel(kk):
    """
    Evaluate a generated kernel.

    Returns
    -------
    D : float
        L1 difference between convolved source PSF and target PSF.
    Wminus : float
        Sum of the negative kernel volume, as defined in Aniano et al.
    """
    kernel_sum = np.sum(kk.kernel)
    if not np.isfinite(kernel_sum) or np.isclose(kernel_sum, 0):
        raise ValueError("Cannot evaluate a kernel with zero or invalid sum.")

    # Preserve the original convention: normalize the kernel in place.
    kk.kernel = kk.kernel / kernel_sum
    target_conv = convolve(kk.source_psf, kk.kernel)
    d_value = np.sum(np.abs(target_conv - kk.target_psf))
    wminus = 0.5 * np.sum(np.abs(kk.kernel) - kk.kernel)
    return d_value, wminus
