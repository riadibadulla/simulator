import astropy.units as u

from . import logging
from .optical_elements import Diffuser, FreeSpace

logger = logging.get_standard_logger(__name__)


@u.quantity_input(z=u.m)
def retrieve_phase_gs(
        wf, target, z, max_iter=10, randomize_phase=True,
        return_intermediate=False
):
    """ Iteratively retrieves phase of the wavefront based on the intensity
    distribution in two known planes. Implementation of Gerchberg-Saxton
    phase retrieval algorithm [1]. Phase is retrieved iteratively by
    propagating between input and target planes (limited by `max_iter`)
    and updating the intensity distribution:

    .. image:: ../source/imgs/gs_docstring.png

    :param wf: input wavefront (with initial intensity distribution)
    :type wf: pyoptica.Wavefront
    :param target: desired target distribution (2D array)
    :type target: numpy.array
    :param z: distance between two planes
    :type z: astropy.Quantity of type length
    :param max_iter: number of maximal iterations (default 10)
    :type max_iter: int
    :param randomize_phase: should phase be randomized?  (default True)
    :type randomize_phase: bool
    :param return_intermediate: should intermediate results (after each
        iteration) be returned? (default false)
    :type return_intermediate: bool
    :return: wavefront with retrieved phase (at input plane), list with
        intermediate results (empty if `return_intermediate` is `False`.)
    :rtype: tuple(pyoptica.Wavefront, list(pyoptica.Wavefront))

    **Example**

    >>> import astropy.units as u
    >>>
    >>> import pyoptica as po
    >>> import pyoptica.holography as poh
    >>>
    >>> wavelength = 500 * u.nm
    >>> pixel_scale = 22 * u.um
    >>> npix = 512
    >>> wf = po.Wavefront(wavelength, pixel_scale, npix)

    Let's retrieve phase distribution (or a hologram) that will produce
    at target plane at `z` intenisty distribution given by a binary grating:

    >>> period = 1 * u.mm
    >>> duty = 0.5
    >>> grating = po.BinaryGrating(period, duty)
    >>> target = (wf * grating).intensity

    Now it's time for GS algorithm. We are not going to ask for intermediate
    results:

    >>> z = 10 * u.cm
    >>> holo, _ = poh.retrieve_phase_gs(wf, target, z)
    >>> po.plotting.plot_wavefront(holo, 'phase')
    >>> po.plotting.plot_wavefront(holo * po.FreeSpace(z))

    **References**

    [1] R. W. Gerchberg and W. O. Saxton (1972) -
    "A practical algorithm for the determination of the phase from image
    and diffraction plane pictures,” Optik 35, 237

    """
    source = wf.intensity[:, :]
    wf_work_copy = wf.copy()  # input wavefront should not be manipulated
    if randomize_phase:
        wf_work_copy *= Diffuser()
    intermediate_results = []
    forward_prop_fs = FreeSpace(z)
    backward_prop_fs = FreeSpace(-z)
    for i in range(max_iter):
        wf_work_copy *= forward_prop_fs
        wf_work_copy.intensity = target
        wf_work_copy *= backward_prop_fs
        wf_work_copy.intensity = source
        if return_intermediate:
            intermediate_results.append(wf_work_copy.copy())
        logger.info(f"Done iteration {i + 1} out of {max_iter}.")
    return wf_work_copy, intermediate_results
