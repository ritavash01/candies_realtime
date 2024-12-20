"""
The feature extraction code for candies.
"""

import logging
from pathlib import Path
import os
from numba import cuda
from rich.progress import track
from rich.logging import RichHandler
import numpy as np
import pandas as pd
from candies.interfaces import Getrawdata
from candies.utilities import kdm, delay2dm, normalise

from candies.base import (
    Candidate,
    Dedispersed,
    DMTransform,
    CandiesError,
    CandidateList,
)
from fetch.utils import get_model
from fetch.data_sequence import DataGenerator

"""
Function for FETCH

"""
# def classify_h5_file(filepath: str, model):
#     """
#     Classify a single .h5 file using the model.

#     Parameters
#     ----------
#     filepath : str
#         Path to the .h5 file.
#     model : keras.Model
#         Loaded classification model.

#     Returns
#     -------
#     probabilities : np.ndarray
#         Classification probabilities for the positive class.
#     """
#     generator = DataGenerator(
#         noise=False,
#         batch_size=1,
#         shuffle=False,
#         list_IDs=[Path(filepath)],
#         labels=[0],  # Placeholder
#     )
#     probabilities = model.predict_generator(
#         verbose=0, generator=generator, steps=len(generator),
#             use_multiprocessing=True
#     )
#     return probabilities[0]
#       #Return the probabilities for the file

@cuda.jit(cache=True, fastmath=True)
def dedisperse(
    dyn,
    ft,
    nf: int,
    nt: int,
    df: float,
    dt: float,
    fh: float,
    dm: float,
    downf: int,
    downt: int,
):
    """
    The JIT-compiled CUDA kernel for dedispersing a dynamic spectrum.

    Parameters
    ----------
    dyn:
        The array in which to place the output dedispersed dynamic spectrum.
    ft:
        The dynamic spectrum to dedisperse.
    nf: int
        The number of frequency channels.
    nt: int
        The number of time samples.
    df: float
        The channel width (in MHz).
    dt: float
        The sampling time (in seconds).
    fh: float
        The highest frequency in the band.
    dm: float
        The DM at which to dedisperse (in pc cm^-3).
    downf: int,
        The downsampling factor along the frequency axis.
    downt: int,
        The downsampling factor along the time axis.
    """

    fi, ti = cuda.grid(2)  # type: ignore

    acc = 0.0
    if fi < nf and ti < nt:
        k1 = kdm * dm / dt
        k2 = k1 * fh**-2
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        acc += ft[fi, xti]
        cuda.atomic.add(dyn, (int(fi / downf), int(ti / downt)), acc)  # type: ignore


@cuda.jit(cache=True, fastmath=True)
def fastdmt(
    dmt,
    ft,
    nf: int,
    nt: int,
    df: float,
    dt: float,
    fh: float,
    ddm: float,
    dmlow: float,
    downt: int,
):
    """
    The JIT-compiled CUDA kernel for obtaining a DM transform.

    Parameters
    ----------
    dmt:
        The array in which to place the output DM transform.
    ft:
        The dynamic spectrum to dedisperse.
    nf: int
        The number of frequency channels.
    nt: int
        The number of time samples.
    df: float
        The channel width (in MHz).
    dt: float
        The sampling time (in seconds).
    fh: float
        The highest frequency in the band.
    ddm: float
        The DM step to use (in pc cm^-3)
    dmlow: float
        The lowest DM value (in pc cm^-3).
    downt: int,
        The downsampling factor along the time axis.
    """

    ti = int(cuda.blockIdx.x)  # type: ignore
    dmi = int(cuda.threadIdx.x)  # type: ignore

    acc = 0.0
    k1 = kdm * (dmlow + dmi * ddm) / dt
    k2 = k1 * fh**-2
    for fi in range(nf):
        f = fh - fi * df
        dbin = int(round(k1 * f**-2 - k2))
        xti = ti + dbin
        if xti >= nt:
            xti -= nt
        acc += ft[fi, xti]
    cuda.atomic.add(dmt, (dmi, int(ti / downt)), acc)  # type: ignore


def featurize(
    candidates: Candidate | CandidateList,
    filterbank: str | Path,
    /,
    gpuid: int = 1,  #change on the basis of system used 
    save: bool = True,
    zoom: bool = True,
    fudging: int = 512,
    verbose: bool = False,
    progressbar: bool = False,
)-> Path: 
    """
    Create the features for a list of candy-dates.

    The classifier uses two features for each candy-date: the dedispersed
    dynamic spectrum, and the DM transform. These two features are created
    by this function for each candy-date in a list, using JIT-compiled CUDA
    kernels created via Numba for each feature.

    We also improve these features, by 1. zooming into the DM-time plane by
    a factor decided by the arrival time, width, DM of each candy-date, and/or
    2. subbanding the frequency-time plane in case of band-limited emission.
    Currently only the former has been implemented.

    Parameters
    ----------
    candidates: Candidate or CandidateList
        A candy-date, or a list of candy-dates, to process.
    filterbank: str | Path
        The path of the filterbank file to process.
    gpuid: int, optional
        The ID of the GPU to be used. The default value is 0.
    save: bool, optional
        Flag to decide whether to save the candy-date(s) or not. Default is True.
    zoom: bool, optional
        Flag to switch on zooming into the DM-time plane. Default is True.
    fudging: int, optional
        A fudge factor employed when zooming into the DM-time plane. The greater
        this factor is, the more we zoom out in the DM-time plane. The default
        value is currently set to 512.
    verbose: bool, optional
        Activate verbose printing. False by default.
    progressbar: bool, optional
        Show the progress bar. True by default.
    """
    if isinstance(candidates, Candidate):
        candidates = CandidateList(candidates=[candidates])

    logging.basicConfig(
        datefmt="[%X]",
        format="%(message)s",
        level=("DEBUG" if verbose else "INFO"),
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    log = logging.getLogger("candies")

    cuda.select_device(gpuid)
    stream = cuda.stream()
    log.debug(f"Selected GPU {gpuid}.")
    generated_files = [] 
    

    with stream.auto_synchronize():
        with cuda.defer_cleanup():
            with Getrawdata() as fil:
                for candidate in track(
                    candidates,
                    disable=(not progressbar),
                    description=f"Featurizing from {filterbank}...",
                ):
                    _, _, data = fil.chop(candidate)
                    nf, nt = data.shape
                    log.debug(f"Read in data with {nf} channels and {nt} samples.")

                    ndms = 256
                    dmlow, dmhigh = 0.0, 2 * candidate.dm
                    if zoom:
                        log.debug("Zoom-in feature active. Calculating DM range.")
                        ddm = delay2dm(
                            fil.fl, fil.fh, fudging * candidate.wbin * fil.dt
                        )
                        if ddm < candidate.dm:
                            dmlow, dmhigh = candidate.dm - ddm, candidate.dm + ddm
                    ddm = (dmhigh - dmlow) / (ndms - 1)
                    log.debug(f"Using DM range: {dmlow} to {dmhigh} pc cm^-3.")

                    downf = int(fil.nf / 256)
                    downt = 1 if candidate.wbin < 3 else int(candidate.wbin / 2)
                    log.debug(
                        f"Downsampling by {downf} in frequency and {downt} in time."
                    )

                    nfdown = int(fil.nf / downf)
                    ntdown = int(nt / downt)

                    gpudata = cuda.to_device(data, stream=stream)
                    gpudd = cuda.device_array(
                        (nfdown, ntdown), order="C", stream=stream
                    )
                    gpudmt = cuda.device_array((ndms, ntdown), order="C", stream=stream)

                    dedisperse[  # type: ignore
                        (int(fil.nf / 32), int(nt / 32)),
                        (32, 32),
                        stream,
                    ](
                        gpudd,
                        gpudata,
                        fil.nf,
                        nt,
                        fil.df,
                        fil.dt,
                        fil.fh,
                        candidate.dm,
                        downf,
                        downt,
                    )

                    fastdmt[  # type: ignore
                        nt,
                        ndms,
                        stream,
                    ](
                        gpudmt,
                        gpudata,
                        nf,
                        nt,
                        fil.df,
                        fil.dt,
                        fil.fh,
                        ddm,
                        dmlow,
                        downt,
                    )

                    ntmid = int(ntdown / 2)

                    dedispersed = gpudd.copy_to_host(stream=stream)
                    dedispersed = dedispersed[:, ntmid - 128 : ntmid + 128]
                    dedispersed = normalise(dedispersed)
                    candidate.dedispersed = Dedispersed(
                        fl=fil.fl,
                        fh=fil.fh,
                        nt=256,
                        nf=256,
                        dm=candidate.dm,
                        data=dedispersed,
                        dt=fil.dt * downt,
                        df=(fil.fh - fil.fl) / 256,
                    )

                    dmtransform = gpudmt.copy_to_host(stream=stream)
                    dmtransform = dmtransform[:, ntmid - 128 : ntmid + 128]
                    dmtransform = normalise(dmtransform)
                    candidate.dmtransform = DMTransform(
                        nt=256,
                        ddm=ddm,
                        ndms=ndms,
                        dmlow=dmlow,
                        dmhigh=dmhigh,
                        dm=candidate.dm,
                        dt=fil.dt * downt,
                        data=dmtransform,
                    )

                    if save:
                        candidate.extras =  {**fil.getdataheader()}
                        fname = "".join([str(candidate), ".h5"])
                        candidate.save(fname)
                        generated_files.append(fname)

    cuda.close()              
    return generated_files          
    
   

def classify(h5file: Path) -> dict:
    """
    Classify the single .h5 file and return the result as a dictionary.
    """
    model = get_model("a")

    generator = DataGenerator(
        noise=False,
        batch_size=1,
        shuffle=False,
        list_IDs=[h5file],
        labels=[0],
    )

    probabilities = model.predict_generator(
        verbose=1,
        workers=4,
        generator=generator,
        steps=1,
        use_multiprocessing=False,
    )

    probability = probabilities[0, 1]
    label = int(probability >= 0.5)

    if probability == 1.0:
        return {
            "candidate": str(h5file),
            "probability": probability,
            "label": label,
        }
    else:
        return {}
