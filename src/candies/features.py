"""
The feature extraction code for Candies.
"""

import logging
import h5py as h5
import numpy as np
from numba import cuda
from pathlib import Path
from rich.progress import track
from rich.logging import RichHandler
from candies.interfaces import Filterbank
from candies.base import Dedispersed, DMTransform, CandidateList
from candies.utilities import kdm, delay2dm, dm2delay, normalise


@cuda.jit
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


@cuda.jit
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
    candidates: CandidateList,
    filterbank: str | Path,
    gpuid: int = 0,
    save: bool = True,
    zoom: bool = True,
    fudging: int = 512,
    verbose: bool = False,
    progressbar: bool = False,
):
    """
    Create the features for a list of candidates.

    The classifier uses two features for each candidate: the dedispersed
    dynamic spectrum, and the DM transform. These two features are created
    by this function for each candidate in a list, using JIT-compiled CUDA
    kernels created via Numba for each feature.

    We also improve these features, by 1. zooming into the DM-time plane by
    a factor decided by the arrival time, width, DM of each candidate, and/or
    2. subbanding the frequency-time plane in case of band-limited emission.
    Currently only the former has been implemented.

    Parameters
    ----------
    candidates: CandidateList
        A list of candidates to process.
    filterbank:
        The path of the filterbank file to process.
    gpuid: int, optional
        The ID of the GPU to be used. The default value is 0.
    save: bool, optional
        Flag to decide whether to save the candidates or not. Default is True.
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

    with Filterbank(filterbank) as fil:
        for candidate in track(
            candidates,
            disable=(not progressbar),
            description="Featurizing...",
        ):
            maxdelay = dm2delay(fil.fl, fil.fh, candidate.dm)

            binbeg = int((candidate.t0 - maxdelay) / fil.dt) - candidate.wbin
            binend = int((candidate.t0 + maxdelay) / fil.dt) + candidate.wbin

            nbegin = noffset = binbeg
            nread = ncount = binend - binbeg
            if (candidate.wbin > 2) and (nread // (candidate.wbin // 2) < 256):
                nread = 256 * candidate.wbin // 2
            elif nread < 256:
                nread = 256
            nbegin = noffset - (nread - ncount) // 2

            if (nbegin >= 0) and (nbegin + nread) <= fil.nt:
                data = fil.get(offset=nbegin, count=nread)
            elif nbegin < 0:
                if (nbegin + nread) <= fil.nt:
                    d = fil.get(offset=0, count=nread + nbegin)
                    dmedian = np.median(d, axis=0)
                    data = np.ones((fil.nf, nread), dtype=fil.dtype) * dmedian[None, :]
                    data[:, -nbegin:] = d
                else:
                    d = fil.get(offset=0, count=fil.nt)
                    dmedian = np.median(d, axis=0)
                    data = np.ones((fil.nf, nread), dtype=fil.dtype) * dmedian[None, :]
                    data[:, -nbegin : -nbegin + fil.nt] = d
            else:
                d = fil.get(offset=nbegin, count=fil.nt - nbegin)
                dmedian = np.median(d, axis=0)
                data = np.ones((fil.nf, nread), dtype=fil.dtype) * dmedian[None, :]
                data[:, fil.nt - nbegin] = d

            nf, nt = data.shape
            log.debug(f"Read in {nf} channels and {nt} samples.")

            ndms = 256
            dmlow, dmhigh = 0.0, 2 * candidate.dm
            if zoom:
                log.debug("Zoom-in feature acitve. Calculating DM range.")
                ddm = delay2dm(fil.fl, fil.fh, fudging * candidate.wbin * fil.dt)
                if ddm < candidate.dm:
                    dmlow, dmhigh = candidate.dm - ddm, candidate.dm + ddm
            ddm = (dmhigh - dmlow) / (ndms - 1)
            log.debug(f"Using DM range: {dmlow} to {dmhigh} pc cm^-3.")

            downf = int(fil.nf / 256)
            downt = 1 if candidate.wbin < 3 else int(candidate.wbin / 2)
            log.debug(f"Downsampling by {downf} in frequency and {downt} in time.")

            nfdown = int(fil.nf / downf)
            ntdown = int(nt / downt)

            gpudata = cuda.to_device(data, stream=stream)
            gpudd = cuda.device_array((nfdown, ntdown), order="C", stream=stream)
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
                fname = (
                    "_".join(
                        [
                            f"MJD{fil.header['tstart']:.7f}",
                            f"T{candidate.t0:.7f}",
                            f"DM{candidate.dm:.5f}",
                            f"SNR{candidate.snr:.5f}",
                        ]
                    )
                    + ".h5"
                )
                candidate.save(fname)
                with h5.File(fname, "a") as f:
                    group = f.create_group("extras")
                    for key, value in fil.header.items():
                        group.attrs[key] = value
    cuda.close()