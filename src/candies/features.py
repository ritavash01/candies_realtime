import logging 
from pathlib import Path 
from numba import cuda 
from rich.progress import track 
from rich.logging import RichHandler 

from candies.interfaces import SIGPROCFilterbank
from candies.utilities import kdm, delay2dm, normalise
from fetch.utils import get_model
from fetch.data_sequence import DataGenerator

from candies.base import (
    Candidate,
    Dedispersed,
    DMTransform,
    CandiesError,
    CandidateList,
)

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
    cuda.atomic.add(dmt, (dmi, int(ti / downt)), acc)  # type; ignore


def featurize(
    candidates: Candidate | CandidateList
    # filterbank: str | Path, we need to replace this 
    gpuid: int = 0, 
    save: bool = True, 
    zoom: bool = True, 
    fudging: int = 512,
    verbose: bool = False,
    progressbar: bool = False, 
):

    if isinstance(candidates,Candidate):
        candidates = CandidateList(candidates = [candidates])
    
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

    with stream.auto_synchronize():
        with cuda.defer_cleanup():
            with Getrawdata() as raw:
                for candidate in track(
                    candidates, 
                    disable = (not progressbar),
                    description = f"Featurizing from raw data",
                ):

                _, _, data = raw.chop(candidate)
                nf, nt = data.shape # why this
                log.debug(f"Read in data with {nf} channels and {nt} samples.")

                ndms = 256 
                dmlow, dmhigh = 0.0, 2 * candidate.dm 

                if zoom:
                        log.debug("Zoom-in feature active. Calculating DM range.")
                        ddm = delay2dm(
                            raw.fl, raw.fh, fudging * candidate.wbin * raw.dt
                        )
                        if ddm < candidate.dm:
                            dmlow, dmhigh = candidate.dm - ddm, candidate.dm + ddm
                    ddm = (dmhigh - dmlow) / (ndms - 1)
                    log.debug(f"Using DM range: {dmlow} to {dmhigh} pc cm^-3.")
                
                    downf = int (raw.nf / 256)
                    downt = 1 if candidate.wbin < 3 else int(candidate.wbin / 2)
                    log.debug(
                        f"Downsampling by {downf} in frequency and {downt} in time."
                    )

                    nfdown = int(raw.nf /downf)
                    ntdown = int(nt/downt)
                    gpudata = cuda.to_device(data, stream=stream)
                    gpudd = cuda.device_array(
                        (nfdown, ntdowndef featurize(
    candidates: Candidate | CandidateList
    # filterbank: str | Path, we need to replace this 
    gpuid: int = 0, 
    save: bool = True, 
    zoom: bool = True, 
    fudging: int = 512,
    verbose: bool = False,
    progressbar: bool = False,
    batch_size: int = 5  
):

    if isinstance(candidates,Candidate):
        candidates = CandidateList(candidates = [candidates])
    
    """

    Grouping candidates into batches 
    """
    candidate_batches = [candidates[i:i+ batch_size] for i in range(0, len(candidates), batch_size)]

    logging.basicConfig(
        datefmt="[%X]",
        format="%(message)s",
        level=("DEBUG" if verbose else "INFO"),
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    log = logging.getLogger("candies")
    classified_list = [] # Final list 
    cuda.select_device(gpuid)
    stream = cuda.stream()
    log.debug(f"Selected GPU {gpuid}.")
    file_path_list = []
    with stream.auto_synchronize():
        with cuda.defer_cleanup():
            with Getrawdata() as raw:  # we are keeping track of batches now 
                for batch in track(  
                    candidate_batches, 
                    disable = (not progressbar),
                    description = f"Featurizing raw data for batch {batch}",
                ):
                # Processing each batch of candidates                                
                    for candidate in batch:

                        _, _, data = raw.chop(candidate)
                        nf, nt = data.shape # why this
                        log.debug(f"Read in data with {nf} channels and {nt} samples.")

                        ndms = 256 
                        dmlow, dmhigh = 0.0, 2 * candidate.dm 

                        if zoom:
                                log.debug("Zoom-in feature active. Calculating DM range.")
                                ddm = delay2dm(
                                    raw.fl, raw.fh, fudging * candidate.wbin * raw.dt
                                )
                                if ddm < candidate.dm:
                                    dmlow, dmhigh = candidate.dm - ddm, candidate.dm + ddm
                            ddm = (dmhigh - dmlow) / (ndms - 1)
                            log.debug(f"Using DM range: {dmlow} to {dmhigh} pc cm^-3.")
                        
                            downf = int (raw.nf ), order="C", stream=stream
                            )
                            gpudmt = cuda.device_array((ndms, ntdown), order="C", stream=stream)

                            dedisperse[  # type: ignore
                                (int(raw.nf / 32), int(nt / 32)),
                                (32, 32),
                                stream,
                            ](
                                gpudd,
                                gpudata,
                                raw.nf,
                                nt,
                                raw.df,
                                raw.dt,
                                raw.fh,
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
                                raw.df,
                                raw.dt,
                                raw.fh,
                                ddm,
                                dmlow,
                                downt,
                            )

                            ntmid = int(ntdown / 2)

                            dedispersed = gpudd.copy_to_host(stream=stream)
                            dedispersed = dedispersed[:, ntmid - 128 : ntmid + 128]
                            dedispersed = normalise(dedispersed)
                            candidate.dedispersed = Dedispersed(
                                fl=raw.fl,
                                fh=raw.fh,
                                nt=256,
                                nf=256,
                                dm=candidate.dm,
                                data=dedispersed,
                                dt=raw.dt * downt,
                                df=(raw.fh - raw.fl) / 256,
                            )

                            if save:
                                candidate.extras = {**raw.getdataheader()}
                                fname = "".join([str(candidate), ".h5"])
                                candidate.save(fname)
"""
                            # This will contain the file paths of the batch                 
                                file_paths.append(fname)
                    
                    Calling FETCH inside candies  
                    
                    model = get_model("a")
                    h5files = file_paths

                    generator = DataGenerator(
                        noise=False,
                        batch_size=8,
                        shuffle=False,
                        list_IDs=h5files,
                        labels=[0] * len(h5files),
                    )
                    probabilities = model.predict_generator(
                        verbose=1,
                        workers=4,
                        generator=generator,
                        steps=len(generator),
                        use_multiprocessing=True,
                    )
                    classified_dataframe = pd.DataFrame(
                        {
                            "candidate": h5files,
                            "probability": probabilities[:, 1],
                            "labels": np.round(probabilities[:, 1] >= 0.5),
                        }
                    )
                    classified_list.append(classified_dataframe)
                final_list = pd.concat(classified_list, ignore_index = True)
                final_list.to_csv("classification.csv")
"""
    cuda.close()
