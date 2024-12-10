from typing import Self
from dataclasses import dataclass
import numpy as np
from shazam import internals
from candies.utilities import dm2delay
from candies.base import CandiesError, Candidate
import shared_memory_reader
import shared_memory_header

@dataclass
class Getrawdata:
    
    fh: float = None
    df: float = None
    dt: float = None
    nf: int = None
    bw: float = None
    nbits: int = None
    nt: float = None
    dtype: np.dtype = np.float32  # Or set an appropriate data type

    def __post_init__(self):
        hdr_dict = self.getdataheader()

        try:
            self.fh = 750     # hdr_dict["Frequency_Ch_0_Hz"] 
            self.df = -0.048828  #hdr_dict["Channel_width_Hz"] / 1e6
            self.dt = 0.00131072  # hdr_dict["Sampling_time_uSec"]
            self.nf = hdr_dict["Channels"]
            self.bw = hdr_dict["Bandwidth_MHz"]
            self.nbits = hdr_dict["Num_bits_per_sample"]
            self.nt = 4 * 100 * 1e6 / self.nf  # Adjust memory block size if needed
        except KeyError:
            raise CandiesError("Dictionary reading was incorrect")

        if self.df < 0:
            self.df = abs(self.df)
            self.fl = self.fh - self.bw + (0.5 * self.df)
        else:
            self.fl = self.fh
            self.fh = self.fl + self.bw - (0.5 * self.df)

    def getdatabuffer(self, count, offset):
        data = shared_memory_reader.get_data(count, offset) # 3 is the beam number
        data = data[2, :]
        return data

    def getdataheader(self):
        hdr_dict = shared_memory_header.get_header("1234")
        return hdr_dict

    def chop(self, candidate: Candidate) -> np.ndarray:
        maxdelay = dm2delay(self.fl, self.fh, candidate.dm)
        binbeg = int((candidate.t0 - maxdelay) / self.dt) - candidate.wbin
        binend = int((candidate.t0 + maxdelay) / self.dt) + candidate.wbin

        nbegin = noffset = binbeg
        nread = ncount = binend - binbeg
        if (candidate.wbin > 2) and (nread // (candidate.wbin // 2) < 256):
            nread = 256 * candidate.wbin // 2
        elif nread < 256:
            nread = 256
        nbegin = noffset - (nread - ncount) // 2

        if (nbegin >= 0) and (nbegin + nread) <= self.nt:
            data = self.getdatabuffer(offset=nbegin, count=nread)
        elif nbegin < 0:
            if (nbegin + nread) <= self.nt:
                d = self.getdatabuffer(offset=0, count=nread + nbegin)
                dmedian = np.median(d, axis=1)
                data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
                data[:, -nbegin:] = d
            else:
                d = self.getdatabuffer(offset=0, count=self.nt)
                dmedian = np.median(d, axis=1)
                data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
                data[:, -nbegin: -nbegin + self.nt] = d
        else:
            d = self.getdatabuffer(offset=nbegin, count=self.nt - nbegin)
            dmedian = np.median(d, axis=1)
            data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
            data[:, :self.nt - nbegin] = d
        return data
