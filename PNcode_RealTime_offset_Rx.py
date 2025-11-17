import time
import uhd
import numpy as np
import argparse
import math, datetime
from collections import deque
from scipy.signal import fftconvolve
import scipy.io
try:
    from pylfsr import LFSR
    _HAS_PYLFSR = True
except Exception:
    _HAS_PYLFSR = False

class RealTime_PN_Detector:
    def __init__(self,
                 device_addr="addr=192.168.40.2",
                 center_freq=3.455e9,
                 sample_rate=2_500_000.0,
                 gain=30,
                 # RX chain options
                 use_lo_offset=True,
                 lo_adjust=1.5e6,
                 if_freq=0.0,            # Hz; software mix-down if TX used IF
                 # PN/BPSK params (must match TX)
                 N_stages=10,
                 taps="10,7",
                 state="0,0,0,0,0,0,0,0,0,1",
                 samples_per_chip=4,
                 alpha=0.25,
                 Lp=6,
                 numPN=20,             # UPDATED: Match TX (was 20, now 10)
                 pad_zeros=1024,
                 preamble_scale=0.3,     # TX used 0.3*preamble
                 master_clock=200e6,
                 rx_antenna="TX/RX"):
        # Initialize RX parameters
        self.device_addr   = device_addr
        self.center_freq   = center_freq
        self.sample_rate   = float(sample_rate)
        self.gain          = gain
        self.use_lo_offset = use_lo_offset
        self.lo_adjust     = float(lo_adjust)
        self.if_freq       = float(if_freq)
        self.master_clock  = master_clock
        self.rx_antenna    = rx_antenna

        # PN/BPSK config
        self.N_stages = int(N_stages)
        self.taps     = [int(t) for t in (taps.split(",") if isinstance(taps, str) else taps)]
        self.state    = [int(s) for s in (state.split(",") if isinstance(state, str) else state)]
        self.O        = int(samples_per_chip)
        self.alpha    = float(alpha)
        self.Lp       = int(Lp)
        self.numPN    = int(numPN)
        self.pad_zeros = int(pad_zeros)
        self.preamble_scale = float(preamble_scale)

        # Derived lengths (CRITICAL UPDATE HERE)
        self.code_length = (2 ** self.N_stages) - 1
        self.P = 2 * self.O * self.Lp                             # SRRC tail (filter len 2*O*Lp+1)
        self.payload_len = self.O * self.code_length * self.numPN + self.P # Reflects numPN=10

        # UHD handles
        self.usrp = None
        self.rx_streamer = None

        # Buffers
        # preamble is loaded in generate_ltf()
        self.ltf = self.generate_ltf()                            # complex64
        # CRITICAL UPDATE: Total packet length is much larger now
        self.packet_total = len(self.ltf) + self.pad_zeros + self.payload_len
        self.big_buffer = deque(maxlen=int(2 * self.packet_total)) # Maxlen is 2x the new total length

        # Stats
        self.noise_power_est = None

    # --------------------- Signal helpers ---------------------
    def _mix_down_if(self, iq: np.ndarray, fs: float, f_if: float) -> np.ndarray:
        if abs(f_if) < 1e-3:
            return iq
        n = np.arange(iq.size, dtype=np.float64)
        osc = np.exp(-1j * 2 * np.pi * f_if * n / fs)  # negative for downmix
        return (iq * osc).astype(np.complex64)

    def srrc(self) -> np.ndarray:
        N = self.O
        a = self.alpha
        Lp = self.Lp
        n = np.arange(-N*Lp, N*Lp+1, dtype=float) + 1e-9
        coeff = 1.0 / math.sqrt(N)
        ccoef = 4 * a * n / N
        num = np.sin(np.pi * n * (1 - a) / N) + ccoef * np.cos(np.pi * n * (1 + a) / N)
        den = (np.pi * n / N) * (1 - ccoef**2)
        h = coeff * (num / den)
        return h

    def oversample_zeros(self, x: np.ndarray, OS: int) -> np.ndarray:
        out = np.zeros(x.size * OS, dtype=float)
        out[OS-1::OS] = x
        return out

    def _pn_bits(self) -> np.ndarray:
        if _HAS_PYLFSR:
            L = LFSR(initstate=self.state, fpoly=self.taps)
            return np.array(L.getFullPeriod(), dtype=int)
        # minimal internal fallback
        N = len(self.state)
        L = (2**N) - 1
        idx = [N - d for d in self.taps]  # degrees->indices (0 is MSB)
        st = self.state[:]
        out = []
        for _ in range(L):
            out.append(st[-1])
            new_bit = 0
            for i in idx:
                new_bit ^= st[i]
            st = [new_bit] + st[:-1]
        return np.array(out, dtype=int)

    def build_payload_template(self) -> np.ndarray:
        # build a local baseband template matching TX (preamble not included)
        bits = self._pn_bits()                                   # (L,)
        symbols = np.where(bits == 0, -1.0, 1.0) * math.sqrt(9/2)  # amplitude A
        x_os = self.oversample_zeros(symbols, self.O)
        h = self.srrc()
        base = np.convolve(x_os, h, mode="full")                # (L*O + P,)
        # overlap-add repeats
        out = np.zeros(self.O*self.code_length*self.numPN + self.P, dtype=base.dtype)
        for i in range(self.numPN):
            s = i * (self.O * self.code_length)
            out[s: s + (self.O*self.code_length + self.P)] += base
        return out.astype(np.complex64)  # real -> complex64

    # --------------------- Preamble & correlation ---------------------
    def generate_ltf(self) -> np.ndarray:
        try:
            mat = scipy.io.loadmat('preamble.mat')
            ltf = mat['ltf'].flatten().astype(np.complex64)
            return (self.preamble_scale * ltf).astype(np.complex64)
        except Exception:
            # CRITICAL UPDATE: fallback preamble is 4x longer (128 pairs for 256 samples)
            patt = np.array([math.sqrt(9/2), -math.sqrt(9/2)] * 128, dtype=float) 
            return (self.preamble_scale * patt.astype(np.complex64))

    def norm_xcorr_peak(self, rx: np.ndarray, tmpl: np.ndarray) -> tuple[int, np.ndarray]:
        # matched filter
        mf = np.conj(tmpl[::-1])
        corr = fftconvolve(rx, mf, mode="valid")
        # template energy
        E_t = np.sqrt(np.sum(np.abs(tmpl)**2)) + 1e-12
        # sliding window energy
        winE = np.sqrt(np.convolve(np.abs(rx)**2, np.ones(len(tmpl), dtype=np.float32), mode="valid")) + 1e-12
        ncc = np.abs(corr) / (E_t * winE)
        idx = int(np.argmax(ncc))
        return idx, ncc

    # --------------------- UHD setup (omitted) ---------------------
    def setup_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP(self.device_addr)
        if self.master_clock: self.usrp.set_master_clock_rate(self.master_clock)
        self.usrp.set_rx_antenna(self.rx_antenna)
        self.usrp.set_rx_rate(self.sample_rate)
        if self.use_lo_offset:
            lo_off = self.sample_rate/2.0 + self.lo_adjust
            treq = uhd.types.TuneRequest(self.center_freq, lo_off)
        else:
            treq = uhd.types.TuneRequest(self.center_freq)
        self.usrp.set_rx_freq(treq)
        self.usrp.set_rx_gain(self.gain)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(cmd)

    def stop_usrp(self):
        if self.rx_streamer:
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self.rx_streamer.issue_stream_cmd(cmd)

    # --------------------- Buffering (omitted) ---------------------
    def _recv_into_buffer(self, need: int):
        """Receive at least `need` fresh samples into big_buffer."""
        meta = uhd.types.RXMetadata()
        frame = np.zeros((1, self.rx_streamer.get_max_num_samps()-16), dtype=np.complex64)
        remaining = need
        while remaining > 0:
            n = self.rx_streamer.recv(frame, meta)
            if n <= 0 or meta.error_code != uhd.types.RXMetadataErrorCode.none: continue
            flat = frame.flatten()[:n]
            self.big_buffer.extend(flat)
            remaining -= len(flat)

    def _prime_buffer_and_noise(self):
        self._recv_into_buffer(self.big_buffer.maxlen)
        noise_p = []
        for _ in range(8):
            arr = np.array(self.big_buffer, dtype=np.complex64)
            noise_p.append(np.mean(np.abs(arr[-4096:])**2))
            self._recv_into_buffer(self.packet_total // 4)
        self.noise_power_est = float(np.mean(noise_p)) if noise_p else 0.0
        print(f"[INFO] Noise power estimate: {10*np.log10(self.noise_power_est+1e-12):.2f} dBFS")

    # --------------------- Main loop (omitted) ---------------------
    def run(self, threshold_factor=8.5, corr_window=128, check_payload=True, start_epoch=None):
        if start_epoch is not None:
            print(f"[INFO] Waiting until {start_epoch} to start RX..."); 
            while time.time() < start_epoch: time.sleep(0.01); print("[INFO] Starting reception.")

        self.setup_usrp()
        try:
            self._prime_buffer_and_noise()
            payload_template = self.build_payload_template() if check_payload else None

            print(f"[INFO] Expecting packet_len={self.packet_total} "
                      f"(preamble={len(self.ltf)}, pad={self.pad_zeros}, payload={self.payload_len})")

            waiting = True
            while True:
                buf = np.array(self.big_buffer, dtype=np.complex64)

                if abs(self.if_freq) > 1e-3:
                    buf = self._mix_down_if(buf, self.sample_rate, self.if_freq)

                search_end = max(0, len(buf) - self.packet_total)
                if search_end < len(self.ltf)+8:
                    self._recv_into_buffer(self.packet_total); continue

                preamble_start, ncc = self.norm_xcorr_peak(buf[:search_end], self.ltf)
                peak = float(np.max(ncc[preamble_start:preamble_start+corr_window]))
                med  = float(np.median(ncc))
                if peak < threshold_factor * med:
                    if waiting: print("[INFO] Waiting for PN/BPSK packet..."); waiting = False
                    self._recv_into_buffer(self.packet_total); continue

                data_start = preamble_start + len(self.ltf) + self.pad_zeros
                data_end   = data_start + self.payload_len
                if data_end > len(buf): self._recv_into_buffer(self.packet_total); continue

                payload = buf[data_start:data_end].astype(np.complex64)
                sig_pow = float(np.mean(np.abs(payload)**2))
                snr_est = 10*np.log10(max(sig_pow - self.noise_power_est, 1e-12) / (self.noise_power_est + 1e-12))
                print(f"[DETECT] preamble@{preamble_start} peak/med={peak/med:.2f}  SNR~{snr_est:.2f} dB")

                if check_payload and payload_template is not None:
                    pt = payload_template[:len(payload)]; num = np.vdot(pt, payload)
                    den = np.linalg.norm(pt) * np.linalg.norm(payload) + 1e-12
                    coh = np.abs(num) / den; print(f"[CHECK] Payload template coherence: {coh:.3f}")

                with open("rx_output.dat", "ab") as f:
                    buf.astype(np.complex64).tofile(f); print("[INFO] Wrote rx_output.dat chunk")

                self._recv_into_buffer(self.packet_total)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            self.stop_usrp()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-time PN/BPSK RX via UHD")
    ap.add_argument("--addr", type=str, default="addr=192.168.40.2", help="USRP device address")
    ap.add_argument("-f", "--freq", type=float, default=3.455e9, help="Center frequency (Hz)")
    ap.add_argument("-g", "--gain", type=float, default=30.0, help="RX gain (dB)")
    ap.add_argument("-r", "--rate", type=float, default=2.5e6, help="Sample rate (S/s)")
    ap.add_argument("--no_lo_offset", action="store_true", help="Disable LO offset on tune")
    ap.add_argument("--if_freq", type=float, default=0.0, help="Software IF downmix (Hz), e.g., 250e3")
    # PN/BPSK params (must match TX)
    ap.add_argument("--N_stages", type=int, default=10)
    ap.add_argument("--taps", type=str, default="10,7")
    ap.add_argument("--state", type=str, default="0,0,0,0,0,0,0,0,0,1")
    ap.add_argument("--samples_per_chip", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--Lp", type=int, default=6)
    # CRITICAL UPDATE: numPN must match TX (20)
    ap.add_argument("--numPN", type=int, default=20) 
    ap.add_argument("--pad_zeros", type=int, default=1024)
    ap.add_argument("--preamble_scale", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=8.5, help="NCC peak/median threshold")
    ap.add_argument("--corr_window", type=int, default=128, help="Peak search window after preamble start")
    ap.add_argument("--check_payload", action="store_true", help="Correlate payload with local PN template")
    ap.add_argument("--start_time", type=str, help="Start time in HH:MM format (24-hour)")

    args = ap.parse_args()

    start_epoch = None
    if args.start_time:
        now = datetime.datetime.now()
        hh, mm = map(int, args.start_time.split(":"))
        start_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if start_dt < now:
            start_dt += datetime.timedelta(days=1)
        start_epoch = start_dt.timestamp()

    rx = RealTime_PN_Detector(
        device_addr=args.addr,
        center_freq=args.freq,
        sample_rate=args.rate,
        gain=args.gain,
        use_lo_offset=(not args.no_lo_offset),
        if_freq=args.if_freq,
        N_stages=args.N_stages,
        taps=args.taps,
        state=args.state,
        samples_per_chip=args.samples_per_chip,
        alpha=args.alpha,
        Lp=args.Lp,
        numPN=args.numPN,
        pad_zeros=args.pad_zeros,
        preamble_scale=args.preamble_scale
    )
    rx.run(threshold_factor=args.threshold,
           corr_window=args.corr_window,
           check_payload=args.check_payload,
           start_epoch=start_epoch)