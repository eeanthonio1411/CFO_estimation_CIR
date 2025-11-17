import uhd
import numpy as np
import time, sys, signal, argparse, importlib, datetime, math
import struct

# Import your BPSK signal generator
import PN_code_SignalGenerationOffset
importlib.reload(PN_code_SignalGenerationOffset)
from PN_code_SignalGenerationOffset import BPSK_Generator

class BPSK_TX:
    lo_adjust = 1.5e6
    master_clock = 200e6

    def __init__(self, addr="192.168.40.2", external_clock=False, chan=0,
                 center_freq=3455e6, gain=27, samp_rate=2_500_000.0, repeat=5,
                 start_epoch=None, use_lo_offset=True, if_freq=0.0,
                 # generator params
                 N_stages=10, taps='10,7', state='0,0,0,0,0,0,0,0,0,1',
                 samples_per_chip=4, alpha=0.25, Lp=6, amplitude=None,
                 pad_zeros=1024, numPN=20, dat_file="tx_signal.dat",
                 tx_freq_offset_hz=2000.0): # ADDED: Tx offset parameter to constructor
        self.addr = addr
        self.external_clock = external_clock
        self.channel = chan
        self.center_freq = center_freq
        self.gain = gain
        self.samp_rate = samp_rate
        self.repeat = repeat
        self.start_epoch = start_epoch
        self.use_lo_offset = use_lo_offset
        self.if_freq = float(if_freq)
        self.tx_freq_offset_hz = tx_freq_offset_hz # Stored offset

        self.usrp = None
        self.txstreamer = None
        self.keep_running = True

        A = amplitude if amplitude is not None else math.sqrt(9/2)
        # UPDATED: We use the default values of the generator, but we MUST pass the sample rate
        self.generator = BPSK_Generator(
            N_stages=N_stages, taps=taps, state=state,
            samples_per_chip=samples_per_chip, samp_rate=self.samp_rate, # samp_rate is necessary for phase calcs
            alpha=alpha, A=A, Lp=Lp, pad_zeros=pad_zeros
        )
        self.numPN = int(numPN)
        self.dat_file = dat_file

    def init_radio(self):
        self.usrp = uhd.usrp.MultiUSRP(f"addr={self.addr}")
        if self.external_clock:
            self.usrp.set_time_source("external")
            self.usrp.set_clock_source("external")
        self.usrp.set_master_clock_rate(self.master_clock)
        self.usrp.set_tx_antenna("TX/RX", self.channel)

    def setup_streamers(self):
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.channel]
        self.txstreamer = self.usrp.get_tx_stream(st_args)

    def tune(self, freq, gain, rate):
        self.usrp.set_tx_rate(rate, self.channel)
        if self.use_lo_offset:
            lo_off = rate/2.0 + self.lo_adjust
            treq = uhd.types.TuneRequest(freq, lo_off)
        else:
            treq = uhd.types.TuneRequest(freq)
        self.usrp.set_tx_freq(treq, self.channel)
        self.usrp.set_tx_gain(gain, self.channel)

    def Set_all_params(self):
        self.init_radio()
        self.setup_streamers()
        self.tune(self.center_freq, self.gain, self.samp_rate)

    @staticmethod
    def _mix_to_if(iq: np.ndarray, fs: float, f_if: float) -> np.ndarray:
        if abs(f_if) < 1e-3:
            return iq
        n = np.arange(iq.size, dtype=np.float64)
        osc = np.exp(1j * 2 * np.pi * f_if * n / fs)
        return (iq * osc).astype(np.complex64)

    def _build_tx_signal(self) -> np.ndarray:
        # CRITICAL CHANGE: Pass numPN=10 and the Tx offset to the generator
        if hasattr(self.generator, "generate_bpsk_packet"):
            bb = self.generator.generate_bpsk_packet(numPN=self.numPN, 
                                                     tx_freq_offset_hz=self.tx_freq_offset_hz)
        elif hasattr(self.generator, "generate_packet"):
            bb = self.generator.generate_packet(numPN=self.numPN, 
                                                tx_freq_offset_hz=self.tx_freq_offset_hz)
        else:
            bb = self.generator.generate_bpsk_samples(numPN=self.numPN)
        iq = self._mix_to_if(bb.astype(np.complex64), self.samp_rate, self.if_freq)
        return iq

    def save_signal_to_dat(self, signal: np.ndarray):
        # Save the generated BPSK signal to a .dat file (binary format)
        with open(self.dat_file, 'ab') as f:
            # Flatten signal and write each complex sample (real, imaginary) as float32
            real_part = np.real(signal).astype(np.float32)
            imag_part = np.imag(signal).astype(np.float32)
            # Interleave real and imaginary parts
            interleaved = np.empty((real_part.size + imag_part.size,), dtype=np.float32)
            interleaved[0::2] = real_part
            interleaved[1::2] = imag_part
            interleaved.tofile(f)
        print(f"[INFO] Saved signal to {self.dat_file}")

    def send_samples(self, samples: np.ndarray):
        meta = uhd.types.TXMetadata()
        meta.start_of_burst = True
        meta.end_of_burst = False

        max_samps = self.txstreamer.get_max_num_samps() - 4
        total = samples.size
        idx = 0
        total_req = 0
        total_sent = 0
        drops = 0

        while idx < total and self.keep_running:
            nsamps = min(total - idx, max_samps)
            buf = np.zeros((1, max_samps), dtype=np.complex64)
            buf[0, :nsamps] = samples[idx:idx + nsamps]
            if idx + nsamps >= total:
                meta.end_of_burst = True
            sent = self.txstreamer.send(buf, meta)
            total_req += nsamps
            total_sent += sent
            if sent == 0:
                drops += 1
                print(f"[WARNING] {time.strftime('%Y-%m-%d %H:%M:%S')} - TX drop.")
            elif sent < nsamps:
                print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - Partial send {sent}/{nsamps}")
            idx += sent
        print(f"[TX SUMMARY] Sent {total_sent}/{total_req} samples. Drops: {drops}")

    def run(self):
        if self.start_epoch is not None:
            print(f"[INFO] Waiting until {self.start_epoch} to start TX...")
            while time.time() < self.start_epoch:
                if not self.keep_running:
                    print("[INFO] Aborted during wait.")
                    return
                time.sleep(0.01)
            print("[INFO] Starting transmission.")

        tx_signal = self._build_tx_signal()
        print(f"[INFO] TX signal length: {len(tx_signal)} @ {self.samp_rate/1e6:.3f} Msps")

        count = 0
        while self.keep_running and count < 20:  # Transmit 20 times
            self.send_samples(tx_signal)
            self.save_signal_to_dat(tx_signal)  # Save signal to .dat file after transmission
            print(f"[INFO] Transmitted BPSK PN packet {count + 1}")
            time.sleep(1.0)  # short break between transmissions
            count += 1

        if count >= 20:
            print(f"[INFO] Completed {count} transmissions. Stopping.")


def handle_interrupt(sig, frame):
    print("\n[INFO] Graceful shutdown triggered.")
    try:
        tx.keep_running = False
    except NameError:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PN/BPSK RealTime TX")
    p.add_argument("--addr", type=str, default="192.168.40.2")
    p.add_argument("--extclk", action="store_true")
    p.add_argument("--chan", type=int, default=0)
    p.add_argument("-f", "--freq", type=float, default=3455e6)
    p.add_argument("-g", "--gain", type=float, default=27.0)
    p.add_argument("-r", "--rate", type=float, default=2.5e6)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--start_time", type=str, help="HH:MM (24h)")
    p.add_argument("--no_lo_offset", action="store_true")
    p.add_argument("--if_freq", type=float, default=0.0)
    # generator params
    p.add_argument("--N_stages", type=int, default=10)
    p.add_argument("--taps", type=str, default="10,7")
    p.add_argument("--state", type=str, default="0,0,0,0,0,0,0,0,0,1")
    p.add_argument("--samples_per_chip", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--Lp", type=int, default=6)
    p.add_argument("--amplitude", type=float, default=None)
    p.add_argument("--pad_zeros", type=int, default=1024)
    # CRITICAL UPDATE 1: Set numPN default to 10 (10x payload)
    p.add_argument("--numPN", type=int, default=20) 
    p.add_argument("--dat_file", type=str, default="tx_signal.dat") 
    # CRITICAL UPDATE 2: Add argument for frequency offset
    p.add_argument("--tx_freq_offset_hz", type=float, default=2000.0, help="Deliberate offset applied to baseband signal (Hz)") 
    args = p.parse_args()

    start_epoch = None
    if args.start_time:
        now = datetime.datetime.now()
        hh, mm = map(int, args.start_time.split(":"))
        start_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if start_dt < now:
            start_dt += datetime.timedelta(days=1)
        start_epoch = start_dt.timestamp()

    tx = BPSK_TX(
        addr=args.addr, external_clock=args.extclk, chan=args.chan,
        center_freq=args.freq, gain=args.gain, samp_rate=args.rate,
        repeat=args.repeat, start_epoch=start_epoch,
        use_lo_offset=(not args.no_lo_offset), if_freq=args.if_freq,
        N_stages=args.N_stages, taps=args.taps, state=args.state,
        samples_per_chip=args.samples_per_chip, alpha=args.alpha, Lp=args.Lp,
        amplitude=args.amplitude, pad_zeros=args.pad_zeros, numPN=args.numPN,
        dat_file=args.dat_file,
        # CRITICAL UPDATE 3: Pass offset to the class constructor
        tx_freq_offset_hz=args.tx_freq_offset_hz 
    )

    signal.signal(signal.SIGINT, handle_interrupt)
    tx.Set_all_params()
    tx.run()