"""
SPECTRA LAB — Smart UV-Vis Analytical System (versi Python / Streamlit, SATU BERKAS)
Politeknik AKA Bogor · Analisis Spektrofotometri · instrumen acuan Shimadzu UV-1700

Cara menjalankan di laptop:
    pip install -r requirements.txt
    streamlit run app.py

Semua pembacaan "instrumen" berasal dari MODEL SIMULASI; aplikasi tidak terhubung ke alat.

Isi berkas:
  BAGIAN 1  Perhitungan & model simulasi (statistik, regresi, LOD/LOQ, Grubbs, kinetika, ...)
  BAGIAN 2  Materi pembelajaran & kuis
  BAGIAN 3  Interpretasi otomatis, grafik laporan, PDF
  BAGIAN 4  Model 3D instrumen (HTML/JavaScript yang disematkan)
  BAGIAN 5  Aplikasi Streamlit (halaman-halaman)
"""
from __future__ import annotations
import types

# =============================================================================
# BAGIAN 1 — PERHITUNGAN & MODEL SIMULASI
# =============================================================================

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Statistik dasar
# ---------------------------------------------------------------------------
# Nilai kritis Grubbs dua sisi, alfa = 0,05 (ISO 5725-2), n = 3..12
GRUBBS_CRIT = {3: 1.155, 4: 1.481, 5: 1.715, 6: 1.887, 7: 2.020, 8: 2.126,
               9: 2.215, 10: 2.290, 11: 2.355, 12: 2.412}


def t95(df: float) -> float:
    """Nilai t dua sisi 95 % (scipy.stats)."""
    if df < 1:
        return float("nan")
    return float(stats.t.ppf(0.975, df))


def describe(values) -> dict:
    """Rata-rata, simpangan baku (n−1), RSD %, galat baku, dan SK 95 %."""
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(a.mean()) if n else float("nan")
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    rsd = abs(sd / mean) * 100 if n > 1 and mean != 0 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "rsd": rsd, "se": se,
            "ci95": t95(n - 1) * se if n > 1 else float("nan"),
            "min": float(a.min()) if n else float("nan"), "max": float(a.max()) if n else float("nan")}


def grubbs(values) -> dict:
    """Uji Grubbs satu pencilan (dua sisi, alfa 0,05) untuk n = 3..12."""
    a = np.asarray(values, dtype=float)
    n = a.size
    if n < 3 or n > 12:
        return {"applicable": False, "n": n}
    m, s = a.mean(), a.std(ddof=1)
    if not s > 0:
        return {"applicable": False, "n": n}
    g = np.abs(a - m) / s
    i = int(np.argmax(g))
    return {"applicable": True, "n": n, "G": float(g[i]), "Gcrit": GRUBBS_CRIT[n],
            "index": i, "value": float(a[i]), "outlier": bool(g[i] > GRUBBS_CRIT[n])}


def horwitz(c_fraction: float) -> float:
    """PRSD_R (%) = 2^(1 − 0,5·log10 C), C sebagai fraksi massa."""
    return 2 ** (1 - 0.5 * math.log10(c_fraction)) if c_fraction > 0 else float("nan")


# ---------------------------------------------------------------------------
# Regresi linear kuadrat terkecil
# ---------------------------------------------------------------------------
@dataclass
class Regression:
    ok: bool
    reason: str = ""
    n: int = 0
    df: int = 0
    slope: float = float("nan")
    intercept: float = float("nan")
    r: float = float("nan")
    r2: float = float("nan")
    sxy: float = float("nan")          # galat baku residual s_y/x
    sa: float = float("nan")           # galat baku kemiringan
    sb: float = float("nan")           # galat baku intersep
    xbar: float = float("nan")
    ybar: float = float("nan")
    Sxx: float = float("nan")
    tcrit: float = float("nan")
    through_zero: bool = False
    xmin: float = float("nan")
    xmax: float = float("nan")
    yhat: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    std_residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ci_slope(self):
        return self.tcrit * self.sa

    @property
    def ci_intercept(self):
        return 0.0 if self.through_zero else self.tcrit * self.sb

    @property
    def intercept_significant(self):
        return (not self.through_zero) and abs(self.intercept) > self.tcrit * self.sb

    @property
    def lod(self):
        """LOD = 3,3·s_y/x / a (pendekatan ICH Q2)."""
        return 3.3 * self.sxy / self.slope

    @property
    def loq(self):
        """LOQ = 10·s_y/x / a."""
        return 10 * self.sxy / self.slope

    def equation(self, digits=5) -> str:
        if self.through_zero:
            return f"y = {self.slope:.{digits}g}x"
        sign = "−" if self.intercept < 0 else "+"
        return f"y = {self.slope:.{digits}g}x {sign} {abs(self.intercept):.{digits - 1}g}"


def linreg(x, y, through_zero: bool = False) -> Regression:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size:
        raise ValueError("Jumlah data x dan y tidak sama")
    min_n = 2 if through_zero else 3
    if n < min_n:
        return Regression(ok=False, reason=f"Minimal {min_n} titik data diperlukan")
    xbar, ybar = x.mean(), y.mean()
    Sxx = float(((x - xbar) ** 2).sum())
    Syy = float(((y - ybar) ** 2).sum())
    Sxy = float(((x - xbar) * (y - ybar)).sum())
    if not Sxx > 0:
        return Regression(ok=False, reason="Semua konsentrasi bernilai sama")
    if through_zero:
        a = float((x * y).sum() / (x * x).sum()); b = 0.0; df = n - 1
    else:
        a = Sxy / Sxx; b = float(ybar - a * xbar); df = n - 2
    yhat = a * x + b
    res = y - yhat
    ss_res = float((res ** 2).sum())
    sxy = math.sqrt(ss_res / df) if df > 0 else float("nan")
    r = Sxy / math.sqrt(Sxx * Syy) if Syy > 0 else float("nan")
    r2 = 1 - ss_res / Syy if Syy > 0 else float("nan")   # SStot terpusat (juga untuk model melalui nol)
    if through_zero:
        sa = sxy / math.sqrt(float((x * x).sum())); sb = 0.0
    else:
        sa = sxy / math.sqrt(Sxx)
        sb = sxy * math.sqrt(float((x * x).sum()) / (n * Sxx))
    return Regression(ok=True, n=n, df=df, slope=a, intercept=b, r=r, r2=r2, sxy=sxy, sa=sa, sb=sb,
                      xbar=float(xbar), ybar=float(ybar), Sxx=Sxx, tcrit=t95(df), through_zero=through_zero,
                      xmin=float(x.min()), xmax=float(x.max()), yhat=yhat, residuals=res,
                      std_residuals=res / sxy if sxy > 0 else res * np.nan)


def inverse_predict(reg: Regression, y0: float, m: int = 1) -> dict:
    """x0 dari rata-rata m pembacaan y0, beserta s(x0) dan SK 95 % (Miller & Miller)."""
    x0 = (y0 - reg.intercept) / reg.slope
    if reg.through_zero:
        sxx0 = reg.Sxx + reg.n * reg.xbar ** 2
        sx0 = (reg.sxy / abs(reg.slope)) * math.sqrt(1 / m + y0 ** 2 / (reg.slope ** 2 * sxx0))
    else:
        sx0 = (reg.sxy / abs(reg.slope)) * math.sqrt(1 / m + 1 / reg.n + (y0 - reg.ybar) ** 2 / (reg.slope ** 2 * reg.Sxx))
    return {"x0": x0, "sx0": sx0, "ci95": reg.tcrit * sx0, "extrapolated": x0 < reg.xmin or x0 > reg.xmax}


# ---------------------------------------------------------------------------
# Absorbansi ↔ transmitansi
# ---------------------------------------------------------------------------
def abs_to_t(A):
    return 100 * np.power(10.0, -np.asarray(A, dtype=float))


def t_to_abs(T):
    return -np.log10(np.asarray(T, dtype=float) / 100)


# ---------------------------------------------------------------------------
# Pengolahan sinyal
# ---------------------------------------------------------------------------
_SG = {5: np.array([-3, 12, 17, 12, -3]) / 35,
       7: np.array([-2, 3, 6, 7, 6, 3, -2]) / 21,
       9: np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21]) / 231}


def smooth(y, w: int = 7) -> np.ndarray:
    """Savitzky–Golay kuadratik; tepi dibiarkan apa adanya (sama dengan versi web)."""
    y = np.asarray(y, dtype=float)
    c = _SG.get(w)
    if c is None or y.size < w:
        return y.copy()
    h = (w - 1) // 2
    out = y.copy()
    out[h:-h] = np.convolve(y, c[::-1], mode="valid")
    return out


def _interp_x(x1, y1, x2, y2, yt):
    return x1 if y2 == y1 else x1 + (yt - y1) * (x2 - x1) / (y2 - y1)


def find_peaks(x, y, prominence: float = 0.01, min_height: float = 0.02, smooth_w: int = 7) -> list[dict]:
    """Maksimum lokal dengan prominensi minimum, λ diinterpolasi parabola, FWHM sederhana."""
    x = np.asarray(x, dtype=float); y0 = np.asarray(y, dtype=float)
    if x.size < 5:
        return []
    order = np.argsort(x)
    xs, ys0 = x[order], y0[order]
    ys = smooth(ys0, smooth_w)
    n = xs.size
    peaks = []
    for i in range(1, n - 1):
        if not (ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]) or ys[i] < min_height:
            continue
        lmin, j = ys[i], i - 1
        while j >= 0 and ys[j] <= ys[i]:
            lmin = min(lmin, ys[j]); j -= 1
        rmin, j = ys[i], i + 1
        while j < n and ys[j] <= ys[i]:
            rmin = min(rmin, ys[j]); j += 1
        prom = ys[i] - max(lmin, rmin)
        if prom < prominence:
            continue
        y1, y2, y3 = ys[i - 1], ys[i], ys[i + 1]
        den = y1 - 2 * y2 + y3
        off = 0.5 * (y1 - y3) / den if den != 0 else 0.0
        lam = xs[i] + off * (xs[i + 1] - xs[i - 1]) / 2
        half = ys[i] - prom / 2
        L = i
        while L > 0 and ys[L] > half:
            L -= 1
        R = i
        while R < n - 1 and ys[R] > half:
            R += 1
        fwhm = _interp_x(xs[R - 1], ys[R - 1], xs[R], ys[R], half) - _interp_x(xs[L], ys[L], xs[L + 1], ys[L + 1], half)
        peaks.append({"lambda": float(lam), "abs": float(ys0[i]), "prominence": float(prom), "fwhm": float(fwhm)})
    return sorted(peaks, key=lambda p: -p["abs"])


def interp_y(x, y, xt):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    if xt < x.min() or xt > x.max():
        return float("nan")
    return float(np.interp(xt, x[order], y[order]))


# ---------------------------------------------------------------------------
# Kinetika
# ---------------------------------------------------------------------------
def kinetic_model(A0, Ainf, k, t):
    """A(t) = A∞ + (A0 − A∞)·e^(−kt)  (orde satu semu)."""
    return Ainf + (A0 - Ainf) * np.exp(-k * np.asarray(t, dtype=float))


def initial_rate(t, A, n_pts: int = 5):
    k = min(n_pts, len(t))
    if k < 3:
        return None
    r = linreg(t[:k], A[:k])
    return {"rate": r.slope, "r2": r.r2, "n": k, "intercept": r.intercept} if r.ok else None


def first_order_fit(t, A, Ainf):
    t = np.asarray(t, dtype=float); A = np.asarray(A, dtype=float)
    d = np.abs(A - Ainf)
    mask = d > 1e-4
    if mask.sum() < 3:
        return None
    r = linreg(t[mask], np.log(d[mask]))
    if not r.ok:
        return None
    k = -r.slope
    return {"k": k, "half_life": math.log(2) / k if k > 0 else float("nan"), "r2": r.r2,
            "n": int(mask.sum()), "lnA0": r.intercept, "t": t[mask], "lny": np.log(d[mask])}


# ---------------------------------------------------------------------------
# Parser angka & CSV (pemisah , ; tab — desimal koma didukung)
# ---------------------------------------------------------------------------
def parse_number(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    if s is None:
        return float("nan")
    s = re.sub(r"\s", "", str(s))
    if not s:
        return float("nan")
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d*,\d+(e[-+]?\d+)?", s, flags=re.I):
        s = s.replace(",", ".")
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?", s, flags=re.I):
        return float("nan")
    return float(s)


def parse_list(text: str) -> list[float]:
    """'0,412; 0,415; 0,409' → [0.412, 0.415, 0.409]; ValueError bila ada yang bukan angka."""
    parts = [p for p in re.split(r"[;\s|]+", text or "") if p]
    vals = [parse_number(p) for p in parts]
    if any(math.isnan(v) for v in vals):
        raise ValueError("Pisahkan nilai dengan titik koma, mis. 0,412; 0,415")
    return vals


def parse_csv(text: str) -> dict:
    lines = [l for l in text.replace("\r", "").split("\n") if l.strip()]
    if not lines:
        return {"ok": False, "reason": "Berkas kosong"}
    sample = "\n".join(lines[:20])
    cnt = {";": sample.count(";"), "\t": sample.count("\t"), ",": sample.count(",")}
    delim = "\t" if cnt["\t"] >= cnt[";"] and cnt["\t"] > 0 else (";" if cnt[";"] > 0 else ",")
    header, data, skipped = None, [], 0
    for line in lines:
        cells = [c.strip().strip('"') for c in line.split(delim)]
        nums = [parse_number(c) for c in cells]
        if sum(not math.isnan(v) for v in nums) >= 2 and not math.isnan(nums[0]):
            data.append(nums)
        else:
            if not data:
                header = cells
            skipped += 1
    if not data:
        return {"ok": False, "reason": "Tidak ditemukan baris numerik (minimal 2 kolom angka)"}
    ncol = max(len(r) for r in data)
    cols = [np.array([r[c] if c < len(r) else np.nan for r in data]) for c in range(ncol)]
    return {"ok": True, "delimiter": delim, "header": header, "cols": cols, "nrows": len(data), "skipped": skipped}


# ---------------------------------------------------------------------------
# Model simulasi instrumen (DISEDERHANAKAN; absorptivitas mendekati literatur)
# ---------------------------------------------------------------------------
ANALYTES = {
    "kmno4": dict(nama="Kalium permanganat", rumus="KMnO₄", pelarut="Akuades", lam_ref=525, abs_ref=0.0155,
                  bands=[(489, .22, 10), (507, .62, 8), (526, 1.0, 8.5), (546, .97, 8), (566, .46, 8.5), (311, .30, 20), (355, .12, 22)],
                  std=[10, 20, 30, 40, 50], reagent=None, warna="#8e2c8f",
                  catatan="Pita multipel 500–570 nm khas transfer muatan ligan→logam MnO₄⁻."),
    "fephen": dict(nama="Fe(II)–1,10-fenantrolin", rumus="[Fe(phen)₃]²⁺", pelarut="Bufer asetat pH ≈ 4,5", lam_ref=510, abs_ref=0.199,
                   bands=[(514, 1.0, 22), (478, .30, 15), (400, .08, 30), (265, 1.6, 18)],
                   std=[0.5, 1, 2, 3, 4, 5], reagent=dict(nama="hidroksilamin + 1,10-fenantrolin", bands=[(265, .55, 18), (228, .9, 14)]),
                   warna="#c0392b", catatan="Kompleks oranye-merah; Fe(III) direduksi dengan hidroksilamin."),
    "crdpc": dict(nama="Cr(VI)–difenilkarbazida", rumus="Cr(VI)–DPC", pelarut="H₂SO₄ (pH ≈ 1–2)", lam_ref=540, abs_ref=0.77,
                  bands=[(542, 1.0, 40), (360, .08, 30), (290, .35, 22)], std=[0.1, 0.2, 0.3, 0.4, 0.5],
                  reagent=dict(nama="difenilkarbazida", bands=[(285, .50, 18), (540, .006, 60), (235, .9, 15)]),
                  warna="#9b2d6f", catatan="Kompleks merah-ungu; blanko reagen penting."),
    "mb": dict(nama="Metilen biru", rumus="C₁₆H₁₈ClN₃S", pelarut="Akuades", lam_ref=664, abs_ref=0.23,
               bands=[(664, 1.0, 19), (612, .42, 24), (292, .62, 14), (246, .32, 14)], std=[0.5, 1, 2, 3, 4],
               reagent=None, warna="#1f5fbf", catatan="Bahu 612 nm berasal dari dimer."),
    "k2cr2o7": dict(nama="Kalium dikromat (HClO₄ 0,001 M)", rumus="K₂Cr₂O₇", pelarut="HClO₄ 0,001 M", lam_ref=350, abs_ref=0.0107,
                    bands=[(257, 1.36, 17), (350, 1.0, 24), (440, .055, 28), (205, .9, 10)], std=[20, 40, 60, 80, 100],
                    reagent=None, warna="#d98a00", catatan="Lazim untuk verifikasi akurasi fotometrik."),
    "pct": dict(nama="Parasetamol (metanol)", rumus="C₈H₉NO₂", pelarut="Metanol", lam_ref=249, abs_ref=0.0668,
                bands=[(249, 1.0, 15), (205, .85, 9), (290, .06, 15)], std=[2, 4, 6, 8, 10, 12],
                reagent=None, warna="#5f6b7a", catatan="Daerah UV wajib kuvet kuarsa."),
}


def _band_sum(bands, lam):
    lam = np.asarray(lam, dtype=float)
    return sum(h * np.exp(-((lam - l) ** 2) / (2 * w * w)) for l, h, w in bands)


for _a in ANALYTES.values():
    _a["k"] = _a["abs_ref"] / float(_band_sum(_a["bands"], _a["lam_ref"]))


def absorptivity(analyte_id: str, lam):
    """Absorbansi per (mg/L · cm) dari model."""
    a = ANALYTES[analyte_id]
    return a["k"] * _band_sum(a["bands"], lam)


def true_abs(analyte_id: str, conc: float, lam, path: float = 1.0):
    return conc * absorptivity(analyte_id, lam) * path


CUVETTES = {
    "kuarsa": dict(nama="Kuarsa", cutoff=190, A=lambda l: 0.02 + 2.5 / (1 + np.exp((l - 175) / 4))),
    "kaca": dict(nama="Kaca optik", cutoff=320, A=lambda l: 0.04 + 4 / (1 + np.exp((l - 318) / 7))),
    "plastik": dict(nama="Polistirena (PS)", cutoff=340, A=lambda l: 0.05 + 4 / (1 + np.exp((l - 330) / 6))),
}
SOLVENT_CUTOFF = {"Akuades": 190, "Metanol": 205, "HClO₄ 0,001 M": 195, "Bufer asetat pH ≈ 4,5": 230, "H₂SO₄ (pH ≈ 1–2)": 200}


def _solvent_A(name, lam):
    return 3 / (1 + np.exp((lam - SOLVENT_CUTOFF.get(name, 190)) / 5))


def lamp_energy(lam, switch_at=340.8):
    lam = np.asarray(lam, dtype=float)
    e = np.where(lam < switch_at,
                 0.15 + 0.85 * np.exp(-((lam - 240) ** 2) / (2 * 70 ** 2)),
                 0.08 + 0.92 * np.exp(-((lam - 850) ** 2) / (2 * 300 ** 2)))
    e = np.where(lam < 200, e * np.maximum(0.08, (lam - 185) / 15), e)
    return np.maximum(e, 0.02)


def instrument_baseline(lam, switch_at=340.8):
    lam = np.asarray(lam, dtype=float)
    return (0.014 + 1.6e-5 * (lam - 500) + 0.006 * np.sin(lam / 37) + np.where(lam < switch_at, 0.0035, 0)
            + 0.01 * np.exp(-((lam - 195) ** 2) / 60))


def _stray(lam):
    return np.where(np.asarray(lam) < 230, 0.0004, 0.0001)


def measure(lam, analyte_id="kmno4", conc=0.0, path=1.0, cuvette="kuarsa", blank="pelarut",
            baseline_done=True, speed="sedang", lamp_switch=340.8, matrix=0.0, rng=None):
    """Absorbansi terukur (simulasi) pada satu atau banyak λ.

    Meliputi serapan kuvet/pelarut/reagen, blanko di sel referensi, sinar sesat,
    offset baseline bila koreksi belum dilakukan, dan derau fotometrik σA = 0,434·σT/T.
    """
    rng = rng or np.random.default_rng()
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    an = ANALYTES[analyte_id]
    cuv = CUVETTES[cuvette]
    sol = an["pelarut"]
    reagent = 0.25 * path * _band_sum(an["reagent"]["bands"], lam) if an["reagent"] else 0.0
    mat = matrix * np.exp(-((lam - 300) ** 2) / (2 * 120 ** 2)) * path
    sample_cell = cuv["A"](lam) + _solvent_A(sol, lam) + reagent + conc * absorptivity(analyte_id, lam) * path + mat
    if blank == "udara":
        ref_cell = np.zeros_like(lam)
    elif blank == "reagen":
        ref_cell = cuv["A"](lam) + _solvent_A(sol, lam) + reagent
    else:
        ref_cell = cuv["A"](lam) + _solvent_A(sol, lam)
    s = _stray(lam)
    Ts = (10 ** (-sample_cell) + s) / (1 + s)
    Tr = (10 ** (-ref_cell) + s) / (1 + s)
    A = -np.log10(Ts / Tr)
    if not baseline_done:
        A = A + instrument_baseline(lam, lamp_switch)
    sig_t = {"cepat": 6e-4, "sedang": 3e-4, "lambat": 1.5e-4}[speed]
    E = lamp_energy(lam, lamp_switch) * Ts
    sig_a = np.minimum(0.4343 * sig_t / np.maximum(E, 1e-6), 0.5)
    A = A + rng.normal(0, 1, lam.size) * sig_a
    return {"A": np.clip(A, -0.5, 4.0), "over": (A > 3.0) | (A < -0.5), "low_energy": E < 2e-3}


def simulate_scan(start, end, interval, **kw) -> dict:
    """Pemindaian dari λ panjang ke pendek seperti UV-1700."""
    lo, hi = max(190, min(start, end)), min(1100, max(start, end))
    n = int(round((hi - lo) / interval)) + 1
    lam = np.round(hi - np.arange(n) * interval, 3)
    m = measure(lam, **kw)
    return {"lam": lam, "A": m["A"], "T": abs_to_t(m["A"]), "over": bool(m["over"].any()),
            "low_energy": bool(m["low_energy"].any()),
            "uv_cutoff": lo < CUVETTES[kw.get("cuvette", "kuarsa")]["cutoff"]}


def prep_conc(c: float, rng=None, rsd: float = 0.004) -> float:
    """Konsentrasi standar 'sebenarnya' dengan galat preparasi acak (RSD ≈ 0,4 %)."""
    rng = rng or np.random.default_rng()
    return c * (1 + rsd * rng.normal())


def hidden_conc(analyte_id: str, seed: int) -> float:
    """Konsentrasi Sampel X tersembunyi (deterministik per sesi)."""
    an = ANALYTES[analyte_id]
    r = np.random.default_rng((seed * 31 + sum(map(ord, analyte_id))) % (2 ** 32)).random()
    lo, hi = an["std"][1], an["std"][-2]
    return float(f"{lo + (hi - lo) * (0.15 + 0.7 * r):.3g}")


# ---------------------------------------------------------------------------
# Analisis sampel (dipakai aplikasi & laporan)
# ---------------------------------------------------------------------------
def dilution_factor(steps=None, manual=None) -> float:
    """FP = Π(V labu / V pipet), atau nilai manual."""
    if manual is not None:
        return float(manual) if manual > 0 else float("nan")
    f = 1.0
    for vp, vl in steps or []:
        if not (vp > 0 and vl > 0):
            return float("nan")
        f *= vl / vp
    return f


def sample_result(reg: Regression, reps, fp=1.0, mass_g=None, vol_ml=None, unit="mg/L",
                  a_min=0.1, a_max=1.0, rsd_max=2.0) -> dict | None:
    reps = [r for r in reps if not math.isnan(r)]
    if not reg.ok or not reps:
        return None
    d = describe(reps)
    ip = inverse_predict(reg, d["mean"], len(reps))
    c_final, ci_final = ip["x0"] * fp, ip["ci95"] * fp
    mass_ok = bool(mass_g and vol_ml and mass_g > 0 and vol_ml > 0 and unit in ("mg/L", "µg/mL", "ppm"))
    kadar = c_final * vol_ml / mass_g if mass_ok else float("nan")   # mg/kg
    flags = []
    if ip["extrapolated"]:
        flags.append(f"C terukur ({ip['x0']:.3g}) di luar rentang standar {reg.xmin:.3g}–{reg.xmax:.3g} {unit}; hasil ekstrapolasi.")
    if d["mean"] < a_min or d["mean"] > a_max:
        flags.append(f"Ā = {d['mean']:.3f} di luar daerah kerja absorbansi ({a_min}–{a_max}).")
    if len(reps) > 1 and d["rsd"] > rsd_max:
        flags.append(f"RSD ulangan {d['rsd']:.2f} % melebihi batas {rsd_max} %.")
    if len(reps) < 2:
        flags.append("Hanya satu pembacaan; presisi tidak dapat dievaluasi.")
    if ip["x0"] < 0:
        flags.append("Konsentrasi negatif: absorbansi sampel di bawah intersep.")
    elif ip["x0"] < reg.loq:
        flags.append(f"C terukur di bawah LOQ ({reg.loq:.3g} {unit}); nilai indikatif.")
    return {"n": len(reps), "A_mean": d["mean"], "sd": d["sd"], "rsd": d["rsd"], "c_meas": ip["x0"],
            "sx0": ip["sx0"], "ci": ip["ci95"], "fp": fp, "c_final": c_final, "ci_final": ci_final,
            "kadar": kadar, "mass_ok": mass_ok, "flags": flags, "extrapolated": ip["extrapolated"]}


def recovery_spike(c_sample, c_spiked, c_added) -> float:
    """%R = (C sampel+spike − C sampel) / C ditambahkan × 100."""
    return (c_spiked - c_sample) / c_added * 100


def recovery_reference(c_measured, c_reference) -> dict:
    return {"R": c_measured / c_reference * 100, "bias": c_measured - c_reference,
            "rel_bias": (c_measured - c_reference) / c_reference * 100}


def residual_runs(residuals) -> int:
    """Jumlah kelompok tanda residual (runs); sedikit runs → pola melengkung."""
    s = [np.sign(v) for v in residuals if v != 0]
    return (1 + sum(1 for a, b in zip(s, s[1:]) if a != b)) if s else 0


def lam_color(l: float) -> str:
    if l < 380:
        return "#a99cf0"
    if l > 780:
        return "#b0302a"
    r = g = b = 0.0
    if l < 440: r, b = -(l - 440) / 60, 1
    elif l < 490: g, b = (l - 440) / 50, 1
    elif l < 510: g, b = 1, -(l - 510) / 20
    elif l < 580: r, g = (l - 510) / 70, 1
    elif l < 645: r, g = 1, -(l - 645) / 65
    else: r = 1
    f = 0.45 + 0.55 * (l - 380) / 40 if l < 420 else (0.45 + 0.55 * (780 - l) / 80 if l > 700 else 1)
    return "#" + "".join(f"{round(255 * (v * f) ** 0.8):02x}" for v in (r, g, b))


def region(l: float) -> str:
    return "UV vakum (tepi)" if l < 200 else "Ultraviolet (UV)" if l < 400 else "Sinar tampak (Vis)" if l <= 780 else "Inframerah dekat (NIR)"


C = types.SimpleNamespace(**{k: v for k, v in dict(globals()).items() if not k.startswith("__")})

# =============================================================================
# BAGIAN 2 — MATERI PEMBELAJARAN & KUIS
# =============================================================================

COMPONENTS = {
    "Lampu deuterium (D₂)": "Sumber kontinu daerah UV ±190–350 nm. Energinya turun tajam di daerah tampak, sehingga alat berganti ke lampu WI di atas λ pergantian (bawaan 340,8 nm pada UV-1700).",
    "Lampu wolfram-iodin (WI)": "Sumber kontinu daerah tampak hingga NIR ±340–1100 nm. Pada layar alat tertulis \"WI\" saat lampu ini aktif.",
    "Monokromator (celah + kisi difraksi)": "Celah masuk, kisi difraksi holografik, dan celah keluar memilih pita sempit (±1 nm) di sekitar λ yang diatur. Kualitasnya menentukan besar sinar sesat.",
    "Pembagi berkas (double-beam)": "Membagi cahaya monokromatis ke berkas referensi dan berkas sampel sehingga fluktuasi lampu terkompensasi.",
    "Kuvet": "Kuarsa (≥190 nm), kaca optik (≥320 nm), plastik PS (≥340 nm). Pegang sisi buram, bilas dengan larutan yang diukur, hindari gelembung.",
    "Detektor fotodioda Si": "Mengubah intensitas cahaya menjadi arus listrik. UV-1700 memakai fotodioda silikon pada kedua jalur.",
    "Pengolah sinyal": "ADC dan prosesor menghitung T = I/I₀ dan A = −log T untuk layar atau perangkat lunak.",
}

LESSONS = {
    "Prinsip kerja": """
Cahaya polikromatis dari lampu diuraikan oleh **monokromator** menjadi pita sempit panjang gelombang.
Berkas itu dibagi menjadi berkas **referensi** (melewati blanko) dan berkas **sampel**. Molekul yang memiliki
transisi elektronik (π→π*, n→π*, transfer muatan) pada energi yang sesuai menyerap sebagian foton.
Detektor membandingkan intensitas yang diteruskan sampel (I) dengan referensi (I₀):

**T = I / I₀** dan **A = −log T = log (I₀ / I)**.
""",
    "Fungsi blanko": """
Blanko berisi semua komponen larutan **kecuali analit**: pelarut, bufer, reagen pembentuk warna.
Karena I₀ diukur melalui blanko, serapan pelarut, reagen, dan dinding kuvet ikut terkoreksi.

- **Blanko pelarut**: cukup untuk analit yang menyerap sendiri (KMnO₄, dikromat).
- **Blanko reagen**: wajib untuk metode kolorimetri (Fe–fenantrolin, Cr(VI)–DPC).
- **Blanko sampel**: sampel tanpa reagen, untuk mengoreksi warna/kekeruhan matriks.
- **Udara**: hanya untuk pemeriksaan kinerja alat.
""",
    "Faktor yang memengaruhi": """
- **pH dan pelarut** mengubah bentuk spesies, sehingga λmaks dan ε bergeser.
- **Waktu dan kestabilan warna**: ukur dalam jendela waktu yang ditetapkan metode.
- **Suhu** memengaruhi kesetimbangan dan laju pembentukan warna.
- **Konsentrasi tinggi**: interaksi antarmolekul dan dimerisasi menyebabkan penyimpangan linearitas.
- **Sinar sesat dan lebar pita** membuat kurva melengkung pada absorbansi tinggi.
- **Gangguan matriks**: kekeruhan, zat berwarna lain, ion pengganggu.
- **Kuvet**: bahan tidak sesuai, goresan, sidik jari, orientasi berubah.
""",
    "Kesalahan umum": """
| Kesalahan | Akibat |
|---|---|
| Tidak melakukan baseline/auto zero | Offset instrumen terbaca; intersep besar |
| Kuvet kaca di daerah UV | Energi hampir nol < 320 nm; data hanya derau |
| Blanko tidak sesuai | Serapan reagen terbaca sebagai analit |
| Sidik jari atau gelembung | Hamburan menaikkan absorbansi |
| A di luar daerah kerja | Galat relatif meningkat |
| Mengekstrapolasi kurva | Hasil tidak dijamin linear |
| Lupa faktor pengenceran | Hasil akhir meleset sebesar FP |
""",
}

QUIZ = [
    ("Absorbansi suatu larutan 0,500. Berapa persen transmitansinya?", ["50,0 %", "31,6 %", "5,0 %", "68,4 %"], 1,
     "%T = 100 × 10⁻ᴬ = 100 × 10⁻⁰·⁵ = 31,6 %. Hubungannya logaritmik, bukan linear."),
    ("Mengapa pengukuran kuantitatif dilakukan pada λmaks?", ["Derau lampu paling besar di sana", "Kepekaan tertinggi dan dA/dλ ≈ 0 sehingga tahan terhadap pergeseran λ kecil", "Hukum Lambert–Beer hanya berlaku pada λmaks", "Agar absorbansi selalu di bawah 0,1"], 1,
     "Di puncak, absorptivitas terbesar dan kemiringan spektrum mendekati nol, sehingga galat akibat ketidaktepatan λ minimal."),
    ("Fungsi utama larutan blanko adalah…", ["Menaikkan absorbansi sampel", "Mengoreksi serapan pelarut, reagen, dan kuvet", "Mengkalibrasi panjang gelombang", "Menghilangkan sinar sesat"], 1,
     "Blanko memuat semua komponen kecuali analit; absorbansi sampel diukur relatif terhadapnya."),
    ("Kuvet kaca dipakai untuk parasetamol pada 249 nm. Akibatnya…", ["Tidak ada masalah", "Kaca menyerap kuat di bawah ±320 nm, energi ke detektor sangat kecil, pembacaan tidak bermakna", "Absorbansi tepat dua kali lipat", "Lampu WI rusak"], 1,
     "Daerah UV memerlukan kuvet kuarsa."),
    ("Kurva y = 0,198x + 0,003 (x dalam mg/L). Sampel A = 0,450 setelah pengenceran 5 kali. C sampel asal?", ["2,26 mg/L", "11,29 mg/L", "0,45 mg/L", "2,29 mg/L"], 1,
     "x = (0,450 − 0,003)/0,198 = 2,258 mg/L; × FP 5 = 11,29 mg/L."),
    ("R² = 0,9990. Pernyataan paling tepat…", ["Linearitas pasti terbukti", "99,9 % variasi A dijelaskan model linear; plot residual tetap perlu diperiksa", "Intersep pasti nol", "Tidak perlu ulangan sampel"], 1,
     "R² tinggi tidak menjamin linearitas; pola residual sistematis dapat muncul pada R² > 0,999."),
    ("Penyebab penyimpangan negatif Lambert–Beer pada konsentrasi tinggi, kecuali…", ["Sinar sesat", "Dimerisasi", "Radiasi polikromatis", "Kuvet bersih dan blanko sesuai"], 3,
     "Kuvet bersih dan blanko sesuai adalah praktik yang benar."),
    ("Keuntungan utama sistem double-beam…", ["Tidak perlu blanko", "Fluktuasi lampu dan drift terkompensasi", "Rentang λ lebih lebar", "Tidak perlu monokromator"], 1,
     "Rasio I/I₀ diukur hampir bersamaan sehingga perubahan intensitas lampu saling meniadakan."),
    ("Galat relatif konsentrasi minimum (galat %T konstan) pada A sekitar…", ["0,043", "0,434", "1,5", "2,5"], 1,
     "Analisis Ringbom/Twyman–Lothian: minimum pada A = 0,434 (T = 36,8 %)."),
    ("Ulangan 0,412; 0,415; 0,409 A. RSD-nya mendekati…", ["0,73 %", "7,3 %", "0,003 %", "3,0 %"], 0,
     "x̄ = 0,4120; s = 0,0030; RSD = 0,73 %."),
]


# =============================================================================
# BAGIAN 3 — INTERPRETASI, GRAFIK LAPORAN, PDF
# =============================================================================

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY, BLUE, CYAN, MUTED = "#0b1b3a", "#1f6fff", "#0a9fbd", "#56688a"


def fnum(x, d=4, dec="koma"):
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return "—"
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace(".", ",") if dec == "koma" else s


def fsig(x, sig=4, dec="koma"):
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return fnum(0, sig - 1, dec)
    d = max(0, min(8, sig - 1 - int(np.floor(np.log10(abs(x))))))
    return fnum(x, d, dec)


# ---------------------------------------------------------------------------
# Interpretasi otomatis (hanya dari data yang tersedia)
# ---------------------------------------------------------------------------
def calib_interpretation(reg: C.Regression, unit: str, r2_min: float, std_res_max: float, dec="koma") -> list[str]:
    if not reg.ok:
        return []
    f = lambda x, d=4: fnum(x, d, dec)
    g = lambda x, s=4: fsig(x, s, dec)
    out = [f"Linearitas. Persamaan regresi {reg.equation()} dengan r = {f(reg.r)} dan R² = {f(reg.r2)}. "
           f"R² {'memenuhi' if reg.r2 >= r2_min else 'tidak memenuhi'} kriteria yang ditetapkan (≥ {f(r2_min, 3)}); "
           f"{f(reg.r2 * 100, 2)} % variasi absorbansi dijelaskan oleh konsentrasi melalui model linear.",
           f"Kepekaan. Kemiringan {g(reg.slope)} A per {unit} (± {g(reg.ci_slope, 2)}, SK 95 %). Bila b = 1 cm dan satuan mg/L, "
           f"kemiringan adalah absorptivitas a (L·mg⁻¹·cm⁻¹); kalikan dengan massa molar (mg/mol) untuk memperoleh ε."]
    if not reg.through_zero:
        out.append(f"Intersep. b = {g(reg.intercept, 3)} ± {g(reg.ci_intercept, 2)} (SK 95 %). " +
                   ("Intersep berbeda nyata dari nol; telusuri serapan latar (blanko, reagen, offset baseline)."
                    if reg.intercept_significant else
                    "Intersep tidak berbeda nyata dari nol pada taraf 5 %, konsisten dengan hukum Lambert–Beer."))
    out.append(f"Batas deteksi. Dari s_y/x = {g(reg.sxy, 3)}: LOD = 3,3·s_y/x/a = {g(reg.lod, 3)} {unit} dan "
               f"LOQ = 10·s_y/x/a = {g(reg.loq, 3)} {unit} (perkiraan dari kurva; LOD formal memakai ulangan blanko).")
    big = int((np.abs(reg.std_residuals) > std_res_max).sum())
    if big:
        out.append(f"Residual. {big} titik memiliki residual terstandar melebihi ±{f(std_res_max, 1)}; periksa pembuatan larutan atau pembacaannya sebelum dihapus.")
    runs = C.residual_runs(reg.residuals)
    if reg.n >= 5 and runs <= 2:
        out.append(f"Pola residual. Tanda residual hanya berganti {runs - 1} kali (pola melengkung), tanda ketidaklinearan.")
    if reg.n < 5:
        out.append(f"Jumlah titik. Kurva hanya memiliki {reg.n} titik; banyak pedoman menganjurkan minimal 5 tingkat konsentrasi.")
    elif reg.r2 >= r2_min:
        out.append("Catatan: R² yang tinggi saja tidak membuktikan linearitas; periksa juga plot residual.")
    return out


# ---------------------------------------------------------------------------
# Grafik Matplotlib (tema terang, untuk laporan)
# ---------------------------------------------------------------------------
def _style(ax, xl, yl):
    ax.set_xlabel(xl, color=MUTED); ax.set_ylabel(yl, color=MUTED)
    ax.grid(True, color="#e6ecf5", lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#b9c6da")
    ax.tick_params(colors=MUTED, labelsize=8)


def _png(fig) -> bytes:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def spectrum_png(lam, A, name, peak=None) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    o = np.argsort(lam)
    ax.plot(np.asarray(lam)[o], np.asarray(A)[o], color=BLUE, lw=1.6, label=name)
    ax.fill_between(np.asarray(lam)[o], np.asarray(A)[o], color=BLUE, alpha=0.08)
    if peak:
        ax.axvline(peak, color=CYAN, ls="--", lw=1); ax.text(peak, ax.get_ylim()[1] * 0.95, " λmaks " + fnum(peak, 1) + " nm", color=CYAN, fontsize=8, va="top")
    _style(ax, "Panjang gelombang (nm)", "Absorbansi"); ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def calib_png(x, y, reg: C.Regression, samples=(), unit="mg/L") -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.scatter(x, y, color=BLUE, zorder=3, label="Standar")
    xs = np.array([min(0, reg.xmin), reg.xmax * 1.05])
    ax.plot(xs, reg.slope * xs + reg.intercept, color=CYAN, lw=1.6, label=f"{reg.equation(4)}   R² = {reg.r2:.4f}")
    for name, cx, ay in samples:
        ax.scatter([cx], [ay], marker="D", color="#c2368a", zorder=4); ax.annotate(name, (cx, ay), textcoords="offset points", xytext=(6, 6), fontsize=8)
    _style(ax, f"Konsentrasi ({unit})", "Absorbansi"); ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8)
    return _png(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _font():
    """DejaVu Sans dari Matplotlib → mendukung λ, ², ±, subskrip."""
    try:
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_pdf(model: dict) -> bytes:
    """model: dict berisi blok laporan (lihat app.report_model)."""
    F, FB = _font()
    st = {
        "h": ParagraphStyle("h", fontName=FB, fontSize=12, leading=15, textColor=colors.HexColor(NAVY), spaceBefore=10, spaceAfter=4),
        "p": ParagraphStyle("p", fontName=F, fontSize=9.5, leading=13.5, textColor=colors.HexColor("#10203f"), alignment=TA_LEFT, spaceAfter=5),
        "note": ParagraphStyle("n", fontName=F, fontSize=8.5, leading=11.5, textColor=colors.HexColor(MUTED)),
        "cell": ParagraphStyle("c", fontName=F, fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("cb", fontName=FB, fontSize=8, leading=10.5, textColor=colors.HexColor("#3d4f70")),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=36 * mm, bottomMargin=18 * mm,
                            title="Laporan Analisis Spektrofotometri UV-Vis", author=model.get("praktikan") or "SPECTRA LAB")
    W = A4[0] - 36 * mm
    story = []
    if model.get("simulated"):
        t = Table([[Paragraph("<b>Mengandung data simulasi/contoh:</b> hasil latihan, bukan hasil pengujian laboratorium.", st["cell"])]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf0d9")), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#8a5200")), ("BOX", (0, 0), (-1, -1), 0, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story += [t, Spacer(1, 6)]
    for b in model["blocks"]:
        kind = b["t"]
        if kind == "h":
            line = Table([[""]], colWidths=[W], rowHeights=[1.2]); line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BLUE))]))
            kt = KeepTogether([Paragraph(b["text"], st["h"]), line, Spacer(1, 5)]); kt.keepWithNext = 1
            story.append(kt)
        elif kind == "p":
            story.append(Paragraph(b["text"], st["p"]))
        elif kind == "note":
            story.append(Paragraph(b["text"], st["note"]))
        elif kind == "kv":
            rows = [[Paragraph(k, st["note"]), Paragraph(str(v), st["cell"])] for k, v in b["rows"]]
            t = Table(rows, colWidths=[52 * mm, W - 52 * mm]); t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
            story += [t, Spacer(1, 4)]
        elif kind == "table":
            data = [[Paragraph(h, st["cellb"]) for h in b["head"]]] + [[Paragraph(str(c), st["cell"]) for c in r] for r in b["rows"]]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe6f0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 6)]
        elif kind == "img":
            img = Image(io.BytesIO(b["png"])); ratio = img.imageHeight / img.imageWidth
            img.drawWidth = W; img.drawHeight = W * ratio
            story += [img, Paragraph(b["cap"], st["note"]), Spacer(1, 6)]
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    def deco(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(NAVY)); canvas.rect(0, A4[1] - 30 * mm, A4[0], 30 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#63dcf2")); canvas.setFont(F, 8.5); canvas.drawString(18 * mm, A4[1] - 11 * mm, "SPECTRA LAB  ·  Smart UV-Vis Analytical System")
        canvas.setFillColor(colors.white); canvas.setFont(FB, 15); canvas.drawString(18 * mm, A4[1] - 19 * mm, "LAPORAN ANALISIS SPEKTROFOTOMETRI UV-VIS")
        canvas.setFillColor(colors.HexColor("#a9bde4")); canvas.setFont(F, 9); canvas.drawString(18 * mm, A4[1] - 25 * mm, (model.get("judul") or "") + ("  ·  " + model["tanggal"] if model.get("tanggal") else ""))
        canvas.setStrokeColor(colors.HexColor("#dfe6f0")); canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFillColor(colors.HexColor(MUTED)); canvas.setFont(F, 7.5)
        canvas.drawString(18 * mm, 9 * mm, f"SPECTRA LAB · dibuat {stamp}"); canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Halaman {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


R = types.SimpleNamespace(fnum=fnum, fsig=fsig, calib_interpretation=calib_interpretation,
                          spectrum_png=spectrum_png, calib_png=calib_png, build_pdf=build_pdf)

# =============================================================================
# BAGIAN 4 — MODEL 3D INSTRUMEN (kanvas JavaScript, disematkan lewat components.html)
# =============================================================================
MODEL3D_HTML = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--fg:#0b1b3a;--muted:#56688a;--line:#d3ddeb;--bg:#ffffff;--soft:#e3edff;--acc:#1f6fff}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.45 'IBM Plex Sans',system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:10px;height:100%;padding:4px}
.stage{position:relative;border-radius:10px;overflow:hidden;min-height:300px}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.tools{position:absolute;top:8px;right:8px;display:flex;gap:6px}
.views{position:absolute;left:8px;bottom:8px;display:flex;flex-wrap:wrap;gap:4px;background:rgba(255,255,255,.85);padding:4px;border-radius:8px;max-width:calc(100% - 16px)}
button{font:inherit;border:1px solid var(--line);background:#fff;color:var(--fg);border-radius:7px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--acc)} .views button{border:0;background:none;color:var(--muted)} .views button.on{background:#fff;color:var(--fg);font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.side{display:flex;flex-direction:column;gap:8px;overflow:auto}
.tg{display:grid;grid-template-columns:1fr 1fr;gap:6px}.tg button{text-align:left;font-size:12px;color:var(--muted)}
.tg button[aria-pressed=true]{border-color:var(--acc);background:var(--soft);color:var(--fg);font-weight:600}
.card{border:1px solid var(--line);border-radius:10px;padding:10px}
h4{margin:0 0 6px;font-size:13px}.hint{color:var(--muted);font-size:12px}
input[type=range]{width:100%;accent-color:var(--acc)}
@media (max-width:640px){.wrap{grid-template-columns:1fr}.stage{height:360px}}
</style></head><body>
<div class="wrap"><div class="stage"><canvas id="c" aria-label="Model 3D spektrofotometer"></canvas>
<div class="tools"><button data-z="0.85" title="Perbesar">＋</button><button data-z="1.18" title="Perkecil">－</button><button id="rs" title="Tampilan awal">⟲</button></div>
<div class="views"><button data-v="iso">Isometrik</button><button data-v="depan">Depan</button><button data-v="samping">Samping</button><button data-v="belakang">Belakang</button><button data-v="atas">Atas</button><button data-v="kompartemen">Kompartemen</button></div></div>
<div class="side"><div class="card"><h4>Tampilan</h4><div class="tg">
<button data-t="lid" aria-pressed="false">Buka tutup</button><button data-t="xray" aria-pressed="false">Tembus pandang</button><button data-t="expl" aria-pressed="false">Terurai</button><button data-t="beam" aria-pressed="false">Jalur cahaya</button><button data-t="labels" aria-pressed="false">Label</button><button data-t="auto" aria-pressed="false">Putar otomatis</button></div></div>
<div class="card"><h4>λ = <span id="lv"></span> nm</h4><input type="range" id="lam" min="190" max="1100" step="1"><div class="hint" id="lamp"></div></div>
<div class="card"><h4 id="ti">Komponen</h4><div class="hint" id="info">Seret untuk memutar, roda/cubit untuk zoom, klik komponen untuk penjelasan.</div></div>
<div class="hint">Model skematis yang disederhanakan; bukan gambar teknik pabrikan.</div></div></div>
<script>
/* =====================================================================
   SPECTRA LAB — model3d.js
   Model 3D interaktif spektrofotometer double-beam bergaya UV-1700.
   Perender kanvas 2D (tanpa WebGL/pustaka): proyeksi perspektif,
   pencahayaan Lambert, pengurutan kedalaman (painter) dengan
   subdivisi bidang, orbit/zoom (tetikus & sentuh), animasi tutup,
   tampilan terurai, mode tembus, jalur cahaya, label, klik komponen.
   Dimensi dalam cm (perkiraan proporsi, bukan gambar teknik pabrikan).
   ===================================================================== */
(function (root) {
  'use strict';
  const SL = root.SL || (root.SL = {});
  const D2R = Math.PI / 180;

  function lamColor(l) {
    if (l < 380) return '#a99cf0'; if (l > 780) return '#b0302a';
    let r = 0, g = 0, b = 0;
    if (l < 440) { r = -(l - 440) / 60; b = 1; } else if (l < 490) { g = (l - 440) / 50; b = 1; }
    else if (l < 510) { g = 1; b = -(l - 510) / 20; } else if (l < 580) { r = (l - 510) / 70; g = 1; }
    else if (l < 645) { r = 1; g = -(l - 645) / 65; } else r = 1;
    const f = l < 420 ? 0.45 + 0.55 * (l - 380) / 40 : l > 700 ? 0.45 + 0.55 * (780 - l) / 80 : 1;
    const c = v => Math.round(255 * Math.pow(v * f, 0.8)).toString(16).padStart(2, '0');
    return '#' + c(r) + c(g) + c(b);
  }
  function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const n = parseInt(c, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function shade(col, k, emi) { const [r, g, b] = hex(col); const f = v => Math.max(0, Math.min(255, Math.round(v * k + (emi || 0) * 255))); return 'rgb(' + f(r) + ',' + f(g) + ',' + f(b) + ')'; }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function rotX(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const y = p[1] - piv[1], z = p[2] - piv[2]; return [p[0], piv[1] + y * c - z * s, piv[2] + y * s + z * c]; }
  function rotY(p, piv, ang) { const s = Math.sin(ang), c = Math.cos(ang); const x = p[0] - piv[0], z = p[2] - piv[2]; return [piv[0] + x * c + z * s, p[1], piv[2] - x * s + z * c]; }

  /* Informasi komponen (untuk klik) */
  const INFO = {
    base: ['Rangka dasar', 'Alas logam/plastik berkaki karet peredam getaran. Getaran meja dapat menggeser optik dan menambah derau.'],
    body: ['Rumah instrumen', 'Melindungi optik dari cahaya luar dan debu. Kompartemen harus selalu tertutup saat pengukuran agar cahaya ruangan tidak terbaca detektor.'],
    cover: ['Penutup atas', 'Panel atas yang menutupi bangku optik. Pada tampilan terurai, panel ini diangkat untuk memperlihatkan komponen di dalamnya.'],
    lid: ['Tutup kompartemen sampel', 'Dibuka untuk memasukkan kuvet. Tutup kembali sebelum membaca: cahaya ruangan yang masuk menurunkan absorbansi terbaca.'],
    compartment: ['Kompartemen sampel', 'Ruang berisi dua dudukan kuvet: sel referensi (belakang) dan sel sampel (depan). Kedua berkas melintas dari kanan ke kiri.'],
    holderR: ['Dudukan sel referensi', 'Tempat kuvet blanko. Intensitas yang melewatinya menjadi I₀.'],
    holderS: ['Dudukan sel sampel', 'Tempat kuvet sampel. Tanda panah pada dudukan menunjukkan arah berkas; sisi bening kuvet harus menghadap berkas.'],
    cuvR: ['Kuvet blanko', 'Berisi pelarut atau blanko reagen.'],
    cuvS: ['Kuvet sampel', 'Berisi larutan yang diukur. Warna pada model mengikuti larutan aktif.'],
    d2: ['Lampu deuterium (D₂)', 'Sumber kontinu daerah UV ±190–350 nm. Menyala terang pada model bila λ di bawah titik pergantian lampu.'],
    wi: ['Lampu wolfram-iodin (WI)', 'Sumber kontinu daerah tampak–NIR ±340–1100 nm. Lampu inilah yang terlihat berpendar oranye dari kisi ventilasi samping alat.'],
    lamphouse: ['Rumah lampu', 'Ruang lampu berventilasi. Label "HOT SURFACE" di sisi alat mengingatkan agar lampu didinginkan sebelum diganti.'],
    mirror: ['Cermin pemilih sumber', 'Berputar memilih berkas dari D₂ atau WI sesuai panjang gelombang.'],
    slit1: ['Celah masuk', 'Membatasi berkas yang masuk ke monokromator; lebar celah menentukan lebar pita spektral (±1 nm).'],
    mono: ['Monokromator', 'Ruang tertutup berisi kisi difraksi. Dinding gelap menekan sinar sesat.'],
    grating: ['Kisi difraksi holografik', 'Mendispersikan cahaya; sudut kisi menentukan λ yang keluar melalui celah keluar.'],
    slit2: ['Celah keluar', 'Meneruskan pita sempit di sekitar λ terpilih menuju sistem berkas ganda.'],
    split: ['Pembagi berkas', 'Membagi cahaya monokromatis menjadi berkas referensi dan berkas sampel.'],
    mirror2: ['Cermin pengarah', 'Membelokkan berkas sampel agar sejajar melewati sel sampel.'],
    detR: ['Detektor referensi', 'Fotodioda silikon yang mengukur I₀ dari sel blanko.'],
    detS: ['Detektor sampel', 'Fotodioda silikon yang mengukur I dari sel sampel. A = log(I₀/I).'],
    pcb: ['Papan elektronik', 'Penguat, ADC, dan prosesor yang menghitung %T dan absorbansi.'],
    lcd: ['Layar LCD', 'Menampilkan λ, data (ABS/%T/konsentrasi), dan lampu aktif (D2/WI).'],
    keypad: ['Papan tombol', 'Tombol MODE, GOTO WL, AUTO ZERO, START/STOP, dan angka untuk mengatur pengukuran tanpa komputer.'],
    vent: ['Kisi ventilasi', 'Membuang panas lampu dan catu daya. Jangan ditutup benda lain.'],
    card: ['Slot IC card', 'Menyimpan metode dan data pada kartu memori.'],
    fan: ['Kipas & catu daya', 'Pendingin dan sumber daya listrik 230 V AC.'],
    plate: ['Pelat identitas', 'Area label model instrumen.']
  };

  const COMPART = new Set(['compartment', 'holderR', 'holderS', 'cuvR', 'cuvS']);
  class Model3D {
    constructor(canvas, opt) {
      this.cv = canvas; this.ctx = canvas.getContext('2d'); this.opt = opt || {};
      this.yaw = -32; this.pitch = 24; this.dist = 84; this.target = [0, 9, 0];
      this.lid = 0; this.lidT = 0; this.expl = 0; this.explT = 0; this.xray = 0; this.xrayT = 0;
      this.beam = true; this.labels = false; this.auto = false; this.sel = null;
      this.lam = 550; this.lampSwitch = 340.8; this.abs = 0; this.status = 'READY'; this.solColor = '#8e2c8f';
      this.dark = false; this._build(); this._bind(); this._loop = this._loop.bind(this);
      this._ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => this.draw()) : null;
      if (this._ro) this._ro.observe(canvas.parentElement || canvas);
      this.draw();
    }
    /* ---------- geometri ---------- */
    _build() {
      const P = this.parts = [];
      const box = (id, grp, x0, x1, y0, y1, z0, z1, col, o) => {
        o = o || {};
        const faces = [
          ['f', [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]],
          ['b', [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]],
          ['r', [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]],
          ['l', [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]],
          ['t', [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]],
          ['u', [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]]
        ];
        const cell = o.cell || 4.5;
        for (const [k, q] of faces) {
          if (o.skip && o.skip.includes(k)) continue;
          const fc = (o.faceCol && o.faceCol[k]) || col;
          const lu = Math.hypot(...sub(q[1], q[0])), lv = Math.hypot(...sub(q[3], q[0]));
          const isText = o.text && k === o.text.face;
          const nu = isText ? 1 : Math.max(1, Math.ceil(lu / cell)), nv = isText ? 1 : Math.max(1, Math.ceil(lv / cell));
          for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
            const L = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
            const bl = (u, v) => L(L(q[0], q[1], u), L(q[3], q[2], u), v);
            P.push({ id, grp, col: fc, alpha: o.alpha == null ? 1 : o.alpha, emi: o.emi || 0, bias: o.bias || 0, tf: o.tf || null, key: k,
              v: [bl(i / nu, j / nv), bl((i + 1) / nu, j / nv), bl((i + 1) / nu, (j + 1) / nv), bl(i / nu, (j + 1) / nv)],
              text: (o.text && k === o.text.face && nu === 1 && nv === 1) ? o.text : null, dyn: o.dyn || null });
          }
        }
      };
      this.box = box;
      const BEIGE = '#e8d698', BEIGE2 = '#efe3b4', GREY = '#5d646e', DARK = '#23272e', CREAM = '#f2ecd6';
      // dasar & kaki
      box('base', 'body', -23, 23, 0, 3, -24.5, 24.5, GREY, { cell: 6 });
      [[-19, 20], [19, 20], [-19, -20], [19, -20]].forEach(([x, z]) => box('base', 'body', x - 1.6, x + 1.6, -1.4, 0, z - 1.6, z + 1.6, '#3d434b', { skip: ['u'] }));
      // dinding rumah (tipis, agar isi terlihat saat penutup diangkat)
      const wallO = { cell: 3 };
      box('body', 'walls', -23, 23, 3, 13.5, 23.9, 24.5, BEIGE, wallO);           // depan
      box('body', 'walls', -23, 23, 3, 13.5, -24.5, -23.9, BEIGE, wallO);         // belakang
      box('body', 'walls', -23, -22.4, 3, 13.5, -23.9, 23.9, BEIGE, wallO);       // kiri
      box('body', 'walls', 22.4, 23, 3, 13.5, -23.9, 23.9, BEIGE, wallO);         // kanan
      // panel depan kompartemen & label
      box('compartment', 'walls', -16, -1, 4, 12.5, 24.5, 24.8, '#dcc681', { bias: -4, skip: ['b'], cell: 99 });
      box('plate', 'walls', 9, 19, 7, 10, 24.5, 24.7, '#f7f4ea', { bias: -4.5, skip: ['b'], cell: 99, text: { face: 'f', w: 100, h: 30, lines: [['UV-VIS', 15, 20, 'center']], color: '#2b3a55', font: '600 15px "IBM Plex Sans Condensed", Arial' } });
      // sisi kanan: ventilasi lampu + slot kartu
      for (let i = 0; i < 7; i++) box('vent', 'walls', 23, 23.25, 6 + i * 1, 6.5 + i * 1, -20, -11, '#bfa868', { bias: -4, cell: 99 });
      box('card', 'walls', 23, 23.2, 11, 11.8, 2, 10, '#2b2f36', { bias: -4, cell: 99 });
      // bangku optik
      box('pcb', 'inner', -21.8, -12, 3, 3.6, -22, -6, '#1d6b45');
      [[-20, -19], [-17, -15], [-14, -10]].forEach(([a, b], k) => box('pcb', 'inner', a, b, 3.6, 4.4, -20 + k * 4, -18 + k * 4, '#15181d'));
      box('fan', 'inner', -21.6, -15, 3, 9, -23.6, -21.8, '#3a4048');
      // rumah lampu
      box('lamphouse', 'inner', 14, 22.2, 3, 4, -23.4, -9, '#3a4048');
      box('lamphouse', 'inner', 21.6, 22.2, 4, 12, -23.4, -9, '#4a515b', { alpha: 0.9 });
      box('d2', 'inner', 18, 20.2, 4, 11.5, -21, -18.6, '#d9d4ff', { dyn: 'd2' });
      box('wi', 'inner', 18, 20.2, 4, 10.5, -14.2, -11.8, '#fff1b8', { dyn: 'wi' });
      box('mirror', 'inner', 15, 15.6, 7.5, 11.5, -18, -14, '#cfd8e6', { dyn: 'mirror' });
      box('slit1', 'inner', 11, 11.4, 6, 13, -18.5, -16.4, DARK); box('slit1', 'inner', 11, 11.4, 6, 13, -15.6, -13.5, DARK);
      // monokromator (dinding semi-transparan)
      box('mono', 'inner', -2, 10.5, 3, 4, -23.2, -9, '#2d323a');
      box('mono', 'inner', -2, 10.5, 4, 12.5, -23.2, -22.7, '#3a4048', { alpha: 0.55 });
      box('mono', 'inner', -2, -1.5, 4, 12.5, -22.7, -9, '#3a4048', { alpha: 0.55 });
      box('grating', 'inner', 1.5, 5.5, 5.5, 12.5, -21.8, -20.6, '#9fb3cf', { faceCol: { f: '#c9d6ea' } });
      box('slit2', 'inner', 1.5, 2.6, 6, 13, -9.4, -9, DARK); box('slit2', 'inner', 3.4, 4.5, 6, 13, -9.4, -9, DARK);
      box('split', 'inner', 2.4, 3.6, 6.5, 12, 1.4, 4.6, '#7fe6f7', { alpha: 0.7, tf: [['y', [3, 0, 3], 45]] });
      box('mirror2', 'inner', 2.4, 3.6, 6.5, 12, 11.4, 14.6, '#cfd8e6', { tf: [['y', [3, 0, 13], -45]] });
      // kompartemen sampel: dinding gelap + lantai
      box('compartment', 'inner', -15, -2, 3, 6, -2, 20, '#2a2e35', { cell: 4 });
      box('compartment', 'inner', -15, -14.5, 6, 13.5, -2, 20, '#2f343c', { alpha: 0.85 });
      box('compartment', 'inner', -2.5, -2, 6, 13.5, -2, 20, '#2f343c', { alpha: 0.85 });
      box('compartment', 'inner', -14.5, -2.5, 6, 13.5, -2, -1.5, '#2f343c');
      box('compartment', 'inner', -14.5, -2.5, 6, 13.5, 19.5, 20, '#2f343c');
      // dudukan & kuvet (referensi belakang z=3, sampel depan z=13)
      const holder = (id, z) => {
        box(id, 'inner', -10.6, -6.4, 6, 12.5, z - 2.1, z - 1.6, DARK); box(id, 'inner', -10.6, -6.4, 6, 12.5, z + 1.6, z + 2.1, DARK);
        box(id, 'inner', -10.6, -10.1, 6, 12.5, z - 1.6, z + 1.6, DARK, { alpha: 0.9 });
        box(id, 'inner', -11.5, -5.5, 6, 6.6, z - 2.6, z + 2.6, '#1b1e23');
      };
      holder('holderR', 3); holder('holderS', 13);
      box('cuvR', 'inner', -9.8, -7.2, 6.6, 14.2, 1.7, 4.3, '#d9f3f7', { alpha: 0.35, bias: -0.4 });
      box('cuvS', 'inner', -9.8, -7.2, 6.6, 14.2, 11.7, 14.3, '#8e2c8f', { alpha: 0.55, bias: -0.4, dyn: 'cuvS' });
      box('detR', 'inner', -21.6, -19.6, 8, 11.5, 1.6, 4.4, '#2a3346'); box('detS', 'inner', -21.6, -19.6, 8, 11.5, 11.6, 14.4, '#2a3346');
      // penutup atas (berlubang di kompartemen) — grup 'cover'
      const covO = { cell: 4.5 };
      box('cover', 'cover', -23, -15, 13.5, 15, -24.5, 24.5, BEIGE2, covO);
      box('cover', 'cover', -2, 23, 13.5, 15, -24.5, 24.5, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, -24.5, -2, BEIGE2, covO);
      box('cover', 'cover', -15, -2, 13.5, 15, 20, 24.5, BEIGE2, covO);
      for (let i = 0; i < 12; i++) box('vent', 'cover', 8 + i * 1.2, 8.6 + i * 1.2, 15, 15.25, -23.5, -15.5, '#c9b371');
      // tutup kompartemen — grup 'lid' (berengsel di z=-2, y=15)
      box('lid', 'lid', -15.2, -1.8, 15, 16.6, -2.2, 20.2, '#e2cf8e', { cell: 4, faceCol: { u: '#1c1f24' } });
      // papan tombol & layar — di atas penutup
      box('keypad', 'cover', 2, 21, 15, 16.2, 13, 23.5, CREAM, { bias: -0.6 });
      const keys = [];
      for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) keys.push([3.6 + c * 3.4, 14.4 + r * 2.2]);
      keys.forEach(([x, z], k) => box('keypad', 'cover', x, x + 2, 16.2, 16.6, z, z + 1.3, k % 5 === 4 ? '#7da1d3' : '#c7cdd6', { bias: -0.8 }));
      // tiang & layar miring
      box('lcd', 'cover', 9, 14, 15, 17.5, 7.5, 10, '#ddd5b9', { bias: -0.6 });
      const tilt = [['x', [11.5, 16.5, 9], -18]];
      box('lcd', 'cover', 3.5, 19.5, 16.5, 28.5, 8.2, 10.2, '#ece5cd', { tf: tilt, bias: -0.7, cell: 99 });
      box('lcd', 'cover', 5, 18, 18, 27, 10.2, 10.35, '#b9efec', { tf: tilt, bias: -4, emi: 0.06, dyn: 'lcd', text: { face: 'f', w: 130, h: 90, lines: [] } });
    }
    /* ---------- kontrol publik ---------- */
    setLam(l) { this.lam = l; this.draw(); }
    setReadout(abs, status) { this.abs = abs; if (status) this.status = status; this.draw(); }
    setSolutionColor(c) { this.solColor = c || '#8e2c8f'; this.draw(); }
    setLampSwitch(v) { this.lampSwitch = v; this.draw(); }
    toggle(what, on) {
      if (what === 'lid') this.lidT = on == null ? (this.lidT ? 0 : 1) : (on ? 1 : 0);
      if (what === 'expl') this.explT = on == null ? (this.explT ? 0 : 1) : (on ? 1 : 0);
      if (what === 'xray') this.xrayT = on == null ? (this.xrayT ? 0 : 1) : (on ? 1 : 0);
      if (what === 'beam') this.beam = on == null ? !this.beam : !!on;
      if (what === 'labels') this.labels = on == null ? !this.labels : !!on;
      if (what === 'auto') this.auto = on == null ? !this.auto : !!on;
      this._kick(); return this.state();
    }
    state() { return { lid: this.lidT, expl: this.explT, xray: this.xrayT, beam: this.beam, labels: this.labels, auto: this.auto }; }
    view(name) {
      const V = { depan: [0, 8], samping: [90, 10], atas: [0, 88], belakang: [180, 14], iso: [-32, 24], kompartemen: [-8, 55] };
      const v = V[name] || V.iso; this._anim = { from: [this.yaw, this.pitch], to: v, t: 0 }; this._kick();
    }
    zoom(f) { this.dist = Math.max(55, Math.min(260, this.dist * f)); this.draw(); }
    reset() { this.dist = 84; this.view('iso'); this.sel = null; if (this.opt.onPick) this.opt.onPick(null); }
    destroy() { this._dead = true; if (this._ro) this._ro.disconnect(); }
    _kick() { if (!this._running) { this._running = true; this._last = performance.now(); requestAnimationFrame(this._loop); } }
    _loop(now) {
      if (this._dead) return;
      const dt = Math.min(0.05, (now - this._last) / 1000); this._last = now;
      const ease = (cur, tgt, sp) => { const d = tgt - cur; return Math.abs(d) < 0.002 ? tgt : cur + d * Math.min(1, dt * sp); };
      this.lid = ease(this.lid, this.lidT, 5); this.expl = ease(this.expl, this.explT, 4); this.xray = ease(this.xray, this.xrayT, 5);
      if (this._anim) {
        const a = this._anim; a.t = Math.min(1, a.t + dt * 2.2); const k = a.t < 0.5 ? 2 * a.t * a.t : 1 - Math.pow(-2 * a.t + 2, 2) / 2;
        let dy = a.to[0] - a.from[0]; while (dy > 180) dy -= 360; while (dy < -180) dy += 360;
        this.yaw = a.from[0] + dy * k; this.pitch = a.from[1] + (a.to[1] - a.from[1]) * k; if (a.t >= 1) this._anim = null;
      }
      if (this.auto && !this._drag) this.yaw += dt * 18;
      this.draw();
      const busy = this.lid !== this.lidT || this.expl !== this.explT || this.xray !== this.xrayT || this._anim || (this.auto && !this._drag);
      if (busy) requestAnimationFrame(this._loop); else this._running = false;
    }
    /* ---------- transformasi ---------- */
    _xf(p, part) {
      let q = p;
      if (part.tf) for (const [ax, piv, deg] of part.tf) q = ax === 'x' ? rotX(q, piv, deg * D2R) : rotY(q, piv, deg * D2R);
      if (part.grp === 'lid') { q = rotX(q, [0, 15, -2.2], -102 * this.lid * D2R); q = [q[0], q[1] + 16 * this.expl, q[2]]; }
      if (part.grp === 'cover') q = [q[0], q[1] + 16 * this.expl, q[2]];
      return q;
    }
    _cam(p) {
      const t = this.target; let x = p[0] - t[0], y = p[1] - t[1] - 7 * this.expl, z = p[2] - t[2];
      const cy = Math.cos(-this.yaw * D2R), sy = Math.sin(-this.yaw * D2R);
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const cp = Math.cos(this.pitch * D2R), sp = Math.sin(this.pitch * D2R);
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      return [x1, y2, z2];
    }
    _proj(c) { const zc = this.dist * (1 + 0.28 * this.expl) - c[2]; const f = this._f; return [this._cx + f * c[0] / zc, this._cy - f * c[1] / zc, zc]; }
    /* ---------- gambar ---------- */
    draw() {
      if (this._dead) return;
      const cv = this.cv, ctx = this.ctx;
      const rect = cv.getBoundingClientRect(); const W = Math.max(240, rect.width || cv.width), H = Math.max(200, rect.height || cv.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._cx = W / 2; this._cy = H / 2 + 10; this._f = Math.min(W, H * 1.45) * 1.0;
      // latar
      const dark = this.dark;
      const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, dark ? '#0d1d42' : '#eef3fa'); g.addColorStop(1, dark ? '#06102a' : '#dfe8f4');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      // bayangan lantai
      const sh = [[-25, 0, -26.5], [25, 0, -26.5], [25, 0, 26.5], [-25, 0, 26.5]].map(p => this._proj(this._cam([p[0] + 1.5, -1.4, p[2] + 1.5])));
      ctx.fillStyle = dark ? 'rgba(0,0,0,.35)' : 'rgba(11,27,58,.10)'; ctx.beginPath(); sh.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
      const light = norm([0.45, 0.85, 0.55]);
      const camPos = [0, 0, this.dist];
      const lampUV = this.lam < this.lampSwitch;
      const items = [];
      const coverA = 1 - 0.82 * this.xray, wallA = 1 - 0.75 * this.xray;
      const closed = this.xray < 0.04 && this.expl < 0.04;
      for (const part of this.parts) {
        const wv = part.v.map(p => this._xf(p, part));
        const cvs = wv.map(p => this._cam(p));
        const n = norm(cross(sub(cvs[1], cvs[0]), sub(cvs[3], cvs[0])));
        const cen = [(cvs[0][0] + cvs[2][0]) / 2, (cvs[0][1] + cvs[2][1]) / 2, (cvs[0][2] + cvs[2][2]) / 2];
        let alpha = part.alpha;
        if (part.grp === 'cover' || part.grp === 'lid') alpha *= coverA;
        if (part.grp === 'walls' || part.id === 'body') alpha *= wallA;
        if (alpha < 0.02) continue;
        if (part.grp === 'inner' && closed && !(this.lid > 0.05 && COMPART.has(part.id))) continue;
        if (part.grp === 'inner') alpha = Math.min(alpha, 1);
        const facing = dot(n, sub(camPos, cen)) > 0;
        if (!facing && alpha > 0.97) continue;
        // pencahayaan di ruang dunia
        const wn = norm(cross(sub(wv[1], wv[0]), sub(wv[3], wv[0])));
        let k = 0.5 + 0.55 * Math.max(0, dot(wn, light));
        let col = part.col, emi = part.emi;
        if (part.dyn === 'd2') { emi = lampUV ? 0.45 : 0.0; col = lampUV ? '#e4dcff' : '#9a96b5'; }
        if (part.dyn === 'wi') { emi = !lampUV ? 0.45 : 0.0; col = !lampUV ? '#ffe7a0' : '#a89c74'; }
        if (part.dyn === 'cuvS') col = this.solColor;
        if (this.sel && part.id === this.sel) { emi += 0.18; }
        items.push({ part, pts: cvs.map(c => this._proj(c)), depth: this.dist - cen[2] + part.bias + (part.grp === 'inner' ? 0.35 : 0) + (facing ? 0 : 0.02), fill: shade(col, k, emi), alpha, facing });
      }
      // jalur cahaya
      if (this.beam) {
        const y = 9.5, by = this.lid > 0.5 || this.xray > 0.5 || this.expl > 0.5;
        const src = lampUV ? [19.1, y, -19.8] : [19.1, y, -13];
        const path = [[src, [15.3, y, -16]], [[15.3, y, -16], [11.2, y, -15]], [[11.2, y, -15], [3.5, y, -21]], [[3.5, y, -21], [3, y, -9.2]], [[3, y, -9.2], [3, y, 3]], [[3, y, 3], [3, y, 13]], [[3, y, 3], [-8.5, y, 3]], [[-8.5, y, 3], [-19.6, y, 3]], [[3, y, 13], [-8.5, y, 13]], [[-8.5, y, 13], [-19.6, y, 13]]];
        const segs = [];
        path.forEach(([a, b], si) => { for (let k = 0; k < 4; k++) { const t0 = k / 4, t1 = (k + 1) / 4; segs.push([[a[0] + (b[0] - a[0]) * t0, a[1], a[2] + (b[2] - a[2]) * t0], [a[0] + (b[0] - a[0]) * t1, a[1], a[2] + (b[2] - a[2]) * t1], si]); } });
        const poly = si => si <= 2; // sebelum kisi: cahaya putih (polikromatis)
        for (const [a, b, si] of segs) {
          const ca = this._cam(a), cb = this._cam(b); const pa = this._proj(ca), pb = this._proj(cb);
          const atten = (si === 9) ? Math.max(0.12, Math.pow(10, -Math.max(0, Math.min(3, this.abs)))) : 1;
          if (closed && !(this.lid > 0.05 && (si >= 6))) continue;
          items.push({ beam: true, pa, pb, depth: this.dist - (ca[2] + cb[2]) / 2 - 0.3, color: poly(si) ? '#fffbe8' : lamColor(this.lam), uv: !poly(si) && this.lam < 380, atten, show: by });
        }
      }
      items.sort((a, b) => b.depth - a.depth);
      this._drawn = [];
      ctx.lineJoin = 'round';
      for (const it of items) {
        if (it.beam) {
          ctx.save(); ctx.globalAlpha = 0.28 * it.atten * (it.show ? 1 : 0.55); ctx.strokeStyle = it.color; ctx.lineWidth = 7; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke();
          ctx.globalAlpha = 0.95 * it.atten; ctx.lineWidth = 2.2; if (it.uv) ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(it.pa[0], it.pa[1]); ctx.lineTo(it.pb[0], it.pb[1]); ctx.stroke(); ctx.restore();
          continue;
        }
        const p = it.pts; ctx.globalAlpha = it.alpha;
        ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]); ctx.closePath();
        ctx.fillStyle = it.fill; ctx.fill();
        if (it.alpha > 0.97) { ctx.strokeStyle = it.fill; ctx.lineWidth = 1.1; ctx.stroke(); } // menutup celah antarsubdivisi
        if (this.sel && it.part.id === this.sel) { ctx.strokeStyle = '#35d3ee'; ctx.lineWidth = 1.2; ctx.stroke(); }
        ctx.globalAlpha = 1;
        if (it.part.text && it.facing) this._text(it.part, p);
        if (it.facing && it.alpha > 0.3) this._drawn.push({ id: it.part.id, p });
      }
      if (this.labels) this._labels(ctx);
    }
    _text(part, p) {
      const ctx = this.ctx, t = part.text;
      // p: [kiri-bawah, kanan-bawah, kanan-atas, kiri-atas]
      const TL = p[3], TR = p[2], BL = p[0];
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0); const dpr = this.cv.width / Math.max(1, this.cv.getBoundingClientRect().width || this.cv.width);
      ctx.setTransform((TR[0] - TL[0]) / t.w * dpr, (TR[1] - TL[1]) / t.w * dpr, (BL[0] - TL[0]) / t.h * dpr, (BL[1] - TL[1]) / t.h * dpr, TL[0] * dpr, TL[1] * dpr);
      if (part.dyn === 'lcd') {
        ctx.fillStyle = '#e98a9a'; [14, 40, 66].forEach(yy => ctx.fillRect(8, yy - 9, 22, 10));
        ctx.fillStyle = '#0b3a48'; ctx.font = '500 16px "IBM Plex Mono", monospace'; ctx.textAlign = 'right';
        ctx.fillText(this.lam.toFixed(1) + 'nm', 124, 20);
        ctx.fillText((this.status === 'OFF' ? '-----' : (this.abs > 3 ? 'OVER' : this.abs.toFixed(3))) + ' ABS', 124, 46);
        ctx.font = '500 11px "IBM Plex Mono", monospace'; ctx.textAlign = 'left'; ctx.fillText(this.lam < this.lampSwitch ? 'D2' : 'WI', 38, 70);
        ctx.textAlign = 'right'; ctx.fillStyle = '#4f8a92'; ctx.fillText(this.status, 124, 70);
      } else {
        ctx.fillStyle = t.color || '#222'; ctx.font = t.font || '12px Arial';
        for (const [s, x, y, al] of t.lines) { ctx.textAlign = al || 'left'; ctx.fillText(s, x === 'c' ? t.w / 2 : (al === 'center' ? t.w / 2 : x), y); }
      }
      ctx.restore();
    }
    _labels(ctx) {
      const A = [['d2', [19, 12, -20]], ['wi', [19, 11, -13]], ['mirror', [15.3, 12, -16]], ['slit1', [11.2, 13.5, -15]], ['grating', [3.5, 13, -21]], ['slit2', [3, 13.5, -9.2]], ['split', [3, 12.5, 3]], ['holderR', [-8.5, 13, 3]], ['holderS', [-8.5, 13, 13]], ['detS', [-20.6, 12, 13]], ['detR', [-20.6, 12, 3]], ['pcb', [-17, 5, -14]], ['lcd', [11.5, 29, 6]], ['keypad', [12, 17, 22]], ['lid', [-8.5, 17, 9]]];
      ctx.font = '600 11px "IBM Plex Sans", Arial';
      const placed = [];
      const L = A.map(([id, p0]) => { const part = this.parts.find(q => q.id === id); const p = part ? this._xf(p0, part) : p0; return { id, s: this._proj(this._cam(p)) }; }).sort((a, b) => a.s[1] - b.s[1]);
      for (const { id, s } of L) {
        const name = INFO[id][0].replace(/ \(.*\)/, '');
        const w = ctx.measureText(name).width + 10; const x = s[0]; let y = s[1] - 16;
        for (let k = 0; k < 8 && placed.some(r => Math.abs(r.x - x) < (r.w + w) / 2 + 2 && Math.abs(r.y - y) < 17); k++) y -= 17;
        placed.push({ x, y, w });
        ctx.strokeStyle = this.dark ? 'rgba(220,235,255,.6)' : 'rgba(11,27,58,.55)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(x, y + 4); ctx.stroke();
        ctx.fillStyle = id === this.sel ? '#1f6fff' : (this.dark ? 'rgba(13,29,66,.92)' : 'rgba(255,255,255,.94)');
        ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - w / 2, y - 11, w, 16, 4) : ctx.rect(x - w / 2, y - 11, w, 16); ctx.fill();
        ctx.fillStyle = id === this.sel ? '#fff' : (this.dark ? '#e6edfb' : '#0b1b3a'); ctx.textAlign = 'center'; ctx.fillText(name, x, y + 1);
        ctx.fillStyle = '#35d3ee'; ctx.beginPath(); ctx.arc(s[0], s[1], 2.5, 0, 7); ctx.fill();
      }
    }
    pickAt(x, y) {
      const inPoly = (p, px, py) => { let c = false; for (let i = 0, j = 3; i < 4; j = i++) { if (((p[i][1] > py) !== (p[j][1] > py)) && (px < (p[j][0] - p[i][0]) * (py - p[i][1]) / (p[j][1] - p[i][1]) + p[i][0])) c = !c; } return c; };
      for (let i = this._drawn.length - 1; i >= 0; i--) if (inPoly(this._drawn[i].p, x, y)) return this._drawn[i].id;
      return null;
    }
    _bind() {
      const cv = this.cv; cv.style.touchAction = 'none';
      let pts = new Map(), start = null, pinch = null;
      cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); start = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch, moved: false }; this._drag = true; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: this.dist }; } });
      cv.addEventListener('pointermove', e => {
        if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pts.size === 2 && pinch) { const [a, b] = [...pts.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); this.dist = Math.max(55, Math.min(260, pinch.dist * pinch.d / Math.max(10, d))); start.moved = true; this.draw(); return; }
        if (!start) return; const dx = e.clientX - start.x, dy = e.clientY - start.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) start.moved = true;
        this.yaw = start.yaw - dx * 0.45; this.pitch = Math.max(-10, Math.min(89, start.pitch + dy * 0.35)); this._anim = null; this.draw();
      });
      const up = e => {
        if (!pts.has(e.pointerId)) return; pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
        if (pts.size === 0) {
          this._drag = false;
          if (start && !start.moved) { const r = cv.getBoundingClientRect(); const id = this.pickAt(e.clientX - r.left, e.clientY - r.top); this.sel = id; this.draw(); if (this.opt.onPick) this.opt.onPick(id ? { id, title: INFO[id][0], text: INFO[id][1] } : null); }
          start = null; if (this.auto) this._kick();
        }
      };
      cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
      cv.addEventListener('wheel', e => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1.08 : 1 / 1.08); }, { passive: false });
      cv.tabIndex = 0;
      cv.addEventListener('keydown', e => {
        const k = e.key; if (k === 'ArrowLeft') this.yaw -= 6; else if (k === 'ArrowRight') this.yaw += 6; else if (k === 'ArrowUp') this.pitch = Math.min(89, this.pitch + 5); else if (k === 'ArrowDown') this.pitch = Math.max(-10, this.pitch - 5); else if (k === '+' || k === '=') this.zoom(0.9); else if (k === '-') this.zoom(1.1); else return;
        e.preventDefault(); this.draw();
      });
    }
  }
  SL.Model3D = Model3D; SL.MODEL_INFO = INFO; SL.lamColor = lamColor;
})(typeof window !== 'undefined' ? window : globalThis);

</script>
<script>
var INIT = /*__INIT__*/{"lam":550,"abs":0,"status":"READY","solColor":"#8e2c8f","lampSwitch":340.8,"xray":true};
(function(){
  var SL = window.SL, $ = function(s){return document.querySelector(s)};
  var m = new SL.Model3D($('#c'), { onPick: function(i){ $('#ti').textContent = i ? i.title : 'Komponen'; $('#info').textContent = i ? i.text : 'Klik komponen untuk penjelasan.'; } });
  m.lampSwitch = INIT.lampSwitch; m.abs = INIT.abs; m.status = INIT.status; m.solColor = INIT.solColor; m.dark = false;
  function sync(){ var st = m.state(); document.querySelectorAll('[data-t]').forEach(function(b){ b.setAttribute('aria-pressed', String(!!st[b.dataset.t])); }); }
  function setLam(l){ m.setLam(l); $('#lv').textContent = l; $('#lam').value = l; $('#lamp').textContent = (l < m.lampSwitch ? 'Lampu D2 aktif' : 'Lampu WI aktif') + ' · ' + (l < 400 ? 'UV' : l <= 780 ? 'sinar tampak' : 'NIR'); }
  m.toggle('beam', true); if (INIT.xray) m.toggle('xray', true); if (INIT.lid) m.toggle('lid', true);
  setLam(INIT.lam); sync();
  document.addEventListener('click', function(e){ var b = e.target.closest('button'); if(!b) return;
    if (b.dataset.t){ m.toggle(b.dataset.t); sync(); }
    if (b.dataset.v){ m.view(b.dataset.v); document.querySelectorAll('.views button').forEach(function(x){ x.classList.toggle('on', x===b); }); }
    if (b.dataset.z) m.zoom(+b.dataset.z); if (b.id==='rs') m.reset(); });
  $('#lam').addEventListener('input', function(e){ setLam(+e.target.value); });
})();
</script></body></html>
'''

# =============================================================================
# BAGIAN 5 — APLIKASI STREAMLIT
# =============================================================================

import io
import json
import math
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="SPECTRA LAB", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Gaya
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif !important; letter-spacing: .01em; }
section[data-testid="stSidebar"] { background: #0b1b3a; }
section[data-testid="stSidebar"] * { color: #dbe6fb; }
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0b1b3a; }
.sl-hero { background: linear-gradient(135deg,#0b1b3a,#13295a); color:#e8f0ff; border-radius:14px; padding:22px 24px; }
.sl-hero h1 { color:#fff; margin:4px 0 8px; letter-spacing:.06em; }
.sl-eyebrow { font:500 .72rem 'IBM Plex Mono', monospace; letter-spacing:.12em; color:#63dcf2; text-transform:uppercase; }
.sl-chip { display:inline-block; font:500 .72rem 'IBM Plex Mono', monospace; padding:3px 9px; border-radius:99px; margin:2px 4px 2px 0; background:#e3edff; color:#1f4fbf; }
.sl-chip.sim { background:#fdf0d9; color:#a86400; } .sl-chip.ok { background:#dcf3e9; color:#12875a; } .sl-chip.bad { background:#fbe3e2; color:#c3312f; }
.sl-lcd { background:#b9efec; color:#0b3a48; border-radius:6px; padding:10px 16px; font-family:'IBM Plex Mono', monospace;
  box-shadow: inset 0 0 0 3px #d9d2b8, inset 0 0 0 5px #8d8a7c, inset 0 2px 14px rgba(0,0,0,.18); }
.sl-lcd .row { display:flex; justify-content:space-between; align-items:baseline; }
.sl-lcd .tag { background:#e98a9a; color:#fff; font-size:.66rem; padding:0 6px; border-radius:2px; }
.sl-lcd .big { font-size:1.5rem; letter-spacing:.08em; }
.sl-lcd .foot { border-top:1px dashed rgba(11,58,72,.3); margin-top:4px; padding-top:3px; font-size:.72rem; color:#4f8a92; display:flex; justify-content:space-between; }
.sl-eq { font:500 1.05rem 'IBM Plex Mono', monospace; background:#edf2f9; border-radius:8px; padding:10px 12px; }
.sl-banner { background:#fdf0d9; border:1px solid #f0c98a; border-radius:8px; padding:9px 12px; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)

PAL = ["#1f6fff", "#0aa5c4", "#c2368a", "#d98a00", "#1a9a5f", "#7a4dd8"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _defaults() -> dict:
    return {
        "settings": {"dec": "koma", "mode": "latihan",
                     "crit": {"r2_min": 0.995, "rsd_max": 2.0, "rec_min": 90.0, "rec_max": 110.0, "std_res_max": 2.0, "a_min": 0.1, "a_max": 1.0}},
        "inst": {"power": "off", "baseline": None, "lamp_switch": 340.8, "cuvette": "kuarsa", "path": 1.0, "blank": "pelarut",
                 "analyte": "kmno4", "solution": "standar", "conc": 30.0, "custom": 25.0, "matrix": 0.0, "start": 400.0, "end": 700.0,
                 "interval": 1.0, "speed": "sedang", "lam": 550.0, "autozero": [], "last_A": 0.0},
        "seed": int(np.random.default_rng().integers(1, 1_000_000)), "counter": 1,
        "spectra": [], "last_scan": None,
        "cal_base": pd.DataFrame({"C": pd.Series(dtype=float), "A": pd.Series(dtype=float), "Pakai": pd.Series(dtype=bool)}), "cal_ver": 0,
        "cal_src": "pengguna", "cal_name": "", "cal_unit": "mg/L", "cal_lambda": None, "cal_zero": False, "cal_analyte": None,
        "samples": [],
        "valid": {"reps": "", "kind": "konsentrasi", "level": "", "level_unit": "mg/L", "rec_mode": "spike", "cs": "", "css": "", "cadd": "", "cm": "", "cr": ""},
        "photo": [], "kin": {"A0": 1.2, "Ainf": 0.02, "k": 0.012, "dur": 300.0, "dt": 5.0, "noise": 0.003, "lam": 590.0, "data": [], "source": "simulasi", "n_init": 6, "ainf_fit": ""},
        "report": {"judul": "Penetapan Kadar dengan Spektrofotometri UV-Vis", "praktikan": "", "nim": "", "kelas": "", "dosen": "", "sampel": "", "tanggal": date.today().isoformat(), "catatan": "", "pdf": None, "generated": False},
        "quiz": {}, "history": [], "log": [],
    }


if "S" not in st.session_state:
    st.session_state.S = _defaults()
S = st.session_state.S
I = S["inst"]
CR = S["settings"]["crit"]


def f(x, d=4):
    return R.fnum(x, d, S["settings"]["dec"])


def g(x, s=4):
    return R.fsig(x, s, S["settings"]["dec"])


def rng():
    S["counter"] += 1
    return np.random.default_rng(S["seed"] * 7919 + S["counter"])


def log(msg):
    S["log"].insert(0, f"{datetime.now():%H.%M.%S} {msg}"); S["log"] = S["log"][:60]


def num_in(text, default=float("nan")):
    v = C.parse_number(text)
    return default if math.isnan(v) else v


def an():
    return C.ANALYTES[I["analyte"]]


def sol_conc():
    if I["solution"] == "blanko":
        return 0.0
    if I["solution"] == "sampelX":
        return C.hidden_conc(I["analyte"], S["seed"])
    return I["custom"] if I["solution"] == "bebas" else I["conc"]


def sol_label():
    a = an()
    if I["solution"] == "blanko":
        return f"Blanko ({a['pelarut']})"
    if I["solution"] == "sampelX":
        return f"{a['rumus']} Sampel X"
    c = sol_conc()
    return f"{a['rumus']} {f(c, 2 if c < 1 else 1)} mg/L"


def baseline_valid(lo, hi):
    b = I["baseline"]
    return bool(b) and b["cuvette"] == I["cuvette"] and b["blank"] == I["blank"] and b["analyte"] == I["analyte"] and min(lo, hi) >= b["start"] and max(lo, hi) <= b["end"]


def az_at(lam):
    return any(abs(z["lam"] - lam) < 0.05 and z["cuvette"] == I["cuvette"] and z["blank"] == I["blank"] for z in I["autozero"])


def read_at(lam, conc=None):
    corrected = baseline_valid(lam, lam) or az_at(lam)
    m = C.measure(lam, analyte_id=I["analyte"], conc=sol_conc() if conc is None else conc, path=I["path"], cuvette=I["cuvette"], blank=I["blank"],
                  baseline_done=corrected, speed="lambat", lamp_switch=I["lamp_switch"], matrix=0 if I["solution"] == "blanko" else I["matrix"], rng=rng())
    return float(m["A"][0]), corrected, bool(m["over"][0])


def cal_points() -> pd.DataFrame:
    df = S.get("cal_cur", S["cal_base"]).copy()
    if df.empty:
        return df
    df = df[pd.to_numeric(df["C"], errors="coerce").notna() & pd.to_numeric(df["A"], errors="coerce").notna()]
    return df[df["Pakai"].fillna(True).astype(bool)]


def cal_reg():
    p = cal_points()
    if len(p) < 2:
        return None
    return C.linreg(p["C"].astype(float).values, p["A"].astype(float).values, S["cal_zero"])


def set_calibration(df, src, name=None, unit=None, lam=None, analyte=None):
    S["cal_base"] = df.reset_index(drop=True); S["cal_ver"] += 1; S["cal_src"] = src; S.pop("cal_cur", None)
    if name is not None: S["cal_name"] = name
    if unit is not None: S["cal_unit"] = unit
    if lam is not None: S["cal_lambda"] = lam
    if analyte is not None: S["cal_analyte"] = analyte


def df_same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Bandingkan isi tabel tanpa terpengaruh perbedaan tipe data/indeks."""
    if a.shape != b.shape:
        return False
    norm = lambda d: d.reset_index(drop=True).astype(object).where(d.reset_index(drop=True).notna(), "").astype(str).values
    return bool((norm(a) == norm(b)).all())


def sample_fp(s):
    if s["dil_mode"] == "manual":
        return s["fp_manual"] if s["fp_manual"] > 0 else float("nan")
    return C.dilution_factor([(r["V pipet (mL)"], r["V labu (mL)"]) for r in s["steps"]])


def sample_res(s, reg):
    try:
        reps = C.parse_list(s["reps"])
    except ValueError:
        return None
    if not reg or not reg.ok or not reps:
        return None
    return C.sample_result(reg, reps, fp=sample_fp(s), mass_g=num_in(s["mass"], None) if s["mass"] else None,
                           vol_ml=num_in(s["vol"], None) if s["vol"] else None, unit=S["cal_unit"],
                           a_min=CR["a_min"], a_max=CR["a_max"], rsd_max=CR["rsd_max"])


def new_sample(name, reps="", src="pengguna", hidden=None):
    S["counter"] += 1
    return {"id": S["counter"], "name": name, "reps": reps, "src": src, "dil_mode": "langkah", "steps": [], "fp_manual": 1.0, "mass": "", "vol": "", "hidden": hidden}


def lcd(lam=None, A=None, mode="SPEKTRUM"):
    lam = I["lam"] if lam is None else lam
    A = I["last_A"] if A is None else A
    on = I["power"] == "ready"
    a_txt = ("OVER" if A > 3 else f"{A:.3f}") if on else "-----"
    t_txt = f"{float(C.abs_to_t(A)):.1f}" if on else "----"
    lamp = "D2" if lam < I["lamp_switch"] else "WI"
    st.markdown(f"""<div class="sl-lcd"><div class="row"><span class="tag">λ</span><span class="big">{lam:.1f} <small>nm</small></span></div>
    <div class="row"><span class="tag">Data</span><span class="big">{a_txt} <small>ABS</small></span></div>
    <div class="row"><span class="tag">%T</span><span>{t_txt} %</span></div>
    <div class="foot"><span>{lamp} · {'READY' if on else I['power'].upper()}</span><span>{mode}</span></div></div>""", unsafe_allow_html=True)


def fig_base(xt, yt, height=380):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=20, t=30, b=50),
                      font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#0b1b3a"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified",
                      separators=",." if S["settings"]["dec"] == "koma" else ".,")
    fig.update_xaxes(title_text=xt, gridcolor="#e5ecf5", zeroline=False)
    fig.update_yaxes(title_text=yt, gridcolor="#e5ecf5", zeroline=False)
    return fig


def src_chip(src):
    lab = {"simulasi": "DATA SIMULASI", "contoh": "DATA CONTOH", "pengguna": "INPUT PENGGUNA", "impor": "IMPOR CSV", "campuran": "CAMPURAN"}.get(src, src)
    cls = "sim" if src in ("simulasi", "contoh") else "ok"
    return f'<span class="sl-chip {cls}">{lab}</span>'


def model3d(height=560, lid=False, xray=True):
    html = MODEL3D_HTML
    init = {"lam": float(I["lam"]), "abs": float(I["last_A"]) if I["power"] == "ready" else 0.0,
            "status": "READY" if I["power"] == "ready" else "OFF", "solColor": "#d9f3f7" if I["solution"] == "blanko" else an()["warna"],
            "lampSwitch": float(I["lamp_switch"]), "xray": xray, "lid": lid}
    html = html.replace('/*__INIT__*/{"lam":550,"abs":0,"status":"READY","solColor":"#8e2c8f","lampSwitch":340.8,"xray":true}', json.dumps(init))
    components.html(html, height=height, scrolling=False)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Dasbor", "Model 3D Instrumen", "Simulasi Instrumen", "Fotometri", "Kinetika", "Analisis Spektrum",
         "Kurva Kalibrasi", "Validasi Data", "Pembelajaran & Kuis", "Panduan Praktikum", "Laporan", "Riwayat"]
with st.sidebar:
    st.markdown("### SPECTRA LAB\n<span style='color:#8ea3c9;font-size:.8rem'>Smart UV-Vis Analytical System</span>", unsafe_allow_html=True)
    page = st.radio("Menu", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    mode = st.radio("Mode kerja", ["latihan", "analisis"], index=0 if S["settings"]["mode"] == "latihan" else 1,
                    format_func=lambda m: "Latihan (data contoh)" if m == "latihan" else "Analisis (data saya)", horizontal=False)
    S["settings"]["mode"] = mode
    dec = st.radio("Pemisah desimal", ["koma", "titik"], index=0 if S["settings"]["dec"] == "koma" else 1,
                   format_func=lambda d: "Koma (0,512)" if d == "koma" else "Titik (0.512)", horizontal=True)
    S["settings"]["dec"] = dec
    st.caption("Politeknik AKA Bogor · Analisis Kimia. Instrumen acuan: Shimadzu UV-1700. "
               "Aplikasi ini simulator edukatif dan tidak terhubung ke alat.")

status = [f'<span class="sl-chip sim">SIMULASI</span>',
          f'<span class="sl-chip {"ok" if I["power"] == "ready" else ""}">{"Instrumen siap" if I["power"] == "ready" else "Instrumen mati"}</span>',
          f'<span class="sl-chip {"ok" if I["baseline"] else "sim"}">{"Baseline " + str(int(I["baseline"]["start"])) + "–" + str(int(I["baseline"]["end"])) + " nm" if I["baseline"] else "Baseline belum"}</span>']
st.markdown(" ".join(status), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
def page_dashboard():
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        st.markdown("""<div class="sl-hero"><div class="sl-eyebrow">Politeknik AKA Bogor · Analisis Spektrofotometri</div>
        <h1>SPECTRA LAB</h1><div>Smart UV-Vis Analytical System: simulasi pengukuran, analisis spektrum, kurva kalibrasi
        Lambert–Beer, evaluasi mutu data, dan laporan otomatis dalam satu ruang kerja.</div><br>
        <span class="sl-chip sim">MODE SIMULASI</span><span class="sl-chip">Double-beam · 190–1100 nm</span></div>""", unsafe_allow_html=True)
        st.write("")
        reg = cal_reg()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spektrum", len(S["spectra"]))
        m2.metric("Titik kalibrasi", len(cal_points()))
        m3.metric("R²", f(reg.r2, 4) if reg and reg.ok else "—")
        m4.metric("λ ukur (nm)", f(S["cal_lambda"], 1) if S["cal_lambda"] else "—")
        st.info("Mulai dari **Panduan Praktikum** untuk alur langkah demi langkah, atau buka **Simulasi Instrumen**.")
    with c2:
        model3d(height=420, xray=False)
        st.caption("Model 3D skematis yang meniru tata letak unit UV-1700 di laboratorium; bukan gambar resmi pabrikan.")
    if S["spectra"]:
        sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), S["spectra"][-1])
        fig = fig_base("Panjang gelombang (nm)", "Absorbansi", 300)
        fig.add_trace(go.Scatter(x=sp["lam"], y=sp["A"], mode="lines", name=sp["name"], line=dict(color=PAL[0], width=2), fill="tozeroy", fillcolor="rgba(31,111,255,.08)"))
        st.subheader("Spektrum terakhir"); st.plotly_chart(fig, use_container_width=True)


def page_model3d():
    st.markdown("Seret untuk memutar, roda/cubit untuk zoom, klik komponen untuk penjelasan. Gunakan **Buka tutup**, "
                "**Tembus pandang**, **Terurai**, dan **Jalur cahaya** di panel kanan. Warna berkas mengikuti λ instrumen virtual.")
    model3d(height=620, xray=True)
    with st.expander("Fungsi komponen utama"):
        for k, v in COMPONENTS.items():
            st.markdown(f"**{k}.** {v}")


def page_instrument():
    st.markdown('<div class="sl-banner"><b>Simulasi, bukan pengukuran nyata.</b> Spektrum dihasilkan model (pita Gauss + derau fotometrik + sinar sesat) '
                'dengan absorptivitas mendekati literatur. Masukkan data alat yang sebenarnya pada Kurva Kalibrasi atau impor CSV.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**1 · Daya & inisialisasi**")
            if I["power"] != "ready":
                if st.button("Nyalakan instrumen", type="primary", use_container_width=True):
                    with st.status("Inisialisasi…", expanded=True) as stt:
                        for chk in ["Memori & ROM", "Filter & celah", "Lampu D2 menyala", "Lampu WI menyala", "Titik nol λ (656,1 nm)", "Energi berkas"]:
                            st.write("✓ " + chk); time.sleep(0.25)
                        stt.update(label="Inisialisasi selesai", state="complete")
                    I["power"] = "ready"; log("Inisialisasi selesai. Alat nyata: panaskan lampu 15–30 menit."); st.rerun()
            else:
                st.success("Lampu D₂ & WI menyala.")
                if st.button("Matikan", use_container_width=True):
                    I.update(power="off", baseline=None, autozero=[]); log("Instrumen dimatikan."); st.rerun()
        with st.container(border=True):
            st.markdown("**2 · Larutan & kuvet**")
            keys = list(C.ANALYTES)
            new_an = st.selectbox("Analit (model simulasi)", keys, index=keys.index(I["analyte"]), format_func=lambda k: f"{C.ANALYTES[k]['nama']} · λref {C.ANALYTES[k]['lam_ref']} nm")
            if new_an != I["analyte"]:
                a = C.ANALYTES[new_an]; I.update(analyte=new_an, conc=float(a["std"][len(a["std"]) // 2]), lam=float(a["lam_ref"]))
                I["start"], I["end"] = (200.0, 400.0) if a["lam_ref"] < 320 else (float(max(190, round((a["lam_ref"] - 150) / 10) * 10)), float(min(1100, round((a["lam_ref"] + 170) / 10) * 10)))
                st.rerun()
            sols = ["blanko", "standar", "sampelX", "bebas"]
            I["solution"] = st.selectbox("Isi sel sampel", sols, index=sols.index(I["solution"]), format_func=lambda s: {"blanko": "Blanko", "standar": "Larutan standar", "sampelX": "Sampel X (tak diketahui)", "bebas": "Konsentrasi bebas"}[s])
            if I["solution"] == "standar":
                stds = [float(x) for x in an()["std"]]
                I["conc"] = st.selectbox("Konsentrasi standar (mg/L)", stds, index=stds.index(I["conc"]) if I["conc"] in stds else len(stds) // 2)
            elif I["solution"] == "bebas":
                I["custom"] = st.number_input("Konsentrasi (mg/L)", min_value=0.0, value=float(I["custom"]), step=0.1)
            bl = ["pelarut", "reagen", "udara"]
            I["blank"] = st.selectbox("Sel referensi (blanko)", bl, index=bl.index(I["blank"]), format_func=lambda b: {"pelarut": "Pelarut", "reagen": "Blanko reagen", "udara": "Udara"}[b])
            cv = list(C.CUVETTES)
            I["cuvette"] = st.selectbox("Kuvet", cv, index=cv.index(I["cuvette"]), format_func=lambda k: f"{C.CUVETTES[k]['nama']} (≥{C.CUVETTES[k]['cutoff']} nm)")
            I["path"] = st.select_slider("Tebal kuvet b (cm)", options=[0.5, 1.0, 2.0, 5.0], value=I["path"])
            I["matrix"] = st.slider("Kekeruhan / serapan matriks (A)", 0.0, 0.15, float(I["matrix"]), 0.005)
            st.caption(f"**{an()['rumus']}** dalam {an()['pelarut']}. {an()['catatan']}")
        with st.container(border=True):
            st.markdown("**3 · Parameter pemindaian**")
            c_a, c_b = st.columns(2)
            I["start"] = c_a.number_input("λ awal (nm)", 190.0, 1100.0, float(I["start"]), 10.0)
            I["end"] = c_b.number_input("λ akhir (nm)", 190.0, 1100.0, float(I["end"]), 10.0)
            iv = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            I["interval"] = c_a.selectbox("Interval (nm)", iv, index=iv.index(I["interval"]))
            sp = ["cepat", "sedang", "lambat"]
            I["speed"] = c_b.selectbox("Kecepatan", sp, index=sp.index(I["speed"]))
            I["lamp_switch"] = st.number_input("Pergantian lampu D2→WI (nm)", 295.0, 364.0, float(I["lamp_switch"]), 0.1)
            lo, hi = min(I["start"], I["end"]), max(I["start"], I["end"])
            st.caption(f"Jumlah titik: {int(round((hi - lo) / I['interval'])) + 1}")
            if lo < C.CUVETTES[I["cuvette"]]["cutoff"]:
                st.error(f"Kuvet {C.CUVETTES[I['cuvette']]['nama']} menyerap kuat di bawah ±{C.CUVETTES[I['cuvette']]['cutoff']} nm; gunakan kuarsa untuk UV.")
            if I["blank"] == "udara":
                st.warning("Referensi udara tidak mengoreksi serapan kuvet dan pelarut.")
            if an()["reagent"] and I["blank"] != "reagen":
                st.warning("Metode ini memakai reagen; gunakan blanko reagen agar serapan reagen terkoreksi.")
            if I["baseline"] and not baseline_valid(lo, hi):
                st.warning("Rentang atau konfigurasi berubah sejak koreksi baseline; ulangi baseline.")
    with right:
        a1, a2 = st.columns([1, 1.4])
        with a1:
            lcd(mode="SPEKTRUM")
            gw = st.number_input("GOTO WL (nm)", 190.0, 1100.0, float(I["lam"]), 0.5)
            k1, k2 = st.columns(2)
            if k1.button("GOTO WL", use_container_width=True):
                I["lam"] = gw
                if I["power"] == "ready":
                    I["last_A"] = read_at(gw)[0]
                st.rerun()
            if k2.button("AUTO ZERO", use_container_width=True, disabled=I["power"] != "ready"):
                I["autozero"] = [{"lam": I["lam"], "cuvette": I["cuvette"], "blank": I["blank"]}]; I["last_A"] = 0.0
                log(f"AUTO ZERO pada {f(I['lam'], 1)} nm."); st.rerun()
        with a2:
            model3d(height=300, lid=True, xray=False)
        b1, b2, b3 = st.columns(3)
        ready = I["power"] == "ready"
        if b1.button("Koreksi baseline", use_container_width=True, disabled=not ready):
            bar = st.progress(0.0, "Baseline: blanko di kedua sel…")
            for k in range(11):
                bar.progress(k / 10); time.sleep(0.07)
            I["baseline"] = {"start": lo, "end": hi, "cuvette": I["cuvette"], "blank": I["blank"], "analyte": I["analyte"], "time": f"{datetime.now():%H.%M}"}
            I["autozero"] = []; log(f"Baseline {lo:.0f}–{hi:.0f} nm tersimpan."); st.rerun()
        scan = b2.button("Mulai scan", type="primary", use_container_width=True, disabled=not ready)
        if not ready:
            st.caption("Nyalakan instrumen untuk mengaktifkan tombol.")
        chart = st.empty()
        if scan:
            ok = baseline_valid(lo, hi)
            sim = C.simulate_scan(lo, hi, I["interval"], analyte_id=I["analyte"], conc=sol_conc(), path=I["path"], cuvette=I["cuvette"],
                                  blank=I["blank"], baseline_done=ok, speed=I["speed"], lamp_switch=I["lamp_switch"],
                                  matrix=0 if I["solution"] == "blanko" else I["matrix"], rng=rng())
            n = len(sim["lam"])
            for k in range(1, 13):
                m = max(2, int(n * k / 12))
                fig = fig_base("Panjang gelombang λ (nm)", "Absorbansi (A)")
                fig.add_trace(go.Scatter(x=sim["lam"][:m], y=sim["A"][:m], mode="lines", name=sol_label(), line=dict(color=PAL[0], width=2)))
                fig.update_xaxes(range=[lo, hi]); chart.plotly_chart(fig, use_container_width=True); time.sleep(0.08)
            S["counter"] += 1
            spec = {"id": S["counter"], "name": sol_label(), "source": "simulasi", "lam": sim["lam"].tolist(), "A": sim["A"].round(5).tolist(),
                    "time": f"{datetime.now():%d/%m %H.%M}", "params": {"analyte": I["analyte"], "baseline": ok, "interval": I["interval"], "speed": I["speed"], "start": lo, "end": hi},
                    "flags": {k: sim[k] for k in ("over", "low_energy", "uv_cutoff")}}
            S["spectra"].append(spec); S["last_scan"] = spec["id"]; S["spectra"] = S["spectra"][-40:]
            log(f"Scan {hi:.0f}→{lo:.0f} nm: {sol_label()}{'' if ok else ' (tanpa baseline berlaku)'}."); st.rerun()
        sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), None)
        if sp:
            peaks = C.find_peaks(sp["lam"], sp["A"])
            fig = fig_base("Panjang gelombang λ (nm)", "Absorbansi (A)")
            fig.add_trace(go.Scatter(x=sp["lam"], y=sp["A"], mode="lines", name=sp["name"], line=dict(color=PAL[0], width=2), fill="tozeroy", fillcolor="rgba(31,111,255,.08)"))
            for i, p in enumerate(peaks[:3]):
                fig.add_vline(x=p["lambda"], line_dash="dash", line_color=PAL[1] if i == 0 else PAL[3],
                              annotation_text=("λmaks " if i == 0 else "") + f(p["lambda"], 1) + " nm", annotation_position="top")
            chart.plotly_chart(fig, use_container_width=True)
            st.markdown(src_chip("simulasi"), unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                if peaks:
                    p0 = peaks[0]
                    st.markdown(f"**λmaks: {f(p0['lambda'], 1)} nm** · A = {f(p0['abs'], 4)} · %T = {f(float(C.abs_to_t(p0['abs'])), 2)} % · FWHM {f(p0['fwhm'], 1)} nm")
                    if st.button(f"Gunakan {f(p0['lambda'], 1)} nm untuk kuantitatif"):
                        I["lam"] = round(p0["lambda"], 1); S["cal_lambda"] = I["lam"]; S["cal_analyte"] = sp["params"]["analyte"]
                        S["cal_name"] = S["cal_name"] or C.ANALYTES[sp["params"]["analyte"]]["nama"]; st.toast(f"λ kerja = {f(I['lam'], 1)} nm"); st.rerun()
                    st.caption("Ukur pada λmaks: kepekaan terbesar dan dA/dλ ≈ 0 sehingga pergeseran λ kecil hampir tidak mengubah A.")
                else:
                    st.info("Tidak ada puncak yang memenuhi prominensi 0,01 A.")
                if not sp["params"]["baseline"]:
                    st.warning("Diambil tanpa koreksi baseline yang berlaku (offset ±0,01–0,02 A).")
                if sp["flags"]["over"]:
                    st.error("Sebagian titik di luar rentang fotometrik −0,5…3,0 A.")
                if sp["flags"]["low_energy"] or sp["flags"]["uv_cutoff"]:
                    st.warning("Energi sangat rendah di sebagian rentang; data di daerah itu tidak dapat dipercaya.")
            with r2:
                lam_r = st.number_input("Baca pada λ (nm)", 190.0, 1100.0, float(round(peaks[0]["lambda"], 1)) if peaks else 500.0, 0.5)
                a_r = C.interp_y(sp["lam"], sp["A"], lam_r)
                st.markdown(f"A = **{f(a_r, 4)}** · %T = **{f(float(C.abs_to_t(a_r)), 2)} %** (interpolasi kurva)")
        with st.container(border=True):
            st.markdown("**Kuantitatif pada λ kerja = " + f(I["lam"], 1) + " nm**")
            q1, q2 = st.columns(2)
            if q1.button("Ukur deret standar → Kurva Kalibrasi", disabled=not ready, use_container_width=True):
                a = an(); r = rng(); rows = []
                for c in a["std"]:
                    A, corr, _ = read_at(I["lam"], conc=C.prep_conc(c, r))
                    rows.append({"C": float(c), "A": round(A, 4), "Pakai": True})
                set_calibration(pd.DataFrame(rows), "simulasi", name=a["nama"], unit="mg/L", lam=I["lam"], analyte=I["analyte"])
                log(f"Deret standar diukur pada {f(I['lam'], 1)} nm" + ("" if corr else " TANPA auto zero/baseline") + ".")
                st.toast("Deret standar terkirim ke Kurva Kalibrasi")
            if q2.button("Ukur Sampel X (triplo)", disabled=not ready, use_container_width=True):
                hx = C.hidden_conc(I["analyte"], S["seed"])
                reps = [read_at(I["lam"], conc=hx)[0] for _ in range(3)]
                S["samples"].append(new_sample(f"Sampel X {an()['rumus']}", "; ".join(f(v, 4) for v in reps), "simulasi", I["analyte"]))
                log("Sampel X diukur triplo."); st.toast("Sampel X ditambahkan ke Kurva Kalibrasi")
        with st.expander("Log instrumen"):
            st.code("\n".join(S["log"]) or "Belum ada aktivitas.", language=None)


def page_spectrum():
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        up = st.file_uploader("Impor CSV (kolom 1 = λ nm, berikutnya = A)", type=["csv", "txt", "tsv"])
        if up is not None and st.button("Tambahkan ke pustaka"):
            res = C.parse_csv(up.getvalue().decode("utf-8", errors="replace"))
            if not res["ok"]:
                st.error(res["reason"])
            else:
                for c in range(1, len(res["cols"])):
                    lam, A = res["cols"][0], res["cols"][c]; m = ~(np.isnan(lam) | np.isnan(A))
                    if m.sum() < 5:
                        continue
                    o = np.argsort(lam[m]); S["counter"] += 1
                    nm = res["header"][c] if res["header"] and c < len(res["header"]) else f"{up.name} #{c}"
                    S["spectra"].append({"id": S["counter"], "name": nm, "source": "impor", "lam": lam[m][o].tolist(), "A": A[m][o].tolist(), "time": f"{datetime.now():%d/%m %H.%M}", "params": {"baseline": True}, "flags": {}})
                st.toast("Spektrum diimpor"); st.rerun()
        if S["settings"]["mode"] == "latihan" and st.button("Muat 3 spektrum contoh KMnO₄"):
            for k, c in enumerate([10, 30, 50]):
                sim = C.simulate_scan(400, 700, 1, analyte_id="kmno4", conc=c, rng=np.random.default_rng(100 + k)); S["counter"] += 1
                S["spectra"].append({"id": S["counter"], "name": f"Contoh KMnO₄ {c} mg/L", "source": "contoh", "lam": sim["lam"][::-1].tolist(), "A": sim["A"][::-1].tolist(), "time": "", "params": {"analyte": "kmno4", "baseline": True}, "flags": {}})
            st.rerun()
        if not S["spectra"]:
            st.info("Pustaka kosong. Pindai di Simulasi Instrumen atau impor CSV.")
            return
        names = {s["id"]: f"{s['name']} [{s['source']}]" for s in S["spectra"]}
        act = st.multiselect("Spektrum ditampilkan", list(names), default=[S["spectra"][-1]["id"]], format_func=names.get)
        ymode = st.radio("Sumbu Y", ["Absorbansi", "%T"], horizontal=True)
        proc = st.radio("Pengolahan", ["Asli", "Dihaluskan (SG-7)", "Normalisasi", "Turunan-1"], horizontal=True)
        prom = st.number_input("Prominensi min (A)", 0.0, 1.0, 0.01, 0.005, format="%.3f")
        minh = st.number_input("Tinggi min (A)", 0.0, 3.0, 0.02, 0.01, format="%.3f")
        if st.button("Hapus spektrum terpilih", disabled=not act):
            S["spectra"] = [s for s in S["spectra"] if s["id"] not in act]; st.rerun()
    with right:
        sel = [s for s in S["spectra"] if s["id"] in act]
        fig = fig_base("Panjang gelombang λ (nm)", {"Turunan-1": "dA/dλ (nm⁻¹)", "Normalisasi": "A / Amaks"}.get(proc, "Transmitansi (%T)" if ymode == "%T" else "Absorbansi (A)"), 440)
        for i, s in enumerate(sel):
            x = np.asarray(s["lam"]); y = np.asarray(s["A"])
            if proc.startswith("Dihaluskan"): y = C.smooth(y, 7)
            if proc == "Normalisasi": y = y / np.nanmax(y)
            if proc == "Turunan-1": y = np.gradient(C.smooth(np.asarray(s["A"]), 9), x)
            elif ymode == "%T" and proc != "Normalisasi": y = C.abs_to_t(y)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s["name"], line=dict(color=PAL[i % len(PAL)], width=2)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Seret untuk memperbesar, klik dua kali untuk kembali; arahkan kursor untuk membaca nilai.")
        if sel:
            lam_r = st.number_input("Baca nilai pada λ (nm)", 190.0, 1100.0, 525.0, 0.5)
            st.dataframe(pd.DataFrame([{"Spektrum": s["name"], "A": C.interp_y(s["lam"], s["A"], lam_r), "%T": float(C.abs_to_t(C.interp_y(s["lam"], s["A"], lam_r)))} for s in sel]), hide_index=True, use_container_width=True)
            tgt = sel[-1]; pk = C.find_peaks(tgt["lam"], tgt["A"], prom, minh)
            st.markdown(f"**Puncak terdeteksi — {tgt['name']}**")
            st.dataframe(pd.DataFrame([{"λ (nm)": round(p["lambda"], 1), "A": round(p["abs"], 4), "%T": round(float(C.abs_to_t(p["abs"])), 2), "Prominensi": round(p["prominence"], 4), "FWHM (nm)": round(p["fwhm"], 1)} for p in pk[:8]]), hide_index=True, use_container_width=True)
            same = len({len(s["lam"]) for s in sel}) == 1 and all(np.allclose(s["lam"], sel[0]["lam"]) for s in sel)
            if same:
                df = pd.DataFrame({"Panjang gelombang (nm)": sel[0]["lam"], **{s["name"] + " (A)": s["A"] for s in sel}})
            else:
                df = pd.concat([pd.DataFrame({"Spektrum": s["name"], "Panjang gelombang (nm)": s["lam"], "Absorbansi": s["A"]}) for s in sel])
            with st.expander("Tabel data"):
                st.dataframe(df, hide_index=True, use_container_width=True, height=300)
            st.download_button("Unduh CSV", df.to_csv(index=False, sep=";" if S["settings"]["dec"] == "koma" else ",", decimal="," if S["settings"]["dec"] == "koma" else "."),
                               "spektrum-spectralab.csv", "text/csv")


def page_calibration():
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])
    S["cal_name"] = c1.text_input("Analit / metode", S["cal_name"], placeholder="mis. Fe dengan 1,10-fenantrolin")
    units = ["mg/L", "µg/mL", "ppm", "µg/L", "mmol/L", "mol/L", "%b/v"]
    S["cal_unit"] = c2.selectbox("Satuan", units, index=units.index(S["cal_unit"]))
    lam_txt = c3.text_input("λ ukur (nm)", f(S["cal_lambda"], 1) if S["cal_lambda"] else "")
    S["cal_lambda"] = num_in(lam_txt, None) if lam_txt else None
    S["cal_zero"] = c4.checkbox("Paksa melalui nol", S["cal_zero"])
    st.caption("Model: absorbansi (y) sebagai fungsi konsentrasi (x), y = ax + b, kuadrat terkecil. Melalui nol hanya dibenarkan bila intersep tidak berbeda nyata dari nol.")
    L, Rr = st.columns([1, 1.6], gap="large")
    with L:
        st.markdown("**Larutan standar** " + src_chip(S["cal_src"]), unsafe_allow_html=True)
        edited = st.data_editor(S["cal_base"], key=f"cal_{S['cal_ver']}", num_rows="dynamic", hide_index=True, use_container_width=True,
                                column_config={"C": st.column_config.NumberColumn(f"C ({S['cal_unit']})", format="%.4f"),
                                               "A": st.column_config.NumberColumn("Absorbansi", format="%.4f"),
                                               "Pakai": st.column_config.CheckboxColumn("Pakai", default=True)})
        S["cal_cur"] = edited
        if not df_same(edited, S["cal_base"]):
            S["cal_base"] = edited.reset_index(drop=True).copy(); S["cal_ver"] += 1
            if S["cal_src"] in ("simulasi", "contoh"):
                S["cal_src"] = "campuran"
        paste = st.text_area("Tempel dua kolom dari Excel (C lalu A)", height=90, placeholder="0,5\t0,101\n1,0\t0,198")
        b1, b2, b3 = st.columns(3)
        if b1.button("Masukkan tempelan") and paste.strip():
            res = C.parse_csv(paste)
            if res["ok"]:
                add = pd.DataFrame({"C": res["cols"][0], "A": res["cols"][1], "Pakai": True}).dropna()
                set_calibration(pd.concat([cal_points(), add], ignore_index=True), "pengguna"); st.rerun()
            else:
                st.error(res["reason"])
        if S["settings"]["mode"] == "latihan" and b2.button("Data contoh"):
            set_calibration(pd.DataFrame({"C": [0.5, 1, 2, 3, 4, 5], "A": [0.102, 0.197, 0.401, 0.598, 0.792, 1.001], "Pakai": True}), "contoh",
                            name="Fe(II)–1,10-fenantrolin (data contoh)", unit="mg/L", lam=510.0, analyte="fephen")
            S["samples"] = [dict(new_sample("Air sumur A (contoh)", "0,452; 0,448; 0,455", "contoh"), steps=[{"V pipet (mL)": 10.0, "V labu (mL)": 50.0}])]
            st.rerun()
        if b3.button("Kosongkan"):
            set_calibration(_defaults()["cal_base"], "pengguna"); st.rerun()
        up = st.file_uploader("Impor CSV standar (C; A)", type=["csv", "txt"], key="calcsv")
        if up is not None and st.button("Impor"):
            res = C.parse_csv(up.getvalue().decode("utf-8", errors="replace"))
            if res["ok"]:
                set_calibration(pd.DataFrame({"C": res["cols"][0], "A": res["cols"][1], "Pakai": True}).dropna(), "impor"); st.rerun()
            else:
                st.error(res["reason"])
    reg = cal_reg()
    with Rr:
        pts = cal_points()
        fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Absorbansi (A)", 380)
        fig.update_layout(hovermode="closest")
        fig.add_trace(go.Scatter(x=pts["C"], y=pts["A"], mode="markers", name="Standar", marker=dict(color=PAL[0], size=10, line=dict(color="white", width=1.5))))
        if reg and reg.ok:
            xs = np.array([min(0, reg.xmin), reg.xmax * 1.05])
            fig.add_trace(go.Scatter(x=xs, y=reg.slope * xs + reg.intercept, mode="lines", name="Regresi", line=dict(color=PAL[1], width=2)))
            for s in S["samples"]:
                r = sample_res(s, reg)
                if r:
                    fig.add_trace(go.Scatter(x=[r["c_meas"]], y=[r["A_mean"]], mode="markers+text", name=s["name"], text=[s["name"]], textposition="top right",
                                             marker=dict(symbol="diamond", size=12, color=PAL[2])))
        st.plotly_chart(fig, use_container_width=True)
        if not reg or not reg.ok:
            st.info(reg.reason if reg else "Minimal 3 titik valid diperlukan untuk regresi.")
        else:
            calib_stats(reg)
    page_samples(reg)


def calib_stats(reg):
    if True:
        st.markdown(f'<div class="sl-eq">{reg.equation()}</div>', unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("R²", f(reg.r2, 5), "memenuhi" if reg.r2 >= CR["r2_min"] else "di bawah kriteria", delta_color="normal" if reg.r2 >= CR["r2_min"] else "inverse")
        m[1].metric("r", f(reg.r, 5)); m[2].metric("Kemiringan a", g(reg.slope), f"± {g(reg.ci_slope, 2)}", delta_color="off")
        m[3].metric("Intersep b", "0 (dipaksa)" if reg.through_zero else g(reg.intercept, 3), None if reg.through_zero else f"± {g(reg.ci_intercept, 2)}", delta_color="off")
        m = st.columns(4)
        m[0].metric("s y/x", g(reg.sxy, 3)); m[1].metric("LOD", f"{g(reg.lod, 3)} {S['cal_unit']}"); m[2].metric("LOQ", f"{g(reg.loq, 3)} {S['cal_unit']}")
        m[3].metric("Rentang", f"{g(reg.xmin, 3)}–{g(reg.xmax, 3)}")
        with st.container(border=True):
            st.markdown("**Interpretasi otomatis**")
            for p in R.calib_interpretation(reg, S["cal_unit"], CR["r2_min"], CR["std_res_max"], S["settings"]["dec"]):
                st.markdown("- " + p)


def page_samples(reg):
    st.subheader("Analisis kuantitatif sampel")
    t1, t2 = st.columns(2)
    if t1.button("+ Tambah sampel"):
        S["samples"].append(new_sample(f"Sampel {len(S['samples']) + 1}")); st.rerun()
    if t2.button("Ukur Sampel X (simulator)", disabled=I["power"] != "ready"):
        akey = S["cal_analyte"] or I["analyte"]; lam = S["cal_lambda"] or C.ANALYTES[akey]["lam_ref"]
        keep = I["analyte"]; I["analyte"] = akey
        reps = [read_at(lam, conc=C.hidden_conc(akey, S["seed"]))[0] for _ in range(3)]; I["analyte"] = keep
        S["samples"].append(new_sample(f"Sampel X {C.ANALYTES[akey]['rumus']}", "; ".join(f(v, 4) for v in reps), "simulasi", akey)); st.rerun()
    for idx, s in enumerate(list(S["samples"])):
        with st.expander(f"{s['name']}", expanded=True):
            a, b = st.columns([1, 1.1], gap="large")
            with a:
                s["name"] = st.text_input("Nama sampel", s["name"], key=f"n{s['id']}")
                s["reps"] = st.text_input("Absorbansi ulangan (pisahkan ;)", s["reps"], key=f"r{s['id']}", placeholder="0,412; 0,415; 0,409")
                s["dil_mode"] = st.radio("Pengenceran", ["langkah", "manual"], index=0 if s["dil_mode"] == "langkah" else 1, horizontal=True, key=f"dm{s['id']}",
                                         format_func=lambda x: "Langkah pengenceran" if x == "langkah" else "FP manual")
                if s["dil_mode"] == "manual":
                    s["fp_manual"] = st.number_input("Faktor pengenceran", min_value=0.0001, value=float(s["fp_manual"]), key=f"fp{s['id']}")
                else:
                    st.caption("Tiap baris: V pipet larutan dimasukkan ke labu V labu. FP = Π(V labu / V pipet).")
                    steps = st.data_editor(pd.DataFrame(s["steps"], columns=["V pipet (mL)", "V labu (mL)"]), num_rows="dynamic", hide_index=True, key=f"st{s['id']}", use_container_width=True)
                    s["steps"] = steps.dropna().to_dict("records")
                m1, m2 = st.columns(2)
                s["mass"] = m1.text_input("Massa sampel padat (g)", s["mass"], key=f"m{s['id']}")
                s["vol"] = m2.text_input("Volume larutan awal (mL)", s["vol"], key=f"v{s['id']}")
                if st.button("Hapus sampel", key=f"del{s['id']}"):
                    S["samples"].pop(idx); st.rerun()
            with b:
                st.markdown(src_chip(s["src"]), unsafe_allow_html=True)
                r = sample_res(s, reg)
                if not r:
                    st.info("Isi absorbansi (angka dipisah ;) dan pastikan regresi valid.")
                    continue
                u = S["cal_unit"]
                st.markdown(f"Ā (n = {r['n']}) = **{f(r['A_mean'], 4)}**" + (f" · RSD {f(r['rsd'], 2)} %" if r["n"] > 1 else ""))
                st.markdown(f"C terukur = **{g(r['c_meas'])} ± {g(r['ci'], 2)} {u}** · FP = {g(r['fp'])}")
                st.markdown(f"### C sampel asal = {g(r['c_final'])} ± {g(r['ci_final'], 2)} {u}")
                if r["mass_ok"]:
                    st.markdown(f"**Kadar dalam sampel padat = {g(r['kadar'])} mg/kg**")
                for fl in r["flags"]:
                    st.warning(fl)
                if not r["flags"]:
                    st.success("Absorbansi sampel di dalam rentang kalibrasi dan daerah kerja.")
                with st.popover("Langkah perhitungan") if hasattr(st, "popover") else st.expander("Langkah perhitungan"):
                    st.latex(r"x = \frac{\bar y - b}{a} = \frac{%s - (%s)}{%s} = %s" % (f(r["A_mean"], 4), g(reg.intercept, 4), g(reg.slope, 5), g(r["c_meas"], 5)))
                    st.latex(r"s_{x_0} = \frac{s_{y/x}}{a}\sqrt{\frac{1}{m}+\frac{1}{n}+\frac{(\bar y_0-\bar y)^2}{a^2\sum(x_i-\bar x)^2}} = %s" % g(r["sx0"], 3))
                    st.latex(r"C_{asal} = x \times FP = %s \times %s = %s\ \mathrm{%s}" % (g(r["c_meas"], 5), g(r["fp"], 4), g(r["c_final"], 5), u.replace("µ", r"\mu ")))
                if s.get("hidden") and S["settings"]["mode"] == "latihan" and st.button("Ungkap nilai sebenarnya (cek diri)", key=f"rv{s['id']}"):
                    tv = C.hidden_conc(s["hidden"], S["seed"])
                    st.info(f"Nilai model: {g(tv, 3)} mg/L · perolehan kembali {f(r['c_final'] / tv * 100, 1)} %")


def precision():
    v = S["valid"]
    try:
        vals = C.parse_list(v["reps"])
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    d = C.describe(vals); gr = C.grubbs(vals)
    hw = None
    if v["kind"] == "konsentrasi":
        lvl = num_in(v["level"], d["mean"]) if v["level"] else d["mean"]
        fac = {"mg/L": 1e-6, "µg/mL": 1e-6, "ppm": 1e-6, "µg/L": 1e-9, "%b/v": 1e-2}.get(v["level_unit"])
        if fac and lvl > 0:
            pr = C.horwitz(lvl * fac); hw = {"level": lvl, "prsd": pr, "ref": 0.67 * pr, "horrat": d["rsd"] / pr}
    return {"vals": vals, "d": d, "g": gr, "pass": d["rsd"] <= CR["rsd_max"], "hw": hw}


def recovery():
    v = S["valid"]
    if v["rec_mode"] == "spike":
        cs, css, cad = num_in(v["cs"]), num_in(v["css"]), num_in(v["cadd"])
        if any(math.isnan(x) for x in (cs, css, cad)) or cad <= 0:
            return None
        Rv = C.recovery_spike(cs, css, cad); return {"mode": "spike", "R": Rv, "pass": CR["rec_min"] <= Rv <= CR["rec_max"], "txt": f"({g(css)} − {g(cs)}) / {g(cad)} × 100"}
    cm, cr = num_in(v["cm"]), num_in(v["cr"])
    if math.isnan(cm) or not cr > 0:
        return None
    o = C.recovery_reference(cm, cr)
    return {"mode": "acuan", "R": o["R"], "bias": o["bias"], "rel": o["rel_bias"], "pass": CR["rec_min"] <= o["R"] <= CR["rec_max"], "txt": f"{g(cm)} / {g(cr)} × 100"}


def page_validation():
    with st.container(border=True):
        st.markdown("**Kriteria penerimaan (ditetapkan pengguna).** Tidak ada satu batas universal; sesuaikan dengan SOP, metode baku, atau instruksi dosen.")
        c = st.columns(7)
        CR["r2_min"] = c[0].number_input("R² min", 0.0, 1.0, CR["r2_min"], 0.001, format="%.3f")
        CR["rsd_max"] = c[1].number_input("RSD maks (%)", 0.0, 100.0, CR["rsd_max"], 0.5)
        CR["rec_min"] = c[2].number_input("Recovery min (%)", 0.0, 200.0, CR["rec_min"], 1.0)
        CR["rec_max"] = c[3].number_input("Recovery maks (%)", 0.0, 200.0, CR["rec_max"], 1.0)
        CR["std_res_max"] = c[4].number_input("|Residual terstandar| maks", 0.5, 5.0, CR["std_res_max"], 0.5)
        CR["a_min"] = c[5].number_input("A min kerja", 0.0, 3.0, CR["a_min"], 0.05)
        CR["a_max"] = c[6].number_input("A maks kerja", 0.0, 3.0, CR["a_max"], 0.05)
    v = S["valid"]
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.subheader("Presisi (ulangan)")
        v["kind"] = st.radio("Jenis data", ["konsentrasi", "absorbansi"], index=0 if v["kind"] == "konsentrasi" else 1, horizontal=True)
        v["reps"] = st.text_area("Nilai ulangan (pisahkan ;)", v["reps"], placeholder="2,251; 2,238; 2,264; 2,247; 2,255; 2,243")
        if v["kind"] == "konsentrasi":
            a, b = st.columns(2)
            v["level"] = a.text_input("Kadar acuan Horwitz (opsional)", v["level"], placeholder="= rata-rata")
            v["level_unit"] = b.selectbox("Satuan", ["mg/L", "µg/mL", "ppm", "µg/L", "%b/v"], index=["mg/L", "µg/mL", "ppm", "µg/L", "%b/v"].index(v["level_unit"]) if v["level_unit"] in ["mg/L", "µg/mL", "ppm", "µg/L", "%b/v"] else 0)
        pr = precision()
        if pr:
            d = pr["d"]
            st.dataframe(pd.DataFrame({"Parameter": ["n", "Rata-rata", "Simpangan baku s", "RSD (%)", "SK 95 % rata-rata", "Rentang"],
                                       "Nilai": [str(d["n"]), g(d["mean"], 5), g(d["sd"], 3), f(d["rsd"], 2), "± " + g(d["ci95"], 3), f"{g(d['min'])} – {g(d['max'])}"]}), hide_index=True, use_container_width=True)
            (st.success if pr["pass"] else st.error)(f"RSD {f(d['rsd'], 2)} % {'≤' if pr['pass'] else '>'} {f(CR['rsd_max'], 2)} %: keterulangan {'memenuhi' if pr['pass'] else 'belum memenuhi'} kriteria.")
            if pr["g"]["applicable"]:
                (st.warning if pr["g"]["outlier"] else st.info)(f"Uji Grubbs (α = 0,05): G = {f(pr['g']['G'], 3)} vs G kritis {f(pr['g']['Gcrit'], 3)} → " + (f"nilai {g(pr['g']['value'])} terindikasi pencilan." if pr["g"]["outlier"] else "tidak ada pencilan."))
            if pr["hw"]:
                h = pr["hw"]; st.caption(f"Horwitz pada {g(h['level'], 3)} {v['level_unit']}: PRSD_R = {f(h['prsd'], 2)} %; 0,67 × PRSD_R = {f(h['ref'], 2)} %; HorRat = {f(h['horrat'], 2)}. Pembanding, bukan pengganti kriteria SOP.")
        else:
            st.info("Masukkan minimal 2 nilai numerik.")
    with p2:
        st.subheader("Akurasi (% recovery)")
        v["rec_mode"] = st.radio("Metode", ["spike", "acuan"], index=0 if v["rec_mode"] == "spike" else 1, horizontal=True, format_func=lambda m: "Adisi standar (spike)" if m == "spike" else "Nilai acuan / CRM")
        if v["rec_mode"] == "spike":
            a, b, c3 = st.columns(3)
            v["cs"] = a.text_input("C sampel", v["cs"]); v["css"] = b.text_input("C sampel + spike", v["css"]); v["cadd"] = c3.text_input("C ditambahkan", v["cadd"])
        else:
            a, b = st.columns(2)
            v["cm"] = a.text_input("C terukur", v["cm"]); v["cr"] = b.text_input("Nilai acuan", v["cr"])
        rc = recovery()
        if rc:
            st.markdown(f'<div class="sl-eq">%R = {rc["txt"]} = <b>{f(rc["R"], 2)} %</b></div>', unsafe_allow_html=True)
            if rc["mode"] == "acuan":
                st.caption(f"Bias = {g(rc['bias'], 3)} · bias relatif = {f(rc['rel'], 2)} %")
            (st.success if rc["pass"] else st.error)(f"Recovery {'dalam' if rc['pass'] else 'di luar'} rentang {f(CR['rec_min'], 0)}–{f(CR['rec_max'], 0)} %.")
        else:
            st.info("Lengkapi nilai untuk menghitung % recovery.")
    st.subheader("Residual regresi kalibrasi")
    reg = cal_reg()
    if reg and reg.ok:
        pts = cal_points()
        fig = fig_base(f"Konsentrasi ({S['cal_unit']})", "Residual (A)", 300); fig.update_layout(hovermode="closest")
        fig.add_trace(go.Scatter(x=pts["C"], y=reg.residuals, mode="markers", name="Residual", marker=dict(size=10, color=PAL[2])))
        fig.add_hline(y=0, line_color=PAL[1])
        for sgn in (1, -1):
            fig.add_hline(y=sgn * CR["std_res_max"] * reg.sxy, line_dash="dot", line_color="#8a99b5")
        a, b = st.columns([1.5, 1])
        a.plotly_chart(fig, use_container_width=True)
        b.dataframe(pd.DataFrame({"C": pts["C"].values, "A": pts["A"].values, "Â": reg.yhat.round(4), "e": reg.residuals.round(4), "e/s": reg.std_residuals.round(2)}), hide_index=True, use_container_width=True)
        runs = C.residual_runs(reg.residuals)
        st.caption(f"{runs} kelompok tanda (runs) dari {reg.n} titik" + ("; pola sistematis → kemungkinan kelengkungan." if reg.n >= 5 and runs <= 2 else "; tidak tampak pola sistematis yang jelas."))
    else:
        st.info("Belum ada regresi kalibrasi.")


def page_photometric():
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown(f"Larutan aktif: **{sol_label()}** · kuvet {C.CUVETTES[I['cuvette']]['nama']} · referensi {I['blank']}  \n<span class='sl-chip sim'>SIMULASI</span>", unsafe_allow_html=True)
        st.caption("Ubah larutan, kuvet, dan blanko di halaman Simulasi Instrumen.")
        lams_txt = st.text_input("Panjang gelombang (maks 3, pisahkan ;)", f(I["lam"], 1))
        reps = st.select_slider("Ulangan per λ", [1, 2, 3, 5, 7, 10], value=3)
        ready = I["power"] == "ready"
        a, b = st.columns(2)
        try:
            lams = C.parse_list(lams_txt)[:3]
        except ValueError:
            lams = []; st.error("Isi λ berupa angka, pisahkan dengan ;")
        if a.button("Auto zero (blanko)", disabled=not ready or not lams, use_container_width=True):
            I["autozero"] = [{"lam": l, "cuvette": I["cuvette"], "blank": I["blank"]} for l in lams]; I["last_A"] = 0.0; st.toast("Auto zero selesai")
        if b.button("Ukur", type="primary", disabled=not ready or not lams, use_container_width=True):
            for l in lams:
                for _ in range(reps):
                    A, corr, _o = read_at(l); S["photo"].append({"Waktu": f"{datetime.now():%H.%M.%S}", "Larutan": sol_label(), "λ (nm)": l, "A": round(A, 4), "%T": round(float(C.abs_to_t(A)), 2), "Terkoreksi": corr})
                    I["lam"], I["last_A"] = l, A
        if not ready:
            st.caption("Nyalakan instrumen di Simulasi Instrumen.")
        lcd(mode="PHOTOMETRIC")
        st.markdown("**Konversi A ↔ %T**")
        cA = st.number_input("Absorbansi", -0.5, 4.0, 0.5, 0.01)
        st.markdown(f"%T = 100 × 10⁻ᴬ = **{f(float(C.abs_to_t(cA)), 3)} %**")
    with right:
        if not S["photo"]:
            st.info("Belum ada pembacaan. Lakukan auto zero dengan blanko, lalu tekan Ukur."); return
        df = pd.DataFrame(S["photo"])
        if not df["Terkoreksi"].all():
            st.warning("Sebagian pembacaan tanpa auto zero/baseline pada λ tersebut (mengandung offset instrumen).")
        grp = df.groupby(["Larutan", "λ (nm)"])["A"].agg(n="count", rerata="mean", s="std").reset_index()
        grp["RSD %"] = (grp["s"] / grp["rerata"] * 100).round(2); grp["%T rerata"] = C.abs_to_t(grp["rerata"]).round(2)
        st.dataframe(grp.round(5), hide_index=True, use_container_width=True)
        st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True, height=320)
        c1, c2 = st.columns(2)
        c1.download_button("Unduh CSV", df.to_csv(index=False, sep=";" if S["settings"]["dec"] == "koma" else ",", decimal="," if S["settings"]["dec"] == "koma" else "."), "fotometri-spectralab.csv", "text/csv")
        if c2.button("Hapus semua"):
            S["photo"] = []; st.rerun()


def kin_result():
    k = S["kin"]; d = k["data"]
    if len(d) < 3:
        return None
    t = np.array([p[0] for p in d]); A = np.array([p[1] for p in d])
    ainf = num_in(k["ainf_fit"]) if k["ainf_fit"] else (k["Ainf"] if k["source"] == "simulasi" else A[-1])
    return {"t": t, "A": A, "ir": C.initial_rate(t, A, k["n_init"]), "fo": C.first_order_fit(t, A, ainf), "avg": (A[-1] - A[0]) / (t[-1] - t[0]), "ainf": ainf}


def page_kinetics():
    st.markdown('<div class="sl-banner"><b>Model generik.</b> Kurva simulasi mengikuti A(t) = A∞ + (A₀ − A∞)·e<sup>−kt</sup> (orde satu semu) dan tidak mewakili reaksi kimia tertentu. Untuk data nyata pilih "Data pengguna".</div>', unsafe_allow_html=True)
    k = S["kin"]
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        src = st.radio("Sumber data", ["simulasi", "pengguna"], index=0 if k["source"] == "simulasi" else 1, horizontal=True, format_func=lambda s: "Simulasi" if s == "simulasi" else "Data pengguna")
        if src != k["source"]:
            k["source"] = src; k["data"] = []
        if src == "simulasi":
            pre = st.selectbox("Skenario", ["Pemudaran warna (orde satu semu)", "Pembentukan kompleks berwarna", "Atur sendiri"])
            if pre.startswith("Pemudaran"): k.update(A0=1.2, Ainf=0.02, k=0.012, lam=590.0)
            elif pre.startswith("Pembentukan"): k.update(A0=0.02, Ainf=0.85, k=0.02, lam=510.0)
            a, b = st.columns(2)
            k["A0"] = a.number_input("A₀", 0.0, 3.0, float(k["A0"]), 0.01, disabled=not pre.startswith("Atur"))
            k["Ainf"] = b.number_input("A∞", 0.0, 3.0, float(k["Ainf"]), 0.01, disabled=not pre.startswith("Atur"))
            k["k"] = a.number_input("k (s⁻¹)", 0.0001, 1.0, float(k["k"]), 0.001, format="%.4f", disabled=not pre.startswith("Atur"))
            k["lam"] = b.number_input("λ (nm)", 190.0, 1100.0, float(k["lam"]), 1.0)
            k["dur"] = a.number_input("Durasi (s)", 10.0, 3600.0, float(k["dur"]), 10.0)
            k["dt"] = b.number_input("Interval (s)", 0.5, 120.0, float(k["dt"]), 0.5)
            k["noise"] = a.number_input("Derau σ (A)", 0.0, 0.05, float(k["noise"]), 0.001, format="%.3f")
            speed = b.selectbox("Kecepatan tayang", [10, 30, 100], index=1, format_func=lambda x: f"{x}×")
            run = st.button("Mulai", type="primary", disabled=I["power"] != "ready")
            if I["power"] != "ready":
                st.caption("Nyalakan instrumen di Simulasi Instrumen.")
            if st.button("Atur ulang"):
                k["data"] = []; st.rerun()
        else:
            txt = st.text_area("Tempel data: waktu (s) dan absorbansi per baris", height=180, placeholder="0;0,812\n30;0,744\n60;0,688")
            run = False
            if st.button("Muat data") and txt.strip():
                res = C.parse_csv(txt)
                if res["ok"]:
                    d = sorted([(t, a) for t, a in zip(res["cols"][0], res["cols"][1]) if not (math.isnan(t) or math.isnan(a))])
                    k["data"] = d if len(d) >= 3 else []; st.toast(f"{len(d)} titik dimuat")
                else:
                    st.error(res["reason"])
        k["n_init"] = int(st.number_input("Titik untuk laju awal", 3, 50, int(k["n_init"])))
        k["ainf_fit"] = st.text_input("A∞ untuk ln-plot (kosong = otomatis)", k["ainf_fit"])
    with right:
        ph = st.empty()

        def draw():
            d = k["data"]; fig = fig_base("Waktu (s)", "Absorbansi", 360)
            if d:
                fig.add_trace(go.Scatter(x=[p[0] for p in d], y=[p[1] for p in d], mode="lines+markers", name="A(t)", line=dict(color=PAL[0]), marker=dict(size=5)))
            if k["source"] == "simulasi":
                fig.update_xaxes(range=[0, k["dur"]])
            ph.plotly_chart(fig, use_container_width=True)
        if run:
            k["data"] = []; g_ = np.random.default_rng(S["seed"] + S["counter"]); n = int(k["dur"] // k["dt"]) + 1
            for i in range(n):
                t = i * k["dt"]; k["data"].append((t, round(float(C.kinetic_model(k["A0"], k["Ainf"], k["k"], t) + g_.normal(0, k["noise"])), 4)))
                if i % max(1, n // 40) == 0 or i == n - 1:
                    draw(); time.sleep(min(0.15, k["dt"] / speed))
            log(f"Kinetika selesai: {n} titik pada {f(k['lam'], 1)} nm.")
        draw()
        r = kin_result()
        if not r:
            st.info("Minimal 3 titik data diperlukan."); return
        st.markdown(src_chip(k["source"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            rows = []
            if r["ir"]: rows += [("Laju awal dA/dt (" + str(r["ir"]["n"]) + " titik)", g(r["ir"]["rate"], 3) + " s⁻¹"), ("R² laju awal", f(r["ir"]["r2"], 4))]
            rows.append(("Laju rata-rata ΔA/Δt", g(r["avg"], 3) + " s⁻¹"))
            if r["fo"]: rows += [("A∞ dipakai", f(r["ainf"], 4)), ("k teramati (orde-1)", g(r["fo"]["k"], 3) + " s⁻¹"), ("t½ = ln2/k", g(r["fo"]["half_life"], 3) + " s"), ("R² ln-plot", f(r["fo"]["r2"], 4))]
            st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Nilai"]), hide_index=True, use_container_width=True)
            if r["fo"] and r["fo"]["r2"] < 0.98:
                st.warning("ln-plot kurang linear; data mungkin tidak orde satu atau A∞ belum tepat.")
        with c2:
            if r["fo"]:
                fo = r["fo"]; fig = fig_base("Waktu (s)", "ln|A − A∞|", 260); fig.update_layout(hovermode="closest")
                fig.add_trace(go.Scatter(x=fo["t"], y=fo["lny"], mode="markers", name="data", marker=dict(color=PAL[2], size=6)))
                fig.add_trace(go.Scatter(x=fo["t"], y=fo["lnA0"] - fo["k"] * fo["t"], mode="lines", name="regresi", line=dict(color=PAL[1])))
                st.plotly_chart(fig, use_container_width=True)
        st.download_button("Unduh CSV", pd.DataFrame(k["data"], columns=["Waktu (s)", "Absorbansi"]).to_csv(index=False), "kinetika-spectralab.csv", "text/csv")


def page_learn():
    tabs = st.tabs(["Prinsip kerja", "Lambert–Beer", "A dan %T", "λmaks", "Blanko", "Faktor", "Kesalahan umum", "Kuis"])
    with tabs[0]:
        st.markdown(LESSONS["Prinsip kerja"]); model3d(height=480, xray=True)
        for k_, v_ in COMPONENTS.items():
            st.markdown(f"**{k_}.** {v_}")
    with tabs[1]:
        st.latex(r"A = \varepsilon \cdot b \cdot C = -\log T")
        a, b, c = st.columns(3)
        eps = a.slider("ε (L·mol⁻¹·cm⁻¹)", 1000, 50000, 11100, 500); bb = b.slider("b (cm)", 0.1, 5.0, 1.0, 0.1); cc = c.slider("C (µmol/L)", 0, 150, 40)
        A = eps * bb * cc * 1e-6
        st.markdown(f'<div class="sl-eq">A = {eps:,} × {f(bb, 1)} × {f(cc * 1e-6, 6)} = <b>{f(A, 3)}</b> · %T = {f(float(C.abs_to_t(A)), 1)} %</div>'.replace(",", "."), unsafe_allow_html=True)
        xs = np.arange(0, 151, 5); fig = fig_base("C (µmol/L)", "Absorbansi", 280)
        fig.add_trace(go.Scatter(x=xs, y=eps * bb * xs * 1e-6, mode="lines", name="A = εbC", line=dict(color=PAL[0])))
        fig.add_trace(go.Scatter(x=[cc], y=[A], mode="markers", name="titik", marker=dict(size=12, color=PAL[2])))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Contoh ε awal ≈ kompleks Fe(II)–fenantrolin pada 510 nm. Berlaku untuk cahaya monokromatis dan larutan encer.")
    with tabs[2]:
        A2 = np.linspace(0.04, 2, 100); T = 10 ** -A2; err = np.abs(0.4343 * 0.005 / (T * np.log10(T))) * 100
        a, b = st.columns(2)
        fig = fig_base("Absorbansi", "%T", 280); fig.add_trace(go.Scatter(x=A2, y=C.abs_to_t(A2), mode="lines", name="%T = 100·10⁻ᴬ", line=dict(color=PAL[0]))); a.plotly_chart(fig, use_container_width=True)
        fig = fig_base("Absorbansi", "Galat relatif C (%)", 280); fig.add_trace(go.Scatter(x=A2, y=err, mode="lines", name="ΔT = 0,5 %", line=dict(color=PAL[2])))
        fig.add_vline(x=0.434, line_dash="dash", line_color=PAL[1], annotation_text="A = 0,434"); fig.add_vrect(x0=0.2, x1=0.8, fillcolor=PAL[4], opacity=0.12, line_width=0)
        b.plotly_chart(fig, use_container_width=True)
        st.markdown("A = 1 → 10 % cahaya diteruskan; A = 2 → 1 %; A = 3 → 0,1 % (batas atas UV-1700). Galat relatif minimum pada A ≈ 0,434, dasar anjuran daerah kerja ±0,2–0,8.")
    with tabs[3]:
        L = st.slider("λ pengukuran (nm)", 440, 600, 510)
        lam = np.arange(400, 621); y = 2 * C.absorptivity("fephen", lam)
        fig = fig_base("λ (nm)", "A (Fe 2 mg/L)", 280); fig.add_trace(go.Scatter(x=lam, y=y, mode="lines", name="Fe(II)–fenantrolin", line=dict(color=PAL[0]))); fig.add_vline(x=L, line_color=PAL[2])
        st.plotly_chart(fig, use_container_width=True)
        a_ = float(C.absorptivity("fephen", L)); d_ = float(C.absorptivity("fephen", L + 1) - C.absorptivity("fephen", L - 1)) / 2
        st.markdown(f"Kepekaan a = **{f(a_, 4)}** L·mg⁻¹·cm⁻¹ ({f(a_ / float(y.max() / 2) * 100, 1)} % dari maksimum); perubahan A per 1 nm = **{f(abs(d_ / a_) * 100, 2)} %**.")
    with tabs[4]:
        st.markdown(LESSONS["Fungsi blanko"])
    with tabs[5]:
        st.markdown(LESSONS["Faktor yang memengaruhi"])
    with tabs[6]:
        st.markdown(LESSONS["Kesalahan umum"])
    with tabs[7]:
        right = 0
        for i, (q, opts, ans, exp) in enumerate(QUIZ):
            st.markdown(f"**{i + 1}. {q}**")
            choice = st.radio(f"Jawaban {i + 1}", list(range(len(opts))), index=None, key=f"quiz{i}", format_func=lambda j, o=opts: f"{chr(65 + j)}. {o[j]}", label_visibility="collapsed")
            if choice is not None:
                (st.success if choice == ans else st.error)(("Benar. " if choice == ans else "Kurang tepat. ") + exp)
                right += choice == ans
        st.metric("Skor", f"{right}/{len(QUIZ)}")


def page_guide():
    exps = {"fephen": ("Penetapan besi dengan 1,10-fenantrolin", "reagen", (400, 650)), "kmno4": ("Penetapan kalium permanganat", "pelarut", (400, 700)), "crdpc": ("Penetapan Cr(VI) metode DPC", "reagen", (400, 700))}
    key = st.selectbox("Percobaan", list(exps), format_func=lambda k_: exps[k_][0])
    nama, blank, rng_ = exps[key]; reg = cal_reg()
    if st.button("Terapkan pengaturan percobaan ke instrumen"):
        a = C.ANALYTES[key]; I.update(analyte=key, blank=blank, cuvette="kuarsa", start=float(rng_[0]), end=float(rng_[1]), solution="standar", conc=float(a["std"][len(a["std"]) // 2]), lam=float(a["lam_ref"])); st.toast("Diterapkan")
    steps = [
        ("Nyalakan instrumen", "Jalankan inisialisasi; alat nyata dipanaskan 15–30 menit.", I["power"] == "ready"),
        ("Siapkan larutan & kuvet", f"Analit {C.ANALYTES[key]['rumus']}, kuvet kuarsa, referensi {blank}.", I["analyte"] == key and I["blank"] == blank),
        ("Koreksi baseline", f"Blanko di kedua sel, {rng_[0]}–{rng_[1]} nm.", bool(I["baseline"]) and I["baseline"]["analyte"] == key),
        ("Pindai spektrum standar", "Tentukan λmaks.", any(s["params"].get("analyte") == key and s["source"] == "simulasi" for s in S["spectra"])),
        ("Ukur deret standar", "Minimal 5 titik pada λmaks.", len(cal_points()) >= 5),
        ("Evaluasi kurva", f"R² ≥ {f(CR['r2_min'], 3)} dan residual tanpa pola.", bool(reg and reg.ok and reg.r2 >= CR["r2_min"])),
        ("Ukur sampel triplo", "Hitung kadar dengan faktor pengenceran.", any(sample_res(s, reg) and sample_res(s, reg)["n"] >= 3 for s in S["samples"]) if reg else False),
        ("Evaluasi presisi & akurasi", "RSD ulangan dan % recovery.", bool(precision() or recovery())),
        ("Susun laporan", "Lengkapi identitas, unduh PDF.", S["report"]["generated"]),
    ]
    done = sum(s[2] for s in steps); st.progress(done / len(steps), f"Kemajuan {done}/{len(steps)}")
    cur = next((i for i, s in enumerate(steps) if not s[2]), None)
    for i, (t, d, ok) in enumerate(steps):
        icon = "✅" if ok else ("👉" if i == cur else "⬜")
        st.markdown(f"{icon} **{i + 1}. {t}** — {d}")


def report_model() -> dict:
    rp = S["report"]; reg = cal_reg(); blocks = []; srcs = set(); u = S["cal_unit"]
    blocks += [{"t": "h", "text": "1. Identitas"}, {"t": "kv", "rows": [("Judul", rp["judul"] or "—"), ("Praktikan / NIM", (rp["praktikan"] or "—") + (f" / {rp['nim']}" if rp["nim"] else "")), ("Kelas", rp["kelas"] or "—"), ("Dosen / asisten", rp["dosen"] or "—"), ("Sampel", rp["sampel"] or "—"), ("Tanggal analisis", rp["tanggal"]), ("Mode kerja", S["settings"]["mode"])]}]
    blocks += [{"t": "h", "text": "2. Instrumen & parameter"}, {"t": "kv", "rows": [("Instrumen acuan", "Shimadzu UV-1700 (double-beam, 190–1100 nm). Pembacaan berasal dari instrumen virtual."), ("λ pengukuran", f"{f(S['cal_lambda'], 1)} nm" if S["cal_lambda"] else "—"), ("Kuvet / tebal", f"{C.CUVETTES[I['cuvette']]['nama']} / {f(I['path'], 1)} cm"), ("Referensi", I["blank"]), ("Koreksi baseline", f"{I['baseline']['start']:.0f}–{I['baseline']['end']:.0f} nm" if I["baseline"] else "tidak tercatat")]}]
    sp = next((s for s in S["spectra"] if s["id"] == S["last_scan"]), S["spectra"][-1] if S["spectra"] else None)
    n = 3
    if sp:
        srcs.add(sp["source"]); pk = C.find_peaks(sp["lam"], sp["A"])
        blocks += [{"t": "h", "text": f"{n}. Spektrum absorbansi"}, {"t": "img", "png": R.spectrum_png(sp["lam"], sp["A"], sp["name"], pk[0]["lambda"] if pk else None), "cap": f"Gambar 1. Spektrum {sp['name']} ({sp['source']})."},
                   {"t": "kv", "rows": [("λmaks", f"{f(pk[0]['lambda'], 1)} nm (A = {f(pk[0]['abs'], 4)})" if pk else "tidak terdeteksi")]}]; n += 1
    sres = []
    if reg and reg.ok:
        pts = cal_points(); srcs.add(S["cal_src"])
        sres = [(s, sample_res(s, reg)) for s in S["samples"]]; sres = [(s, r) for s, r in sres if r]
        blocks += [{"t": "h", "text": f"{n}. Kurva kalibrasi"},
                   {"t": "table", "head": ["No", f"C ({u})", "A", "Â", "Residual"], "rows": [[i + 1, g(c_), f(a_, 4), f(reg.yhat[i], 4), f(reg.residuals[i], 4)] for i, (c_, a_) in enumerate(zip(pts["C"], pts["A"]))]},
                   {"t": "img", "png": R.calib_png(pts["C"].values, pts["A"].values, reg, [(s["name"], r["c_meas"], r["A_mean"]) for s, r in sres], u), "cap": f"Gambar {2 if sp else 1}. Kurva kalibrasi {S['cal_name']}."},
                   {"t": "kv", "rows": [("Persamaan", reg.equation()), ("r / R²", f"{f(reg.r, 5)} / {f(reg.r2, 5)}"), ("s y/x", g(reg.sxy, 3)), ("LOD / LOQ", f"{g(reg.lod, 3)} / {g(reg.loq, 3)} {u}"), ("Rentang", f"{g(reg.xmin, 3)}–{g(reg.xmax, 3)} {u}")]}]; n += 1
        if sres:
            for s, _ in sres: srcs.add(s["src"])
            blocks += [{"t": "h", "text": f"{n}. Hasil penetapan kadar"}, {"t": "table", "head": ["Sampel", "n", "A rerata", "RSD %", "C terukur", "FP", "C asal ± SK95%", "Kadar mg/kg"],
                       "rows": [[s["name"], r["n"], f(r["A_mean"], 4), f(r["rsd"], 2) if r["n"] > 1 else "-", g(r["c_meas"]), g(r["fp"]), f"{g(r['c_final'])} ± {g(r['ci_final'], 2)} {u}", g(r["kadar"]) if r["mass_ok"] else "-"] for s, r in sres]}]
            for s, r in sres:
                if r["flags"]: blocks.append({"t": "note", "text": s["name"] + ": " + " ".join(r["flags"])})
    pr, rc = precision(), recovery()
    rows = []
    if reg and reg.ok: rows.append(("Linearitas", f"R² = {f(reg.r2, 4)} (≥ {f(CR['r2_min'], 3)}): {'memenuhi' if reg.r2 >= CR['r2_min'] else 'tidak memenuhi'}"))
    if pr: rows.append(("Presisi", f"n = {pr['d']['n']}, RSD = {f(pr['d']['rsd'], 2)} % (≤ {f(CR['rsd_max'], 2)} %): {'memenuhi' if pr['pass'] else 'tidak memenuhi'}"))
    if rc: rows.append(("Akurasi", f"% recovery = {f(rc['R'], 2)} % ({f(CR['rec_min'], 0)}–{f(CR['rec_max'], 0)} %): {'memenuhi' if rc['pass'] else 'tidak memenuhi'}"))
    if rows: blocks += [{"t": "h", "text": "Evaluasi mutu data"}, {"t": "kv", "rows": rows}]
    interp = R.calib_interpretation(reg, u, CR["r2_min"], CR["std_res_max"], S["settings"]["dec"]) if reg and reg.ok else []
    concl = [f"Konsentrasi {S['cal_name'] or 'analit'} dalam {s['name']} sebesar {g(r['c_final'])} ± {g(r['ci_final'], 2)} {u} (SK 95 %, n = {r['n']})" + (f", setara {g(r['kadar'])} mg/kg sampel." if r["mass_ok"] else ".") for s, r in sres]
    if reg and reg.ok: concl.append(f"Kurva kalibrasi {'memenuhi' if reg.r2 >= CR['r2_min'] else 'belum memenuhi'} kriteria linearitas (R² = {f(reg.r2, 4)}).")
    if pr: concl.append(f"Presisi {'memenuhi' if pr['pass'] else 'belum memenuhi'} kriteria (RSD {f(pr['d']['rsd'], 2)} %).")
    if rc: concl.append(f"Akurasi {'memenuhi' if rc['pass'] else 'belum memenuhi'} kriteria (% recovery {f(rc['R'], 2)} %).")
    if not concl: concl.append("Data belum cukup untuk menarik kesimpulan kuantitatif.")
    simulated = bool(srcs & {"simulasi", "contoh", "campuran"})
    if simulated: concl.append("Sebagian atau seluruh data berasal dari model simulasi/data contoh; hasil ini latihan, bukan hasil pengujian laboratorium.")
    if interp: blocks += [{"t": "h", "text": "Interpretasi"}] + [{"t": "p", "text": p} for p in interp]
    blocks += [{"t": "h", "text": "Kesimpulan"}] + [{"t": "p", "text": p} for p in concl]
    if rp["catatan"]: blocks += [{"t": "h", "text": "Catatan praktikan"}, {"t": "p", "text": rp["catatan"]}]
    blocks.append({"t": "note", "text": "Sumber data: " + (", ".join(sorted(srcs)) or "tidak ada") + ". Dibuat dengan SPECTRA LAB; aplikasi tidak terhubung langsung ke instrumen."})
    return {"blocks": blocks, "simulated": simulated, "judul": rp["judul"], "tanggal": rp["tanggal"], "praktikan": rp["praktikan"]}


def page_report():
    rp = S["report"]
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        for k_, lab in [("judul", "Judul percobaan"), ("praktikan", "Nama praktikan"), ("nim", "NIM"), ("kelas", "Kelas"), ("dosen", "Dosen / asisten"), ("sampel", "Identitas sampel")]:
            rp[k_] = st.text_input(lab, rp[k_])
        rp["tanggal"] = st.date_input("Tanggal analisis", date.fromisoformat(rp["tanggal"])).isoformat()
        rp["catatan"] = st.text_area("Catatan praktikan (opsional)", rp["catatan"])
        if st.button("Buat laporan PDF", type="primary"):
            with st.spinner("Menyusun laporan…"):
                rp["pdf"] = R.build_pdf(report_model()); rp["generated"] = True
        if rp["pdf"]:
            st.download_button("Unduh laporan PDF", rp["pdf"], f"laporan-uvvis-{rp['nim'] or 'spectralab'}.pdf", "application/pdf")
        proj = {k_: v_ for k_, v_ in S.items() if k_ not in ("cal_base", "cal_cur", "history")}
        proj["report"] = {k_: v_ for k_, v_ in rp.items() if k_ != "pdf"}; proj["cal_points"] = S["cal_base"].to_dict("records")
        st.download_button("Ekspor proyek (.json)", json.dumps(proj, default=str, ensure_ascii=False), "proyek-spectralab.json", "application/json")
        up = st.file_uploader("Impor proyek (.json)", type=["json"])
        if up is not None and st.button("Muat proyek"):
            try:
                o = json.loads(up.getvalue())
                new = _defaults(); [new.__setitem__(k_, v_) for k_, v_ in o.items() if k_ in new and k_ not in ("cal_base",)]
                new["cal_base"] = pd.DataFrame(o.get("cal_points", []), columns=["C", "A", "Pakai"]); new["report"]["pdf"] = None
                st.session_state.S = new; st.rerun()
            except Exception as e:
                st.error(f"Impor gagal: {e}")
    with right:
        st.markdown("**Pratinjau isi laporan** (mengikuti data yang tersedia)")
        for b in report_model()["blocks"]:
            if b["t"] == "h": st.markdown(f"#### {b['text']}")
            elif b["t"] == "p": st.markdown(b["text"])
            elif b["t"] == "note": st.caption(b["text"])
            elif b["t"] == "kv": st.dataframe(pd.DataFrame(b["rows"], columns=["", "Nilai"]), hide_index=True, use_container_width=True)
            elif b["t"] == "table": st.dataframe(pd.DataFrame(b["rows"], columns=b["head"]), hide_index=True, use_container_width=True)
            elif b["t"] == "img": st.image(b["png"], caption=b["cap"])


def page_history():
    st.caption("Riwayat tersimpan selama sesi ini. Untuk menyimpan permanen, unduh berkas JSON atau gunakan Ekspor proyek di menu Laporan.")
    if st.button("Simpan keadaan sekarang", type="primary"):
        reg = cal_reg()
        S["history"].insert(0, {"waktu": f"{datetime.now():%d/%m/%Y %H.%M}", "judul": S["report"]["judul"], "analit": S["cal_name"], "R²": round(reg.r2, 5) if reg and reg.ok else None,
                                "titik": len(cal_points()), "hasil": "; ".join(f"{s['name']}: {g(r['c_final'])} {S['cal_unit']}" for s in S["samples"] for r in [sample_res(s, reg)] if r) if reg else "",
                                "cal": S["cal_base"].to_dict("records"), "samples": json.loads(json.dumps(S["samples"], default=str))})
        st.toast("Tersimpan")
    if not S["history"]:
        st.info("Belum ada riwayat."); return
    st.dataframe(pd.DataFrame([{k_: v_ for k_, v_ in h.items() if k_ not in ("cal", "samples")} for h in S["history"]]), hide_index=True, use_container_width=True)
    i = st.selectbox("Pilih entri", range(len(S["history"])), format_func=lambda j: f"{S['history'][j]['waktu']} · {S['history'][j]['judul']}")
    c1, c2 = st.columns(2)
    if c1.button("Muat ke kurva kalibrasi"):
        h = S["history"][i]; set_calibration(pd.DataFrame(h["cal"], columns=["C", "A", "Pakai"]), "pengguna"); S["samples"] = h["samples"]; st.toast("Dimuat")
    c2.download_button("Unduh entri (.json)", json.dumps(S["history"][i], default=str, ensure_ascii=False), "riwayat-spectralab.json", "application/json")


st.title(page)
{"Dasbor": page_dashboard, "Model 3D Instrumen": page_model3d, "Simulasi Instrumen": page_instrument, "Fotometri": page_photometric,
 "Kinetika": page_kinetics, "Analisis Spektrum": page_spectrum, "Kurva Kalibrasi": page_calibration, "Validasi Data": page_validation,
 "Pembelajaran & Kuis": page_learn, "Panduan Praktikum": page_guide, "Laporan": page_report, "Riwayat": page_history}[page]()
