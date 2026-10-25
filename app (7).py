"""
SPEKTRO-1700 VIRTUAL LAB
Simulator & Pengolah Data Spektrofotometer UV-Vis Shimadzu UV-1700 PharmaSpec
Laboratorium Instrumentasi · Politeknik AKA Bogor
Alur kerja mengikuti Instruksi Kerja IK/LAB.INST/6 (berlaku 31 Agustus 2026).

Menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

Catatan penting
  * Semua pembacaan "instrumen" berasal dari MODEL SIMULASI; aplikasi tidak terhubung ke alat
    maupun ke perangkat lunak UVProbe. Data nyata dapat diketik/ditempel/diimpor (CSV) dan
    diolah dengan modul statistik yang sama.
  * Absorptivitas analit mendekati nilai literatur (orde besaran), bukan nilai sertifikat.

Struktur berkas
  BAGIAN 1  Statistik, regresi, validasi, pengolahan spektrum
  BAGIAN 2  Model fisik instrumen (sumber, kuvet, blanko, sinar sesat, drift, derau)
  BAGIAN 3  Instruksi kerja, materi, kuis
  BAGIAN 4  Grafik laporan & PDF
  BAGIAN 5  Model 3D instrumen (Plotly)
  BAGIAN 6  Aplikasi Streamlit
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from scipy import signal, stats

NAN = float("nan")

# =============================================================================
# BAGIAN 1 — STATISTIK, REGRESI, VALIDASI, PENGOLAHAN SPEKTRUM
# =============================================================================


def t_crit(df: float, alpha: float = 0.05) -> float:
    """Nilai t dua sisi."""
    return float(stats.t.ppf(1 - alpha / 2, df)) if df >= 1 else NAN


def f_crit(d1: int, d2: int, alpha: float = 0.05) -> float:
    return float(stats.f.ppf(1 - alpha, d1, d2)) if d1 >= 1 and d2 >= 1 else NAN


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": NAN, "sd": NAN, "rsd": NAN, "se": NAN, "ci95": NAN, "min": NAN, "max": NAN}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else NAN
    rsd = abs(sd / mean) * 100 if n > 1 and mean != 0 else NAN
    se = sd / math.sqrt(n) if n > 1 else NAN
    return {"n": n, "mean": mean, "sd": sd, "rsd": rsd, "se": se,
            "ci95": t_crit(n - 1) * se if n > 1 else NAN, "min": float(a.min()), "max": float(a.max())}


def grubbs(values, alpha: float = 0.05) -> dict:
    """Uji Grubbs dua sisi satu pencilan; nilai kritis dihitung dari distribusi t (berlaku umum n ≥ 3)."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n < 3:
        return {"ok": False, "n": n}
    m, s = a.mean(), a.std(ddof=1)
    if not s > 0:
        return {"ok": False, "n": n}
    g = np.abs(a - m) / s
    i = int(np.argmax(g))
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    gc = (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))
    return {"ok": True, "n": n, "G": float(g[i]), "Gcrit": float(gc), "index": i, "value": float(a[i]), "outlier": bool(g[i] > gc)}


def horwitz_prsd(c_fraction: float) -> float:
    """PRSD_R (%) = 2^(1 − 0,5·log10 C), C = fraksi massa."""
    return 2 ** (1 - 0.5 * math.log10(c_fraction)) if c_fraction > 0 else NAN


def t_test_mean(values, mu: float) -> dict | None:
    d = describe(values)
    if d["n"] < 2 or not d["sd"] > 0:
        return None
    t = abs(d["mean"] - mu) / (d["sd"] / math.sqrt(d["n"]))
    tc = t_crit(d["n"] - 1)
    return {"t": t, "tcrit": tc, "significant": t > tc, "p": float(2 * (1 - stats.t.cdf(t, d["n"] - 1)))}


# ---------------------------------------------------------------------------
# Regresi linear kuadrat terkecil (dengan opsi zero interception seperti UVProbe)
# ---------------------------------------------------------------------------
@dataclass
class Fit:
    ok: bool
    reason: str = ""
    n: int = 0
    df: int = 0
    a: float = NAN            # kemiringan
    b: float = NAN            # intersep
    r: float = NAN
    r2: float = NAN
    syx: float = NAN
    sa: float = NAN
    sb: float = NAN
    xbar: float = NAN
    ybar: float = NAN
    Sxx: float = NAN
    sumx2: float = NAN
    tc: float = NAN
    zero: bool = False
    xmin: float = NAN
    xmax: float = NAN
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    yhat: np.ndarray = field(default_factory=lambda: np.array([]))
    res: np.ndarray = field(default_factory=lambda: np.array([]))
    sres: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ci_a(self):
        return self.tc * self.sa

    @property
    def ci_b(self):
        return 0.0 if self.zero else self.tc * self.sb

    @property
    def b_significant(self):
        return (not self.zero) and abs(self.b) > self.tc * self.sb

    @property
    def lod(self):
        return 3.3 * self.syx / self.a

    @property
    def loq(self):
        return 10 * self.syx / self.a

    def equation(self, sig: int = 5) -> str:
        if self.zero:
            return f"y = {self.a:.{sig}g}x"
        sign = "−" if self.b < 0 else "+"
        return f"y = {self.a:.{sig}g}x {sign} {abs(self.b):.{sig - 1}g}"


def fit_line(x, y, zero: bool = False) -> Fit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    min_n = 2 if zero else 3
    if n < min_n:
        return Fit(ok=False, reason=f"Minimal {min_n} titik standar dengan absorbansi terbaca diperlukan.")
    xbar, ybar = float(x.mean()), float(y.mean())
    Sxx = float(((x - xbar) ** 2).sum())
    Syy = float(((y - ybar) ** 2).sum())
    Sxy = float(((x - xbar) * (y - ybar)).sum())
    sumx2 = float((x * x).sum())
    if not Sxx > 0:
        return Fit(ok=False, reason="Semua konsentrasi standar bernilai sama.")
    if zero:
        a, b, df = float((x * y).sum() / sumx2), 0.0, n - 1
    else:
        a = Sxy / Sxx
        b, df = ybar - a * xbar, n - 2
    yhat = a * x + b
    res = y - yhat
    ss = float((res ** 2).sum())
    syx = math.sqrt(ss / df) if df > 0 else NAN
    r = Sxy / math.sqrt(Sxx * Syy) if Syy > 0 else NAN
    r2 = 1 - ss / Syy if Syy > 0 else NAN          # SStot terpusat, juga untuk model melalui nol
    if zero:
        sa, sb = syx / math.sqrt(sumx2), 0.0
    else:
        sa = syx / math.sqrt(Sxx)
        sb = syx * math.sqrt(sumx2 / (n * Sxx))
    return Fit(ok=True, n=n, df=df, a=a, b=b, r=r, r2=r2, syx=syx, sa=sa, sb=sb, xbar=xbar, ybar=ybar, Sxx=Sxx,
               sumx2=sumx2, tc=t_crit(df), zero=zero, xmin=float(x.min()), xmax=float(x.max()), x=x, y=y,
               yhat=yhat, res=res, sres=res / syx if syx and syx > 0 else res * NAN)


def mandel_test(x, y, alpha: float = 0.01) -> dict | None:
    """Uji kecocokan Mandel (DIN 38402-51): linear vs kuadratik."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 5 or np.unique(x).size < 4:
        return None
    ss1 = float(((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2).sum())
    ss2 = float(((y - np.polyval(np.polyfit(x, y, 2), x)) ** 2).sum())
    s2 = ss2 / (n - 3)
    if not s2 > 0:
        return None
    F = (ss1 - ss2) / s2
    Fc = f_crit(1, n - 3, alpha)
    return {"F": F, "Fcrit": Fc, "nonlinear": F > Fc, "df2": n - 3}


def lack_of_fit(x, y) -> dict | None:
    """Uji lack-of-fit; hanya bila ada ulangan pada tingkat konsentrasi yang sama."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    levels = np.unique(x)
    k, n = levels.size, x.size
    if n - k < 1 or k < 3:
        return None
    ss_pe = float(sum(((y[x == l] - y[x == l].mean()) ** 2).sum() for l in levels))
    ss_res = float(((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2).sum())
    df_pe, df_lof = n - k, k - 2
    if not ss_pe > 0:
        return None
    F = ((ss_res - ss_pe) / df_lof) / (ss_pe / df_pe)
    Fc = f_crit(df_lof, df_pe)
    return {"F": F, "Fcrit": Fc, "lof": F > Fc, "df": (df_lof, df_pe)}


def residual_runs(res) -> int:
    s = [np.sign(v) for v in res if v != 0]
    return (1 + sum(1 for p, q in zip(s, s[1:]) if p != q)) if s else 0


def inverse_predict(fit: Fit, y0: float, m: int = 1) -> dict:
    """x0 dari rerata m pembacaan y0 dan s(x0) (Miller & Miller)."""
    x0 = (y0 - fit.b) / fit.a
    if fit.zero:
        sx0 = (fit.syx / abs(fit.a)) * math.sqrt(1 / m + y0 ** 2 / (fit.a ** 2 * fit.sumx2))
    else:
        sx0 = (fit.syx / abs(fit.a)) * math.sqrt(1 / m + 1 / fit.n + (y0 - fit.ybar) ** 2 / (fit.a ** 2 * fit.Sxx))
    return {"x0": x0, "sx0": sx0, "ci95": fit.tc * sx0, "extrapolated": bool(x0 < fit.xmin or x0 > fit.xmax)}


def dilution_factor(steps) -> float:
    """FP = Π(V labu / V pipet)."""
    f = 1.0
    for vp, vl in steps or []:
        if not (vp and vl and vp > 0 and vl > 0):
            return NAN
        f *= vl / vp
    return f


def sample_result(fit: Fit, reps, fp=1.0, mass_g=None, vol_ml=None, unit="mg/L", a_min=0.1, a_max=1.0, rsd_max=2.0):
    reps = [r for r in reps if r is not None and math.isfinite(r)]
    if not fit or not fit.ok or not reps:
        return None
    d = describe(reps)
    ip = inverse_predict(fit, d["mean"], len(reps))
    fp = fp if fp and math.isfinite(fp) else 1.0
    c_final, ci_final = ip["x0"] * fp, ip["ci95"] * fp
    mass_ok = bool(mass_g and vol_ml and mass_g > 0 and vol_ml > 0 and unit in ("mg/L", "ppm", "µg/mL"))
    kadar = c_final * vol_ml / mass_g if mass_ok else NAN     # mg/kg
    flags = []
    if ip["extrapolated"]:
        flags.append(f"C terukur di luar rentang standar ({fit.xmin:.3g}–{fit.xmax:.3g} {unit}); hasil ekstrapolasi tidak tervalidasi.")
    if d["mean"] < a_min or d["mean"] > a_max:
        flags.append(f"Ā = {d['mean']:.3f} di luar daerah kerja absorbansi {a_min}–{a_max}.")
    if len(reps) > 1 and d["rsd"] > rsd_max:
        flags.append(f"RSD ulangan {d['rsd']:.2f} % melebihi batas {rsd_max} %.")
    if len(reps) < 2:
        flags.append("Hanya satu pembacaan; presisi tidak dapat dievaluasi.")
    if ip["x0"] < 0:
        flags.append("Konsentrasi negatif: absorbansi sampel di bawah intersep (periksa blanko/autozero).")
    elif ip["x0"] < fit.loq:
        flags.append(f"C terukur di bawah LOQ ({fit.loq:.3g} {unit}); nilai hanya indikatif.")
    return {"n": len(reps), "A_mean": d["mean"], "sd": d["sd"], "rsd": d["rsd"], "c_meas": ip["x0"], "sx0": ip["sx0"],
            "ci": ip["ci95"], "fp": fp, "c_final": c_final, "ci_final": ci_final, "kadar": kadar, "mass_ok": mass_ok,
            "flags": flags}


def ringbom_error(A, dT=0.005):
    """Galat relatif konsentrasi (%) untuk galat transmitansi absolut konstan ΔT."""
    A = np.asarray(A, float)
    T = 10 ** -A
    return np.abs(0.4343 * dT / (T * np.log10(T))) * 100


# ---------------------------------------------------------------------------
# Spektrum
# ---------------------------------------------------------------------------
def smooth(y, w: int = 7):
    y = np.asarray(y, float)
    if w < 5 or y.size < w + 2:
        return y.copy()
    return signal.savgol_filter(y, w if w % 2 else w + 1, 2)


def peak_pick(x, y, threshold: float = 0.01, smooth_w: int = 7, valleys: bool = True) -> list[dict]:
    """Puncak & lembah (seperti Operation → Peak Pick di UVProbe), λ diinterpolasi parabola."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 7:
        return []
    o = np.argsort(x)
    xs, yr = x[o], y[o]
    ys = smooth(yr, smooth_w)
    step = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    out = []
    for kind, sig in (("Puncak", ys), ("Lembah", -ys)):
        if kind == "Lembah" and not valleys:
            continue
        idx, prop = signal.find_peaks(sig, prominence=threshold)
        if idx.size == 0:
            continue
        w = signal.peak_widths(sig, idx, rel_height=0.5)[0] * step
        for k, i in enumerate(idx):
            off = 0.0
            if 0 < i < xs.size - 1:
                y1, y2, y3 = sig[i - 1], sig[i], sig[i + 1]
                den = y1 - 2 * y2 + y3
                off = 0.5 * (y1 - y3) / den if den != 0 else 0.0
            out.append({"Jenis": kind, "λ (nm)": float(xs[i] + off * step), "Abs": float(yr[i]),
                        "Prominensi": float(prop["prominences"][k]), "FWHM (nm)": float(w[k]) if kind == "Puncak" else NAN})
    return sorted(out, key=lambda p: (p["Jenis"] != "Puncak", -p["Abs"] if p["Jenis"] == "Puncak" else p["Abs"]))


def interp_y(x, y, xt):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or xt < x.min() or xt > x.max():
        return NAN
    o = np.argsort(x)
    return float(np.interp(xt, x[o], y[o]))


def parse_number(s) -> float:
    if isinstance(s, (int, float, np.floating)):
        return float(s)
    if s is None:
        return NAN
    s = re.sub(r"\s", "", str(s))
    if not s:
        return NAN
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d*,\d+(e[-+]?\d+)?", s, flags=re.I):
        s = s.replace(",", ".")
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?", s, flags=re.I):
        return NAN
    return float(s)


def parse_list(text: str) -> list[float]:
    parts = [p for p in re.split(r"[;\s|]+", text or "") if p]
    vals = [parse_number(p) for p in parts]
    if any(math.isnan(v) for v in vals):
        raise ValueError("Pisahkan nilai dengan titik koma, mis. 0,412; 0,415")
    return vals


def parse_table(text: str) -> dict:
    """CSV/TSV/tempelan Excel; pemisah ; , tab; desimal koma didukung."""
    lines = [l for l in (text or "").replace("\r", "").split("\n") if l.strip()]
    if not lines:
        return {"ok": False, "reason": "Data kosong."}
    sample = "\n".join(lines[:20])
    if sample.count("\t"):
        delim = "\t"
    elif sample.count(";"):
        delim = ";"
    else:
        delim = ","
    header, rows = None, []
    for line in lines:
        cells = [c.strip().strip('"') for c in line.split(delim)]
        nums = [parse_number(c) for c in cells]
        if sum(not math.isnan(v) for v in nums) >= 2 and not math.isnan(nums[0]):
            rows.append(nums)
        elif not rows:
            header = cells
    if not rows:
        return {"ok": False, "reason": "Tidak ada baris numerik (minimal 2 kolom angka)."}
    nc = max(len(r) for r in rows)
    cols = [np.array([r[c] if c < len(r) else NAN for r in rows]) for c in range(nc)]
    return {"ok": True, "header": header, "cols": cols, "nrows": len(rows)}


def lam_color(l: float) -> str:
    if l < 380:
        return "#9b8cf0"
    if l > 780:
        return "#a8322c"
    r = g = b = 0.0
    if l < 440:
        r, b = -(l - 440) / 60, 1
    elif l < 490:
        g, b = (l - 440) / 50, 1
    elif l < 510:
        g, b = 1, -(l - 510) / 20
    elif l < 580:
        r, g = (l - 510) / 70, 1
    elif l < 645:
        r, g = 1, -(l - 645) / 65
    else:
        r = 1
    f = 0.45 + 0.55 * (l - 380) / 40 if l < 420 else (0.45 + 0.55 * (780 - l) / 80 if l > 700 else 1)
    return "#" + "".join(f"{round(255 * (max(0.0, v) * f) ** 0.8):02x}" for v in (r, g, b))


def complementary_hint(l: float) -> str:
    table = [(435, "kuning-hijau"), (480, "kuning"), (490, "jingga"), (500, "merah"), (560, "ungu"), (580, "biru"),
             (595, "biru-kehijauan"), (650, "hijau-kebiruan"), (780, "hijau")]
    for lim, c in table:
        if l < lim:
            return c
    return "—"


# =============================================================================
# BAGIAN 2 — MODEL FISIK INSTRUMEN (DISEDERHANAKAN)
# =============================================================================
UV1700 = {
    "Rentang panjang gelombang": "190–1100 nm",
    "Lebar pita spektral": "1 nm (tetap)",
    "Sistem optik": "Berkas ganda (double-beam), kisi difraksi holografik",
    "Sumber": "Lampu deuterium (UV) & wolfram-iodin (Vis–NIR), pergantian otomatis",
    "Detektor": "Fotodioda silikon",
    "Antarmuka": "RS-232C (kendali PC via UVProbe), Centronics (printer), slot IC card",
    "Daya": "230 V AC, 50/60 Hz",
}

# pita = (λ pusat, tinggi relatif, σ nm); a = absorptivitas (L·mg⁻¹·cm⁻¹) pada lam
ANALIT = {
    "fe": dict(nama="Besi(II)–1,10-fenantrolin", rumus="[Fe(phen)₃]²⁺", lam=510, a=0.199, unit="mg/L",
               pita=[(511, 1.0, 23), (478, 0.55, 11), (396, 0.07, 30), (265, 5.0, 15)], std=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
               pelarut="Air (bufer asetat pH 4,5)", reagen=[(265, 0.85, 16), (228, 1.3, 12)], warna="#d2452b",
               kinetik=dict(k1=0.35, k2=0.0002), mr=55.845,
               catatan="Fe(III) direduksi hidroksilamin; kompleks stabil pada pH 3–9. ε ≈ 1,1 × 10⁴ L·mol⁻¹·cm⁻¹."),
    "no2": dict(nama="Nitrit (Griess: sulfanilamida–NED)", rumus="NO₂⁻–N (zat warna azo)", lam=543, a=3.2, unit="mg/L",
                pita=[(543, 1.0, 42), (330, 0.12, 30)], std=[0.02, 0.05, 0.08, 0.12, 0.16, 0.20],
                pelarut="Air", reagen=[(260, 0.6, 18), (543, 0.004, 60)], warna="#c2185b",
                kinetik=dict(k1=0.25, k2=0.0008), mr=14.007,
                catatan="Diazotasi sulfanilamida lalu kopling dengan NED; ukur 10 menit–2 jam setelah penambahan reagen."),
    "po4": dict(nama="Fosfat (biru molibdenum, asam askorbat)", rumus="PO₄³⁻–P", lam=880, a=0.72, unit="mg/L",
                pita=[(882, 1.0, 85), (712, 0.5, 55)], std=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                pelarut="Air", reagen=[(880, 0.010, 120), (230, 1.0, 14)], warna="#1e5fbf",
                kinetik=dict(k1=0.22, k2=0.004), mr=30.974,
                catatan="Warna terbentuk ±10 menit dan memudar perlahan; ukur dalam jendela 10–30 menit."),
    "cr": dict(nama="Kromium(VI)–difenilkarbazida", rumus="Cr(VI)–DPC", lam=540, a=0.77, unit="mg/L",
               pita=[(542, 1.0, 40), (360, 0.08, 30), (290, 0.35, 22)], std=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
               pelarut="H₂SO₄ encer (pH ≈ 1–2)", reagen=[(285, 0.5, 18), (540, 0.006, 60), (235, 0.9, 15)], warna="#8e2c7a",
               kinetik=dict(k1=1.2, k2=0.012), mr=51.996,
               catatan="Kompleks merah-ungu; terbentuk cepat dan memudar; blanko reagen wajib."),
    "kmno4": dict(nama="Kalium permanganat", rumus="KMnO₄", lam=525, a=0.0152, unit="mg/L",
                  pita=[(489, 0.22, 10), (507, 0.62, 8), (526, 1.0, 8.5), (546, 0.97, 8), (566, 0.46, 8.5), (311, 0.30, 20)],
                  std=[10, 20, 30, 40, 50, 60], pelarut="Air", reagen=None, warna="#7d2c8f",
                  kinetik=dict(k1=50.0, k2=0.0005), mr=158.03,
                  catatan="Struktur vibronik 500–570 nm dari transisi transfer muatan ligan→logam MnO₄⁻."),
    "kafein": dict(nama="Kafein (UV, air)", rumus="C₈H₁₀N₄O₂", lam=273, a=0.0502, unit="mg/L",
                   pita=[(273, 1.0, 14), (205, 2.1, 9)], std=[2, 4, 6, 8, 10, 12], pelarut="Air", reagen=None,
                   warna="#e8eef2", kinetik=dict(k1=50.0, k2=0.0), mr=194.19,
                   catatan="Daerah UV: wajib kuvet kuarsa. ε ≈ 9,7 × 10³ L·mol⁻¹·cm⁻¹ pada 273 nm."),
}


def _bands(b, lam):
    lam = np.asarray(lam, float)
    out = np.zeros_like(lam, dtype=float)
    for l0, h, w in b or []:
        out = out + h * np.exp(-0.5 * ((lam - l0) / w) ** 2)
    return out


for _an in ANALIT.values():
    _an["k"] = _an["a"] / float(_bands(_an["pita"], _an["lam"]))


def absorptivity(key: str, lam):
    """Absorbansi per (mg/L·cm) menurut model."""
    return ANALIT[key]["k"] * _bands(ANALIT[key]["pita"], lam)


KUVET = {
    "kuarsa": dict(nama="Kuarsa (Q)", cutoff=190, A=lambda l: 0.020 + 2.5 / (1 + np.exp((l - 172) / 4))),
    "kaca": dict(nama="Kaca optik", cutoff=320, A=lambda l: 0.040 + 4.0 / (1 + np.exp((l - 318) / 7))),
    "plastik": dict(nama="Polistirena (PS)", cutoff=340, A=lambda l: 0.050 + 4.0 / (1 + np.exp((l - 330) / 6))),
}
CUTOFF_PELARUT = {"Air": 190, "Air (bufer asetat pH 4,5)": 230, "H₂SO₄ encer (pH ≈ 1–2)": 200}


def A_solvent(name, lam):
    return 3.0 / (1 + np.exp((np.asarray(lam, float) - CUTOFF_PELARUT.get(name, 190)) / 5))


def lamp_energy(lam, switch=340.8):
    lam = np.asarray(lam, float)
    e = np.where(lam < switch, 0.15 + 0.85 * np.exp(-0.5 * ((lam - 240) / 70) ** 2),
                 0.08 + 0.92 * np.exp(-0.5 * ((lam - 850) / 300) ** 2))
    e = np.where(lam < 200, e * np.maximum(0.08, (lam - 185) / 15), e)
    return np.maximum(e, 0.02)


def instrument_baseline(lam, switch=340.8):
    """Ketakseimbangan optik kedua berkas (terkoreksi oleh baseline/autozero)."""
    lam = np.asarray(lam, float)
    return (0.012 + 1.8e-5 * (lam - 500) + 0.005 * np.sin(lam / 41) + np.where(lam < switch, 0.004, 0.0)
            + 0.012 * np.exp(-((lam - 195) ** 2) / 70))


def stray_light(lam):
    return np.where(np.asarray(lam) < 230, 4e-4, 1e-4)


def drift_curve(t_min: float) -> float:
    """Drift kumulatif (A) terhadap waktu sejak lampu menyala; laju turun setelah pemanasan."""
    t = max(0.0, t_min)
    return 0.006 * (1 - math.exp(-t / 5.0)) + 4e-5 * t


NOISE_T = {"Fast": 6e-4, "Medium": 3e-4, "Slow": 1.6e-4}


def cell_absorbance(content: dict, lam, key: str, cfg: dict):
    """Absorbansi isi satu sel (kuvet + larutan) terhadap udara."""
    lam = np.asarray(lam, float)
    if not content or content.get("kind") == "kosong":
        return np.zeros_like(lam)
    an = ANALIT[key]
    b = cfg["b"]
    A = KUVET[cfg["kuvet"]]["A"](lam) + A_solvent(an["pelarut"], lam) * b
    kind = content["kind"]
    with_reagent = an["reagen"] and (kind in ("standar", "sampel") or (kind == "blanko" and cfg["blanko"] == "reagen"))
    if with_reagent:
        A = A + _bands(an["reagen"], lam) * b
    if kind in ("standar", "sampel"):
        A = A + content.get("conc", 0.0) * absorptivity(key, lam) * b
    if kind == "sampel":
        A = A + content.get("matriks", 0.0) * (400.0 / lam) ** 2 * b
    return A


def raw_reading(lam, key: str, sample: dict, ref: dict, cfg: dict, t_warm: float, rng=None, speed="Medium"):
    """Sinyal mentah A = −log(I/I₀) sebelum dikurangi koreksi baseline/autozero."""
    rng = rng or np.random.default_rng()
    lam = np.atleast_1d(np.asarray(lam, float))
    sw = cfg.get("lamp_switch", 340.8)
    s = stray_light(lam)
    Ts = (10 ** -cell_absorbance(sample, lam, key, cfg) + s) / (1 + s)
    Tr = (10 ** -cell_absorbance(ref, lam, key, cfg) + s) / (1 + s)
    if cfg.get("silika"):
        Ts = Ts * 2e-3                                  # kantong silika gel menghalangi berkas sampel
    if cfg.get("lid_open"):
        Ts, Tr = Ts + 0.02, Tr + 0.02                   # cahaya ruangan masuk ke detektor
    E = lamp_energy(lam, sw)
    A = -np.log10(Ts / Tr) + instrument_baseline(lam, sw) + 0.25 * drift_curve(t_warm)
    sig = 0.4343 * NOISE_T.get(speed, 3e-4) / E * np.sqrt(1 / Ts ** 2 + 1 / Tr ** 2) / math.sqrt(2)
    if cfg.get("lid_open"):
        sig = sig * 4
    sig = np.minimum(sig, 0.5)
    noise = rng.normal(0, 1, lam.size) * sig
    return {"A": A, "noise": noise, "low_energy": (E * Ts) < 1e-3}


def hidden_conc(key: str, seed: int, idx: int = 0) -> float:
    std = ANALIT[key]["std"]
    r = np.random.default_rng((seed * 131 + idx * 17 + sum(map(ord, key))) % (2 ** 32)).random()
    lo, hi = std[1], std[-2]
    return float(f"{lo + (hi - lo) * (0.1 + 0.8 * r):.3g}")


def prep_conc(c: float, rng, rsd: float = 0.003) -> float:
    """Galat preparasi (pipet/labu) acak, RSD ≈ 0,3 %."""
    return c * (1 + rsd * rng.normal())


def color_kinetics(key: str, t_min, A_plateau: float, rng=None, noise: float = 0.002):
    """A(t) = A_p·(1 − e^(−k₁t))·e^(−k₂t): pembentukan lalu pemudaran warna (model generik)."""
    rng = rng or np.random.default_rng()
    k = ANALIT[key]["kinetik"]
    t = np.asarray(t_min, float)
    return A_plateau * (1 - np.exp(-k["k1"] * t)) * np.exp(-k["k2"] * t) + rng.normal(0, noise, t.size)


def stable_window(t, A, tol_pct: float = 1.0):
    t = np.asarray(t, float)
    A = np.asarray(A, float)
    if t.size < 7:
        return None
    As = smooth(A, 7)
    i = int(np.argmax(As))
    lim = As[i] * (1 - tol_pct / 100)
    lo = i
    while lo > 0 and As[lo - 1] >= lim:
        lo -= 1
    hi = i
    while hi < As.size - 1 and As[hi + 1] >= lim:
        hi += 1
    return {"t0": float(t[lo]), "t1": float(t[hi]), "Amax": float(As[i]), "tmax": float(t[i])}


# =============================================================================
# BAGIAN 3 — INSTRUKSI KERJA, MATERI, KUIS
# =============================================================================
IK_META = {"kode": "IK/LAB.INST/6", "berlaku": "31 Agustus 2026", "revisi": "-",
           "judul": "Instruksi Kerja Penggunaan Alat Spektrofotometer UV-Vis Shimadzu UV-1700 PharmaSpec",
           "lab": "Laboratorium Instrumentasi, Politeknik AKA Bogor"}

IK_PERSIAPAN = [
    ("kabel", "Hubungkan kabel terminal utama ke sumber listrik."),
    ("ups", "Tekan tombol power pada UPS."),
    ("silika", "Keluarkan silika gel dari kompartemen dan buka monitor (layar) pada UV-1700."),
    ("power", "Nyalakan UV-1700 dengan tombol POWER di sisi kiri alat; tunggu pengecekan otomatis selesai."),
    ("warmup", "Setelah pengecekan selesai, tunggu 15 menit (warming up)."),
    ("pc", "Tekan F4 (PC Control) pada UV-1700; nyalakan CPU, monitor, dan printer; buka UVProbe → Connect."),
]

IK_SPEKTRUM = [
    "Window → Spectrum.",
    "Edit → Method: rentang λ (start = λ tertinggi, end = λ terendah), scan speed, Instrument Parameters → measuring mode Absorbance → OK.",
    "Masukkan blanko pada kedua kompartemen → Baseline → OK.",
    "Ganti blanko di kompartemen sampel dengan standar konsentrasi tertinggi → Start → OK → tunggu.",
    "Operation → Peak Pick untuk melihat λ maksimum.",
    "File → Save As → nama berkas Tanggal_Kelas_Kelompok → Save. Cetak: klik tabel peak pick → klik kanan → Print.",
]

IK_FOTOMETRIK = [
    "Window → Photometric; Edit → Method.",
    "Wavelength (nm): ketik λ → Add → Next.",
    "Type: pilih metode (Multi Point untuk standar > 1 titik); Unit; WL1; Zero Interception → Next 3×.",
    "File name: (...) → pilih folder → nama Tanggal_Kelas_Kelompok → Open → Finish.",
    "Standard Table: isi Sample ID dan Conc tiap standar.",
    "Sel sampel diisi blanko → Autozero.",
    "Ganti dengan standar → Read Std (ulangi untuk seluruh deret).",
    "Sample Table: isi Sample ID → ganti dengan sampel → Read Unk (ulangi untuk seluruh sampel).",
    "File → Save; cetak Standard Table dan Sample Table (klik kanan → Print).",
]

IK_MATIKAN = [
    ("logbook", "Catat semua kejadian pada Log Book Alat dan tanda tangani dengan nama jelas."),
    ("disconnect", "UVProbe: klik Disconnect."),
    ("exit", "File → Exit."),
    ("shutdown_pc", "Windows → Shutdown Computer."),
    ("monitor", "Matikan monitor komputer dan printer."),
    ("mode", "Tekan MODE pada UV-1700 (keluar dari PC Control)."),
    ("power_off", "Matikan UV-1700 (POWER sisi kiri), tutup monitor alat, masukkan kembali silika gel ke kompartemen."),
    ("ups_off", "Tekan OFF pada UPS."),
    ("cabut", "Cabut kabel dari kabel perpanjangan, kecuali kabel UPS."),
    ("rapi", "Rapikan area kerja dari sampel dan standar; rapikan kursi."),
    ("kuvet", "Keringkan kuvet kuarsa/kaca: bilas etanol lalu aseton, balik kuvet pada posisi aman hingga kering."),
]

SELF_CHECK = ["Memori ROM", "Memori RAM", "Inisialisasi filter", "Inisialisasi celah (slit)", "Lampu WI menyala",
              "Lampu D₂ menyala", "Titik nol panjang gelombang (garis emisi D₂ 656,1 nm)", "Energi berkas", "Arus gelap detektor"]

KOMPONEN = {
    "Lampu D₂": "Sumber kontinu UV (±190–350 nm). Di atas λ pergantian, alat beralih ke lampu WI; tampilan 'Light' pada LCD menunjukkan lampu aktif.",
    "Lampu WI": "Wolfram-iodin (halogen), sumber kontinu tampak–NIR (±340–1100 nm). Pendaran oranye pada kisi ventilasi sisi kanan berasal dari lampu ini; label HOT SURFACE mengingatkan agar lampu didinginkan sebelum diganti.",
    "Cermin pemilih sumber": "Mengarahkan berkas dari D₂ atau WI ke celah masuk sesuai λ.",
    "Celah masuk/keluar": "Menentukan lebar pita spektral (1 nm pada UV-1700). Lebar pita yang sempit relatif terhadap lebar pita absorpsi menjaga linearitas Lambert–Beer.",
    "Kisi difraksi": "Kisi holografik mendispersikan cahaya; rotasi kisi memilih λ yang keluar melalui celah keluar.",
    "Pembagi berkas": "Membagi berkas monokromatis ke jalur referensi dan sampel (double-beam) sehingga fluktuasi lampu dan sebagian drift terkompensasi.",
    "Kompartemen sampel": "Dua dudukan kuvet: referensi (belakang) dan sampel (depan). Saat tidak dipakai berisi silika gel untuk menjaga kelembapan rendah; silika gel wajib dikeluarkan sebelum pengukuran.",
    "Detektor": "Fotodioda silikon pada kedua jalur; rasio I/I₀ diubah menjadi A = log(I₀/I).",
    "Panel tombol & LCD": "F1–F4, RETURN, MODE, PRINT, angka, GOTO WL, AUTO ZERO, START/STOP, ENTER. F4 = PC Control (kendali dari UVProbe); MODE untuk keluar.",
    "Sisi kanan": "Slot IC card (EJECT), kenop CONTRAST layar, kisi ventilasi rumah lampu.",
    "Sisi kiri & belakang": "Saklar POWER, soket AC, sekring, kipas; port I/O (RS-232C) ke PC dan Centronics ke printer.",
}

LESSONS = {
    "Prinsip & optik UV-1700": """
UV-1700 adalah spektrofotometer **berkas ganda** dengan lebar pita tetap 1 nm. Cahaya dari lampu D₂ (UV) atau WI (tampak–NIR)
didispersikan kisi holografik; pita sempit yang keluar dibagi menjadi berkas **referensi** dan **sampel**. Fotodioda silikon
mengukur I₀ (referensi) dan I (sampel) hampir bersamaan, sehingga **A = log(I₀/I)** relatif tidak sensitif terhadap fluktuasi intensitas lampu.

Konsekuensi praktis: berkas ganda mengompensasi fluktuasi **intensitas**, tetapi tidak mengoreksi ketakseimbangan optik kedua jalur
dan perbedaan kuvet — itulah fungsi **Baseline** (mode spektrum) dan **Autozero** (mode fotometrik).
""",
    "Lambert–Beer & penyimpangannya": """
**A = ε·b·C** berlaku untuk radiasi monokromatis, larutan encer, dan tanpa interaksi spesies. Penyimpangan yang lazim:

- **Negatif pada A tinggi** — sinar sesat: A_terukur = −log[(T + s)/(1 + s)]; dengan s = 10⁻⁴, A tidak dapat melampaui ±4 dan kurva melengkung jauh sebelum itu.
- **Lebar pita** — bila lebar pita alat tidak ≪ lebar pita absorpsi (rasio < 0,1 dianjurkan). Dengan 1 nm, pita lebar seperti Fe–fenantrolin (FWHM ±60 nm) praktis bebas efek ini, tetapi struktur vibronik KMnO₄ lebih sensitif.
- **Kimia** — dimerisasi, kesetimbangan asam–basa, pembentukan kompleks yang belum sempurna.
- **Instrumental** — cahaya ruangan (tutup kompartemen terbuka), sidik jari, gelembung, kuvet tidak cocok.
""",
    "λmaks & daerah kerja": """
Pengukuran pada **λmaks** memaksimalkan kepekaan dan meminimalkan galat akibat ketidaktepatan λ karena **dA/dλ ≈ 0**.
Galat relatif konsentrasi untuk galat transmitansi konstan (Ringbom/Twyman–Lothian) minimum pada **A = 0,434**;
daerah kerja yang dianjurkan berkisar **0,2–0,8 A** (atau sesuai SOP).
""",
    "Blanko, baseline, autozero": """
- **Blanko pelarut** cukup untuk analit yang menyerap sendiri (KMnO₄, kafein).
- **Blanko reagen** wajib untuk metode kolorimetri (Fe–fenantrolin, nitrit, fosfat, Cr–DPC) karena reagen dapat menyerap pada λ ukur.
- **Baseline** merekam kurva koreksi di seluruh rentang pemindaian dengan blanko di kedua sel.
- **Autozero** menolkan pembacaan pada λ tertentu dengan blanko di sel sampel. Autozero dengan larutan selain blanko menggeser **semua** pembacaan berikutnya sebesar absorbansi larutan tersebut.
""",
    "Pemanasan & drift": """
Intensitas dan spektrum emisi lampu belum stabil sesaat setelah dinyalakan. Instruksi kerja menetapkan **pemanasan 15 menit**.
Dalam simulator, drift kumulatif dimodelkan sebagai D(t) = 0,006·(1 − e^(−t/5)) + 4·10⁻⁵·t (A), yang setelah kompensasi berkas ganda
menghasilkan laju drift ±10⁻⁴ A/menit setelah pemanasan, tetapi ±10× lebih besar pada menit-menit pertama.
Implikasinya: autozero ulang sebelum deret standar dan sebelum sampel bila pengukuran berlangsung lama.
""",
    "Zero interception (statistik)": """
Pilihan **Zero Interception** di UVProbe memaksa kurva melalui titik (0,0). Model ini hanya sahih bila intersep model bebas
**tidak berbeda nyata dari nol**, yaitu |b| ≤ t(0,05; n−2)·s_b. Bila intersep nyata (mis. akibat serapan reagen yang tidak terkompensasi
atau autozero yang keliru), pemaksaan melalui nol memindahkan galat sistematis ke kemiringan dan menghasilkan bias yang
paling besar pada konsentrasi rendah. Aplikasi ini menampilkan kedua model untuk perbandingan.
""",
    "Kesalahan umum": """
| Kesalahan | Akibat yang teramati |
|---|---|
| Silika gel tertinggal di kompartemen | Energi berkas sampel ≈ 0; pembacaan sangat tinggi/OVER |
| Mengukur sebelum pemanasan selesai | Drift; nilai blanko bergeser selama deret |
| Tutup kompartemen terbuka | Cahaya ruangan menurunkan A dan menambah derau |
| Kuvet kaca/plastik di bawah 320–340 nm | Energi hampir nol; data hanya derau |
| Blanko pelarut untuk metode kolorimetri | Serapan reagen terbaca sebagai analit; intersep positif |
| Autozero dengan larutan standar | Seluruh pembacaan bergeser negatif |
| A di luar daerah kerja / ekstrapolasi | Galat relatif meningkat; hasil tidak tervalidasi |
| Lupa faktor pengenceran | Hasil akhir meleset sebesar FP |
""",
}

QUIZ = [
    ("Menurut IK/LAB.INST/6, berapa lama UV-1700 dipanaskan setelah pengecekan otomatis?", ["5 menit", "15 menit", "30 menit", "Tidak perlu"], 1,
     "IK menetapkan 15 menit warming up agar intensitas lampu stabil dan drift kecil."),
    ("Saat mengatur metode spektrum di UVProbe, nilai Start diisi…", ["λ terendah", "λ tertinggi", "λ maksimum analit", "Bebas"], 1,
     "IK: start = λ tertinggi, end = λ terendah; pemindaian berlangsung dari λ panjang ke pendek."),
    ("Pembacaan menunjukkan A ≈ 2,8 di seluruh rentang segera setelah alat dinyalakan. Penyebab paling mungkin…", ["Lampu WI aktif", "Silika gel belum dikeluarkan dari kompartemen", "Kuvet kuarsa", "Konsentrasi standar terlalu rendah"], 1,
     "Kantong silika gel menghalangi berkas sampel sehingga I ≪ I₀."),
    ("Tombol pada UV-1700 untuk menyerahkan kendali ke UVProbe adalah…", ["F1", "F4 (PC Control)", "AUTO ZERO", "PRINT"], 1,
     "F4 = PC Control; MODE digunakan untuk keluar saat mematikan alat."),
    ("Autozero dilakukan dengan standar 2 mg/L (A = 0,40) di sel sampel. Pembacaan standar 4 mg/L akan menjadi sekitar…", ["0,80", "0,40", "0,00", "1,20"], 1,
     "Seluruh pembacaan dikurangi 0,40: 0,80 − 0,40 = 0,40."),
    ("Zero Interception sahih digunakan bila…", ["R² > 0,99", "Intersep model bebas tidak berbeda nyata dari nol", "Jumlah standar ≥ 3", "Selalu"], 1,
     "Uji: |b| ≤ t·s_b. Bila intersep nyata, pemaksaan melalui nol menghasilkan bias."),
    ("Galat relatif konsentrasi minimum (ΔT konstan) terjadi pada A ≈", ["0,10", "0,434", "1,0", "2,0"], 1,
     "Turunan galat relatif terhadap T bernilai nol pada T = 1/e, yaitu A = 0,434."),
    ("Penetapan Fe(II)–fenantrolin pada 510 nm sebaiknya memakai blanko…", ["Udara", "Akuades", "Blanko reagen", "Standar terendah"], 2,
     "Blanko reagen mengompensasi serapan hidroksilamin, fenantrolin, dan bufer."),
    ("R² = 0,9995 tetapi tanda residual berpola (−, −, +, +, +, −, −). Kesimpulan yang tepat…", ["Linear sempurna", "Indikasi kelengkungan; lakukan uji Mandel/periksa residual", "Intersep nol", "Harus menghapus titik tengah"], 1,
     "R² tidak membuktikan linearitas; pola residual sistematis menunjukkan ketidaksesuaian model."),
    ("Kuvet kuarsa dikeringkan setelah praktik dengan cara…", ["Dipanaskan di oven 105 °C", "Dibilas etanol lalu aseton, dibalik hingga kering", "Dilap tisu kering bagian bening", "Dibiarkan berisi akuades"], 1,
     "Sesuai IK bagian D."),
]


# =============================================================================
# BAGIAN 4 — GRAFIK LAPORAN & PDF
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

INK, ACC, ACC2, MUTED = "#13233f", "#b8860b", "#1f6f8b", "#5b6b82"


def fnum(x, d=4, dec="koma"):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
    except (TypeError, ValueError):
        return str(x)
    s = f"{float(x):.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace(".", ",") if dec == "koma" else s


def fsig(x, sig=4, dec="koma"):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    if x == 0:
        return fnum(0, sig - 1, dec)
    d = max(0, min(8, sig - 1 - int(np.floor(np.log10(abs(x))))))
    return fnum(x, d, dec)


def _style(ax, xl, yl):
    ax.set_xlabel(xl, color=MUTED)
    ax.set_ylabel(yl, color=MUTED)
    ax.grid(True, color="#e7ebf1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c3ccd9")
    ax.tick_params(colors=MUTED, labelsize=8)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def spectrum_png(spectra, peaks=None) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    pal = [ACC2, ACC, "#a23b72", "#2e7d32"]
    for i, s in enumerate(spectra):
        o = np.argsort(s["lam"])
        ax.plot(np.asarray(s["lam"])[o], np.asarray(s["A"])[o], lw=1.5, color=pal[i % 4], label=s["name"])
    for p in (peaks or [])[:3]:
        ax.axvline(p, color=ACC, ls="--", lw=0.9)
        ax.text(p, ax.get_ylim()[1] * 0.95, f" {p:.1f} nm".replace(".", ","), color=ACC, fontsize=8, va="top")
    _style(ax, "Panjang gelombang (nm)", "Absorbansi")
    ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def calib_png(fit: Fit, unit: str, samples=()) -> bytes:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    ax.scatter(fit.x, fit.y, color=ACC2, zorder=3, label="Standar")
    xs = np.array([0, fit.xmax * 1.05])
    ax.plot(xs, fit.a * xs + fit.b, color=ACC, lw=1.5, label=f"{fit.equation(4)}   R² = {fit.r2:.4f}".replace(".", ","))
    for name, cx, ay in samples:
        ax.scatter([cx], [ay], marker="D", color="#a23b72", zorder=4)
        ax.annotate(name, (cx, ay), textcoords="offset points", xytext=(6, 6), fontsize=8)
    _style(ax, "", "Absorbansi")
    ax.set_ylim(bottom=min(0, float(np.min(fit.y)) - 0.02))
    ax.legend(frameon=False, fontsize=8)
    ax2.axhline(0, color=ACC, lw=1)
    ax2.scatter(fit.x, fit.res, color="#a23b72", s=18)
    _style(ax2, f"Konsentrasi ({unit})", "Residual")
    return _png(fig)


def _font():
    try:
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_pdf(model: dict) -> bytes:
    """model: {'title','subtitle','blocks':[{'t':'h'|'p'|'note'|'kv'|'table'|'img', ...}], 'simulated': bool}"""
    F, FB = _font()
    st_ = {
        "h": ParagraphStyle("h", fontName=FB, fontSize=11.5, leading=14.5, textColor=colors.HexColor(INK), spaceBefore=9, spaceAfter=3),
        "p": ParagraphStyle("p", fontName=F, fontSize=9.3, leading=13.2, textColor=colors.HexColor("#16233a"), spaceAfter=4),
        "note": ParagraphStyle("n", fontName=F, fontSize=8.2, leading=11, textColor=colors.HexColor(MUTED)),
        "cell": ParagraphStyle("c", fontName=F, fontSize=8.2, leading=10.5),
        "cellb": ParagraphStyle("cb", fontName=FB, fontSize=7.9, leading=10, textColor=colors.HexColor("#2d3b55")),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=34 * mm, bottomMargin=18 * mm,
                            title=model.get("title", "Laporan"), author=model.get("author", "SPEKTRO-1700 Virtual Lab"))
    W = A4[0] - 36 * mm
    story = []
    if model.get("simulated"):
        t = Table([[Paragraph("<b>Mengandung data simulasi/contoh.</b> Hasil latihan, bukan hasil pengujian laboratorium.", st_["cell"])]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbf1d8")), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story += [t, Spacer(1, 6)]
    for b in model["blocks"]:
        k = b["t"]
        if k == "h":
            line = Table([[""]], colWidths=[W], rowHeights=[1.1])
            line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ACC))]))
            kt = KeepTogether([Paragraph(b["text"], st_["h"]), line, Spacer(1, 4)])
            story.append(kt)
        elif k == "p":
            story.append(Paragraph(b["text"], st_["p"]))
        elif k == "note":
            story.append(Paragraph(b["text"], st_["note"]))
        elif k == "kv":
            rows = [[Paragraph(str(a), st_["note"]), Paragraph(str(v), st_["cell"])] for a, v in b["rows"]]
            t = Table(rows, colWidths=[50 * mm, W - 50 * mm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story += [t, Spacer(1, 4)]
        elif k == "table":
            data = [[Paragraph(str(h), st_["cellb"]) for h in b["head"]]] + [[Paragraph(str(c), st_["cell"]) for c in r] for r in b["rows"]]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1ede1")), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe3ea")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 6)]
        elif k == "img":
            img = Image(io.BytesIO(b["png"]))
            ratio = img.imageHeight / img.imageWidth
            img.drawWidth, img.drawHeight = W, W * ratio
            story += [img, Paragraph(b.get("cap", ""), st_["note"]), Spacer(1, 6)]
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    def deco(c, d):
        c.saveState()
        c.setFillColor(colors.HexColor(INK))
        c.rect(0, A4[1] - 27 * mm, A4[0], 27 * mm, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#e9c46a"))
        c.setFont(F, 8)
        c.drawString(18 * mm, A4[1] - 10 * mm, "SPEKTRO-1700 VIRTUAL LAB  ·  Shimadzu UV-1700 PharmaSpec  ·  " + IK_META["kode"])
        c.setFillColor(colors.white)
        c.setFont(FB, 13.5)
        c.drawString(18 * mm, A4[1] - 17.5 * mm, model.get("title", "LAPORAN"))
        c.setFillColor(colors.HexColor("#c9d3e3"))
        c.setFont(F, 8.5)
        c.drawString(18 * mm, A4[1] - 23 * mm, model.get("subtitle", ""))
        c.setStrokeColor(colors.HexColor("#dfe3ea"))
        c.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        c.setFillColor(colors.HexColor(MUTED))
        c.setFont(F, 7.3)
        c.drawString(18 * mm, 9 * mm, f"Dibuat {stamp} · simulator edukatif, tidak terhubung ke instrumen")
        c.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Halaman {d.page}")
        c.restoreState()

    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


# =============================================================================
# BAGIAN 5 — MODEL 3D INSTRUMEN (Plotly Mesh3d; satuan cm, proporsi perkiraan)
# =============================================================================
BEIGE, BEIGE2, CREAM, GREY, DARK = "#dcc27a", "#e6d49b", "#efe8d3", "#5e646d", "#24282f"


def _box_vertices(x0, x1, y0, y1, z0, z1):
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)


_TRI_I = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
_TRI_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
_TRI_K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]


def _rot_x(v, pivot, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = v - pivot
    y = p[:, 1] * c - p[:, 2] * s
    z = p[:, 1] * s + p[:, 2] * c
    return np.column_stack([p[:, 0], y, z]) + pivot


def instrument_parts(lid_open: bool, exploded: bool, silika: bool, lam: float, lamp_switch: float, sample_color: str):
    """Daftar komponen: (komponen_info, grup, verts, warna, opasitas_dasar, pancaran)."""
    P = []
    up = 9.0 if exploded else 0.0

    def add(info, grp, box, col, op=1.0, emi=0.0, tf=None):
        v = _box_vertices(*box)
        if tf:
            v = tf(v)
        if grp == "atas":
            v = v + np.array([0, 0, up])
        P.append((info, grp, v, col, op, emi))

    # dasar & kaki (depan = y negatif)
    add("Sisi kiri & belakang", "rumah", (-22, 22, -24, 24, 0, 3), GREY)
    for x, y in [(-18, -20), (18, -20), (-18, 20), (18, 20)]:
        add("Sisi kiri & belakang", "rumah", (x - 1.8, x + 1.8, y - 1.8, y + 1.8, -1.6, 0), "#3b4048")
    # rumah instrumen (berlubang pada kompartemen x −16..−2, y −20..6)
    add("Sisi kanan", "rumah", (-2, 22, -24, 24, 3, 13.5), BEIGE)
    add("Sisi kiri & belakang", "rumah", (-22, -16, -24, 24, 3, 13.5), BEIGE)
    add("Kompartemen sampel", "rumah", (-16, -2, -24, -20, 3, 13.5), "#d3b768")
    add("Kompartemen sampel", "rumah", (-16, -2, 6, 24, 3, 13.5), BEIGE)
    # interior kompartemen
    add("Kompartemen sampel", "dalam_k", (-16, -2, -20, 6, 3, 4), DARK)
    add("Kompartemen sampel", "dalam_k", (-16, -15.6, -20, 6, 4, 13.5), "#2c3139")
    add("Kompartemen sampel", "dalam_k", (-2.4, -2, -20, 6, 4, 13.5), "#2c3139")
    for yb, nm in [(2.0, "ref"), (-10.0, "smp")]:
        add("Kompartemen sampel", "dalam_k", (-10.8, -7.2, yb - 1.8, yb + 1.8, 4, 4.6), "#15181d")
        add("Kompartemen sampel", "dalam_k", (-10.8, -7.2, yb + 1.0, yb + 1.8, 4.6, 11), "#15181d")
        add("Kompartemen sampel", "dalam_k", (-10.8, -7.2, yb - 1.8, yb - 1.0, 4.6, 11), "#15181d")
        add("Kompartemen sampel", "dalam_k", (-9.5, -8.5, yb - 0.5, yb + 0.5, 4.6, 12.5),
            "#d9f1f6" if nm == "ref" else sample_color, 0.55)
    if silika:
        add("Kompartemen sampel", "dalam_k", (-6.5, -3, -9, -3, 4, 5.4), "#f4f4f0")
    # penutup kompartemen (engsel di belakang, y = 6, z = 15)
    lid_tf = (lambda v: _rot_x(v, np.array([0, 6.0, 15.0]), -100)) if lid_open else None
    add("Kompartemen sampel", "atas", (-16.2, -1.8, -20.4, 6.2, 13.5, 15.2), BEIGE2, tf=lid_tf)
    # atap bagian lain
    add("Sisi kanan", "atas", (-2, 22, -24, 24, 13.5, 15), BEIGE2)
    add("Sisi kiri & belakang", "atas", (-22, -16, -24, 24, 13.5, 15), BEIGE2)
    add("Kompartemen sampel", "atas", (-16, -2, 6.2, 24, 13.5, 15), BEIGE2)
    for i in range(14):                                   # kisi ventilasi atas sisi belakang
        add("Sisi kanan", "atas", (-1 + i * 1.6, -0.3 + i * 1.6, 12, 23, 15, 15.25), "#bfa45d")
    # panel tombol & layar
    add("Panel tombol & LCD", "atas", (3, 21, -23, -12, 15, 16), CREAM)
    for r in range(4):
        for c in range(5):
            add("Panel tombol & LCD", "atas", (4 + c * 3.4, 6 + c * 3.4, -22 + r * 2.4, -20.8 + r * 2.4, 16, 16.4),
                "#6f8fbf" if c == 4 else "#c8ccd3")
    tilt = lambda v: _rot_x(v, np.array([0, -9.0, 15.0]), -18)  # noqa: E731
    add("Panel tombol & LCD", "atas", (4, 20, -11, -8.6, 15, 27), CREAM, tf=tilt)
    add("Panel tombol & LCD", "atas", (6, 18, -11.2, -11.0, 17.5, 25.5), "#9ff0ea", emi=0.4, tf=tilt)
    # sisi kanan: ventilasi lampu, IC card, contrast
    wi_on = lam >= lamp_switch
    add("Lampu WI", "rumah", (22, 22.3, 10, 21, 5, 11.5), "#e08a3c" if wi_on else "#bfa45d", emi=0.5 if wi_on else 0)
    add("Sisi kanan", "rumah", (22, 22.3, -16, -6, 11, 12), "#2b2f36")
    add("Sisi kanan", "rumah", (22, 22.4, -4, -2, 10.8, 12.2), "#d9d2bf")
    # sisi kiri: saklar & kipas
    add("Sisi kiri & belakang", "rumah", (-22.3, -22, -2, 3, 6, 9.5), "#f4f4f4")
    add("Sisi kiri & belakang", "rumah", (-22.3, -22, 8, 20, 4.5, 12), "#3c4149")
    # bangku optik
    add("Lampu D₂", "dalam", (12, 21.5, 8, 22.5, 3, 4), "#3a4048")
    add("Lampu D₂", "dalam", (17, 19, 18, 20, 4, 11.5), "#e2dbff" if not wi_on else "#8f8aa8", emi=0.6 if not wi_on else 0)
    add("Lampu WI", "dalam", (17, 19, 11.5, 13.5, 4, 10.5), "#ffe39a" if wi_on else "#a69a72", emi=0.6 if wi_on else 0)
    add("Cermin pemilih sumber", "dalam", (14, 14.6, 14, 18, 6.5, 10.5), "#cfd8e6")
    add("Celah masuk/keluar", "dalam", (11, 11.4, 14, 18, 5, 12), DARK)
    add("Kisi difraksi", "dalam", (-1, 10, 9.5, 23, 3, 4), "#2c3138")
    add("Kisi difraksi", "dalam", (2, 6, 20.5, 21.6, 5, 11.5), "#a9bad6")
    add("Celah masuk/keluar", "dalam", (2, 5, 9, 9.4, 5, 12), DARK)
    add("Pembagi berkas", "dalam", (2.8, 4.2, 0.3, 1.7, 6, 11.5), "#7fe3f2", 0.7)
    add("Pembagi berkas", "dalam", (2.8, 4.2, -10.7, -9.3, 6, 11.5), "#cfd8e6")
    add("Detektor", "dalam", (-20.5, -18, 0.5, 3.5, 6, 11), "#2a3346")
    add("Detektor", "dalam", (-20.5, -18, -11.5, -8.5, 6, 11), "#2a3346")
    add("Sisi kiri & belakang", "dalam", (-21, -17, 9, 22, 3, 3.6), "#1d6b45")
    return P


def beam_segments(lam: float, lamp_switch: float):
    zb = 8.5
    src = (18, 19, zb) if lam < lamp_switch else (18, 12.5, zb)
    pts = [src, (14.3, 16, zb), (11.2, 16, zb), (4, 21, zb), (3.5, 9.2, zb), (3.5, 1, zb)]
    poly = [(pts[i], pts[i + 1], "#fff6d6") for i in range(3)]
    mono = [(pts[3], pts[4], lam_color(lam)), (pts[4], pts[5], lam_color(lam))]
    split = [((3.5, 1, zb), (-18, 1, zb), lam_color(lam)), ((3.5, 1, zb), (3.5, -10, zb), lam_color(lam)),
             ((3.5, -10, zb), (-18, -10, zb), lam_color(lam))]
    return poly + mono + split


# =============================================================================
# BAGIAN 6 — APLIKASI STREAMLIT
# =============================================================================
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="SPEKTRO-1700 Virtual Lab", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important; letter-spacing: .005em; }
section[data-testid="stSidebar"] { background: #13233f; }
section[data-testid="stSidebar"] * { color: #e3e9f3; }
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] input { color: #13233f !important; }
.k-hero { background: linear-gradient(120deg,#13233f 0%,#1d3a63 60%,#6b5a2a 140%); color:#eef2f8; border-radius:14px; padding:22px 26px; }
.k-hero h1 { color:#fff; margin:2px 0 8px; }
.k-eyebrow { font:500 .72rem 'IBM Plex Mono', monospace; letter-spacing:.14em; color:#e9c46a; text-transform:uppercase; }
.k-chip { display:inline-block; font:500 .72rem 'IBM Plex Mono', monospace; padding:3px 9px; border-radius:99px; margin:2px 4px 2px 0; background:#e8edf5; color:#2a4870; }
.k-chip.sim { background:#fbf1d8; color:#8a5a00; } .k-chip.ok { background:#dff2e7; color:#18794e; } .k-chip.bad { background:#fbe5e3; color:#b42318; }
.k-lcd { background:#a9ece6; color:#3a3a3a; border-radius:4px; padding:8px 14px 10px; font-family:'IBM Plex Mono', monospace;
  box-shadow: inset 0 0 0 4px #d8d1b9, inset 0 0 0 6px #9b9788, inset 0 2px 16px rgba(0,0,0,.18); min-height:168px; }
.k-lcd .title { background:#e98c9b; color:#fff; display:inline-block; padding:0 8px; font-size:.72rem; margin-bottom:6px; }
.k-lcd .row { display:flex; align-items:baseline; gap:12px; margin:4px 0; }
.k-lcd .tag { background:#e98c9b; color:#fff; font-size:.68rem; padding:0 6px; min-width:56px; }
.k-lcd .big { font-size:1.55rem; letter-spacing:.12em; margin-left:auto; color:#2f2f2f; }
.k-lcd .buf { font-size:.8rem; color:#0b5e66; border-top:1px dashed rgba(0,0,0,.25); margin-top:6px; padding-top:3px; }
.k-eq { font:500 1.04rem 'IBM Plex Mono', monospace; background:#f3f0e6; border-left:3px solid #b8860b; border-radius:6px; padding:10px 12px; }
.k-banner { background:#fbf1d8; border:1px solid #edd08f; border-radius:8px; padding:9px 12px; font-size:.86rem; }
.k-step { border-left:3px solid #c9d3e3; padding:2px 0 2px 10px; margin:4px 0; }
.k-step.done { border-color:#18794e; } .k-step.now { border-color:#b8860b; background:#fcf7ea; }
</style>
""", unsafe_allow_html=True)

PAL = ["#1f6f8b", "#b8860b", "#a23b72", "#2e7d32", "#5c4fb3", "#d1495b"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _defaults() -> dict:
    return {
        "ui": {"dec": "koma", "kelas": "", "kelompok": "", "tanggal": date.today().isoformat(), "mode": "latihan"},
        "crit": {"r2_min": 0.995, "rsd_max": 2.0, "rec_min": 90.0, "rec_max": 110.0, "sres_max": 2.0, "a_min": 0.1, "a_max": 1.0},
        "hw": {"kabel": False, "ups": False, "silika": True, "monitor_alat": False, "power": False, "selfcheck": False,
               "selfcheck_warn": False, "t_on": None, "skip": 0.0, "pc_control": False, "pc_on": False, "uvprobe": False,
               "connected": False, "lid_open": False, "lamp_switch": 340.8},
        "cfg": {"analit": "fe", "kuvet": "kuarsa", "b": 1.0, "blanko": "reagen", "matriks": 0.0},
        "cell": {"ref": {"kind": "blanko"}, "smp": {"kind": "blanko"}, "label": "Blanko"},
        "lcd": {"lam": 550.0, "A": 0.0, "buf": "", "await": None},
        "zero": None, "baseline": None,
        "sp": {"start": 700.0, "end": 400.0, "interval": 1.0, "speed": "Medium", "mode": "Abs", "thr": 0.01},
        "spectra": [], "active": None,
        "pho": None,
        "std": [], "std_ver": 0, "smp": [], "src": "simulasi",
        "valid": {"reps": "", "kind": "konsentrasi", "level": "", "level_unit": "mg/L", "rec_mode": "spike", "cs": "", "css": "", "cadd": "", "cref": ""},
        "ts": {"data": [], "dur": 60.0, "dt": 1.0, "tol": 1.0},
        "events": [], "logbook": [], "off": {k: False for k, _ in IK_MATIKAN},
        "report": {"judul": "Penetapan Kadar secara Spektrofotometri UV-Vis", "praktikan": "", "nim": "", "dosen": "", "sampel": "", "catatan": ""},
        "seed": int(np.random.default_rng().integers(1, 999_999)), "counter": 1, "quiz": {},
    }


if "S" not in st.session_state:
    st.session_state.S = _defaults()
S = st.session_state.S
HW, CFG, CR = S["hw"], S["cfg"], S["crit"]


def DEC():
    return S["ui"]["dec"]


def f(x, d=4):
    return fnum(x, d, DEC())


def g(x, s=4):
    return fsig(x, s, DEC())


def cf(x):
    """Angka ringkas tanpa nol berlebih (label)."""
    try:
        t = f"{float(x):g}"
    except (TypeError, ValueError):
        return "—"
    return t.replace(".", ",") if DEC() == "koma" else t


def nid():
    S["counter"] += 1
    return S["counter"]


def rng():
    return np.random.default_rng(S["seed"] * 7919 + nid())


def event(msg):
    S["events"].insert(0, f"{datetime.now():%H.%M.%S}  {msg}")
    S["events"] = S["events"][:80]


def num(text, default=NAN):
    v = parse_number(text)
    return default if math.isnan(v) else v


def AN():
    return ANALIT[CFG["analit"]]


def ready() -> bool:
    return HW["power"] and HW["selfcheck"]


def t_warm() -> float:
    if not HW["power"] or HW["t_on"] is None:
        return 0.0
    return (time.time() - HW["t_on"]) / 60.0 + HW["skip"]


def file_stub() -> str:
    d = datetime.fromisoformat(S["ui"]["tanggal"]).strftime("%d%m%Y")
    k = re.sub(r"\W+", "", S["ui"]["kelas"]) or "Kelas"
    gp = re.sub(r"\W+", "", S["ui"]["kelompok"]) or "Kel"
    return f"{d}_{k}_{gp}"


def cfg_now() -> dict:
    return {**CFG, "silika": HW["silika"], "lid_open": HW["lid_open"], "lamp_switch": HW["lamp_switch"]}


def zero_curve(lam):
    """Kurva koreksi: autozero (pada λ-nya) diprioritaskan, lalu baseline (dalam rentangnya)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    out = np.full(lam.size, np.nan)
    b = S["baseline"]
    if b and b["key"] == CFG["analit"] and b["kuvet"] == CFG["kuvet"] and b["b"] == CFG["b"]:
        m = (lam >= b["lo"] - 1e-9) & (lam <= b["hi"] + 1e-9)
        out[m] = np.interp(lam[m], b["lam"], b["A"])
    z = S["zero"]
    if z and z["key"] == CFG["analit"] and z["kuvet"] == CFG["kuvet"] and z["b"] == CFG["b"] and (not b or z["t"] >= b["t"]):
        out[np.abs(lam - z["lam"]) < 0.05] = z["A"]
    return out


def read(lam, speed="Slow", content=None):
    """Pembacaan terkoreksi (A) seperti yang ditampilkan alat."""
    lam = np.atleast_1d(np.asarray(lam, float))
    smp = content or S["cell"]["smp"]
    r = raw_reading(lam, CFG["analit"], smp, S["cell"]["ref"], cfg_now(), t_warm(), rng(), speed)
    zc = zero_curve(lam)
    corrected = np.isfinite(zc)
    A = r["A"] + r["noise"] - np.where(corrected, zc, 0.0)
    return {"A": np.clip(A, -4.0, 4.0), "corrected": corrected, "low_energy": r["low_energy"]}


def do_zero(lam):
    r = raw_reading(np.atleast_1d(lam), CFG["analit"], S["cell"]["smp"], S["cell"]["ref"], cfg_now(), t_warm(), rng(), "Slow")
    S["zero"] = {"lam": float(lam), "A": float(r["A"][0] + 0.3 * r["noise"][0]), "key": CFG["analit"], "kuvet": CFG["kuvet"],
                 "b": CFG["b"], "t": time.time(), "isi": S["cell"]["label"]}
    S["lcd"]["A"] = 0.0
    event(f"AUTO ZERO pada {f(lam, 1)} nm (sel sampel: {S['cell']['label']}).")
    if S["cell"]["smp"]["kind"] != "blanko":
        event("PERINGATAN: autozero tidak dilakukan dengan blanko; seluruh pembacaan bergeser.")


def set_cell(kind, conc=0.0, label="", matriks=0.0):
    S["cell"]["smp"] = {"kind": kind, "conc": conc, "matriks": matriks}
    S["cell"]["label"] = label or kind.capitalize()


def lcd_html(title=None):
    on = ready()
    lam = S["lcd"]["lam"]
    A = S["lcd"]["A"]
    data = ("OVER" if A > 3.999 else f"{A:.3f}") if on else "-----"
    light = ("D2" if lam < HW["lamp_switch"] else "WI") if on else "--"
    title = title or ("PC Control" if HW["pc_control"] else "Photometric")
    buf = S["lcd"]["buf"]
    aw = S["lcd"]["await"]
    footer = f"<div class='buf'>{'GOTO WL: ' if aw == 'wl' else 'Input: '}{buf or '_'}</div>" if (aw or buf) else ""
    if not HW["power"]:
        return "<div class='k-lcd' style='background:#9aa39f'><div class='buf' style='border:0'>Daya mati</div></div>"
    if not HW["selfcheck"]:
        return "<div class='k-lcd'><div class='title'>Initialize</div><div class='row'>Pengecekan otomatis…</div></div>"
    return (f"<div class='k-lcd'><div class='title'>{title}</div>"
            f"<div class='row'><span class='tag'>λ :</span><span class='big'>{lam:.1f}nm</span></div>"
            f"<div class='row'><span class='tag'>Data :</span><span class='big'>{data} ABS</span></div>"
            f"<div class='row'><span class='tag'>Light :</span><span>{light}</span></div>{footer}</div>")


def fig_base(xt, yt, height=380):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=20, t=30, b=50),
                      font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#13233f"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified",
                      separators=",." if DEC() == "koma" else ".,")
    fig.update_xaxes(title_text=xt, gridcolor="#e8ecf2", zeroline=False)
    fig.update_yaxes(title_text=yt, gridcolor="#e8ecf2", zeroline=False)
    return fig


def chip(text, cls=""):
    return f"<span class='k-chip {cls}'>{text}</span>"


def src_chip(src):
    lab = {"simulasi": "DATA SIMULASI", "contoh": "DATA CONTOH", "pengguna": "DATA PENGGUNA", "impor": "IMPOR CSV", "campuran": "CAMPURAN"}.get(src, src.upper())
    return chip(lab, "sim" if src in ("simulasi", "contoh", "campuran") else "ok")


def pdf_button(col, label, key, builder, fname):
    """PDF dibuat hanya saat diminta (menghindari render ulang setiap interaksi)."""
    if col.button(label, key=f"pb_{key}"):
        with st.spinner("Menyiapkan PDF…"):
            S["_pdf_" + key] = builder()
    if S.get("_pdf_" + key):
        col.download_button("Unduh " + fname, S["_pdf_" + key], fname, "application/pdf", key=f"pd_{key}")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";" if DEC() == "koma" else ",", decimal="," if DEC() == "koma" else ".").encode("utf-8-sig")


def std_df() -> pd.DataFrame:
    rows = S["std"]
    return pd.DataFrame({"ID Sampel": pd.Series([r.get("ID Sampel", "") for r in rows], dtype=object),
                         "Konsentrasi": pd.Series([r.get("Konsentrasi") for r in rows], dtype=float),
                         "Abs": pd.Series([r.get("Abs") for r in rows], dtype=float),
                         "Pakai": pd.Series([bool(r.get("Pakai", True)) for r in rows], dtype=bool)})


def set_std(rows, src=None):
    S["std"] = rows
    S["std_ver"] += 1
    if src:
        S["src"] = src


def std_points():
    df = std_df()
    if df.empty:
        return df
    df = df[df["Konsentrasi"].notna() & df["Abs"].notna() & df["Pakai"]]
    return df


def current_fit(zero=None):
    p = std_points()
    if len(p) < 2:
        return None
    z = (S["pho"] or {}).get("zero", False) if zero is None else zero
    return fit_line(p["Konsentrasi"].values, p["Abs"].values, z)


def unit():
    return (S["pho"] or {}).get("unit", "mg/L")


def sample_fp(s):
    return dilution_factor([(r.get("V pipet (mL)"), r.get("V labu (mL)")) for r in s.get("steps", [])]) if s.get("steps") else 1.0


def sample_res(s, fit):
    try:
        reps = parse_list(s["reps"])
    except ValueError:
        return None
    if not fit or not fit.ok or not reps:
        return None
    return sample_result(fit, reps, fp=sample_fp(s), mass_g=num(s["mass"], None) if s.get("mass") else None,
                         vol_ml=num(s["vol"], None) if s.get("vol") else None, unit=unit(),
                         a_min=CR["a_min"], a_max=CR["a_max"], rsd_max=CR["rsd_max"])


def new_sample(name, reps="", src="pengguna", hidden=None, matriks=0.0):
    return {"id": nid(), "v": 0, "name": name, "reps": reps, "src": src, "steps": [], "mass": "", "vol": "", "hidden": hidden, "matriks": matriks}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Beranda", "Model 3D Instrumen", "1 · Persiapan Alat", "2 · Panel UV-1700", "3 · Mode Spektrum (λmaks)",
         "4 · Mode Fotometrik", "5 · Evaluasi Kalibrasi", "6 · Validasi Metode", "7 · Time Scan (Kestabilan Warna)",
         "8 · Mematikan Alat & Log Book", "Pembelajaran & Kuis", "Laporan PDF", "Proyek & Data"]

with st.sidebar:
    st.markdown("### SPEKTRO-1700\n<span style='color:#9fb0cc;font-size:.82rem'>Virtual Lab · Shimadzu UV-1700 PharmaSpec</span>", unsafe_allow_html=True)
    page = st.radio("Menu", PAGES, label_visibility="collapsed")
    st.divider()
    S["ui"]["kelas"] = st.text_input("Kelas", S["ui"]["kelas"], placeholder="mis. 2A")
    S["ui"]["kelompok"] = st.text_input("Kelompok", S["ui"]["kelompok"], placeholder="mis. Kel4")
    S["ui"]["tanggal"] = st.date_input("Tanggal praktik", date.fromisoformat(S["ui"]["tanggal"])).isoformat()
    st.caption(f"Nama berkas UVProbe: **{file_stub()}**")
    S["ui"]["dec"] = st.radio("Pemisah desimal", ["koma", "titik"], index=0 if DEC() == "koma" else 1, horizontal=True,
                              format_func=lambda d: "Koma (0,512)" if d == "koma" else "Titik (0.512)")
    S["ui"]["mode"] = st.radio("Mode kerja", ["latihan", "analisis"], index=0 if S["ui"]["mode"] == "latihan" else 1,
                               format_func=lambda m: "Latihan (simulasi)" if m == "latihan" else "Analisis (data alat nyata)")
    st.caption(f"Acuan: {IK_META['kode']} · berlaku {IK_META['berlaku']}. Simulator edukatif; tidak terhubung ke alat.")

tw = t_warm()
status = [chip("SIMULASI", "sim"),
          chip("UV-1700 ON" if HW["power"] else "UV-1700 OFF", "ok" if HW["power"] else "bad"),
          chip(("Pemanasan " + f(min(tw, 15), 1) + "/15 menit") if HW["power"] and tw < 15 else ("Siap ukur" if ready() else "Belum siap"),
               "ok" if ready() and tw >= 15 else "sim"),
          chip("PC Control" if HW["pc_control"] else "Standalone"),
          chip("UVProbe Connected" if HW["connected"] else "UVProbe —", "ok" if HW["connected"] else ""),
          chip("Silika gel di kompartemen", "bad") if HW["silika"] else "",
          chip("Tutup kompartemen TERBUKA", "bad") if HW["lid_open"] else ""]
st.markdown(" ".join(s for s in status if s), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Komponen UI bersama
# ---------------------------------------------------------------------------
def compartment_panel(where: str):
    """Isi kompartemen & pengaturan larutan (dipakai beberapa halaman)."""
    with st.container(border=True):
        st.markdown("**Kompartemen sampel**")
        keys = list(ANALIT)
        new = st.selectbox(f"Analit / metode ({where})", keys, index=keys.index(CFG["analit"]),
                           format_func=lambda k: f"{ANALIT[k]['nama']} · λref {ANALIT[k]['lam']} nm")
        if new != CFG["analit"]:
            CFG["analit"] = new
            a = ANALIT[new]
            S["sp"]["start"], S["sp"]["end"] = (350.0, 200.0) if a["lam"] < 320 else (float(min(1100, round(a["lam"] + 180, -1))), float(max(320, round(a["lam"] - 160, -1))))
            S["lcd"]["lam"] = float(a["lam"])
            set_cell("blanko", label="Blanko")
            event(f"Metode diganti: {a['nama']}.")
            st.rerun()
        c1, c2 = st.columns(2)
        kv = list(KUVET)
        CFG["kuvet"] = c1.selectbox(f"Kuvet ({where})", kv, index=kv.index(CFG["kuvet"]),
                                    format_func=lambda k: f"{KUVET[k]['nama']} ≥{KUVET[k]['cutoff']} nm")
        CFG["b"] = c2.select_slider(f"Tebal kuvet b, cm ({where})", options=[0.5, 1.0, 2.0, 5.0], value=CFG["b"])
        bl = ["reagen", "pelarut"]
        CFG["blanko"] = c1.selectbox(f"Jenis blanko ({where})", bl, index=bl.index(CFG["blanko"]),
                                     format_func=lambda b: "Blanko reagen" if b == "reagen" else "Pelarut/akuades")
        ref_opts = ["blanko", "kosong"]
        rk = c2.selectbox(f"Sel referensi ({where})", ref_opts, index=ref_opts.index(S["cell"]["ref"]["kind"]),
                          format_func=lambda k: "Blanko" if k == "blanko" else "Kosong (udara)")
        S["cell"]["ref"] = {"kind": rk}
        a = AN()
        opts = ["Blanko"] + [f"Standar {cf(c)} {a['unit']}" for c in a["std"]] + [s["name"] for s in S["smp"] if s.get("hidden") is not None] + ["Kosong (udara)"]
        cur = S["cell"]["label"] if S["cell"]["label"] in opts else "Blanko"
        pick = st.selectbox(f"Isi sel sampel ({where})", opts, index=opts.index(cur))
        if pick != S["cell"]["label"]:
            if pick == "Blanko":
                set_cell("blanko", label=pick)
            elif pick.startswith("Standar"):
                c = a["std"][opts.index(pick) - 1]
                set_cell("standar", prep_conc(c, rng()), pick)
            elif pick.startswith("Kosong"):
                set_cell("kosong", label=pick)
            else:
                s = next(s for s in S["smp"] if s["name"] == pick)
                set_cell("sampel", s["hidden"], pick, s.get("matriks", 0.0))
            event(f"Sel sampel diisi: {pick}.")
        HW["lid_open"] = st.toggle(f"Tutup kompartemen terbuka ({where})", HW["lid_open"])
        if HW["silika"]:
            st.error("Silika gel masih di kompartemen. Keluarkan sebelum mengukur (Persiapan langkah 3).")
            if st.button(f"Keluarkan silika gel ({where})"):
                HW["silika"] = False
                event("Silika gel dikeluarkan dari kompartemen.")
                st.rerun()
        if a["reagen"] and CFG["blanko"] != "reagen":
            st.warning("Metode kolorimetri: blanko pelarut tidak mengompensasi serapan reagen.")
        st.caption(a["catatan"])


def need_ready(what="pengukuran"):
    if not HW["power"]:
        st.info(f"UV-1700 belum menyala. Lakukan **1 · Persiapan Alat** sebelum {what}.")
        return False
    if not HW["selfcheck"]:
        st.info("Pengecekan otomatis belum selesai.")
        return False
    if t_warm() < 15:
        st.warning(f"Pemanasan baru {f(t_warm(), 1)} menit dari 15 menit; drift masih tinggi (pengukuran tetap dapat dilakukan).")
    return True


def need_uvprobe():
    if not (HW["pc_control"] and HW["connected"]):
        st.info("Modul ini meniru UVProbe. Tekan **F4 (PC Control)** di panel alat lalu **Connect** di halaman Persiapan.")
        return False
    return True


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
def page_home():
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        st.markdown(f"""<div class="k-hero"><div class="k-eyebrow">{IK_META['lab']}</div>
        <h1>SPEKTRO-1700 Virtual Lab</h1>
        <div>Simulator dan pengolah data spektrofotometer UV-Vis <b>Shimadzu UV-1700 PharmaSpec</b> yang mengikuti
        instruksi kerja <b>{IK_META['kode']}</b>: persiapan alat, pencarian λmaks, analisis fotometrik multi-point,
        evaluasi statistik kurva kalibrasi, validasi, hingga prosedur mematikan alat dan log book.</div><br>
        {chip('MODE SIMULASI', 'sim')}{chip('Double-beam · 190–1100 nm · 1 nm')}</div>""", unsafe_allow_html=True)
        st.write("")
        fit = current_fit()
        m = st.columns(4)
        m[0].metric("Spektrum", len(S["spectra"]))
        m[1].metric("Standar terbaca", len(std_points()))
        m[2].metric("R²", f(fit.r2, 4) if fit and fit.ok else "—")
        m[3].metric("Sampel", len(S["smp"]))
        st.markdown("**Alur kerja sesuai IK**")
        flow = [("Persiapan", HW["connected"]), ("λmaks", bool(S["spectra"])), ("Fotometrik", len(std_points()) >= 3),
                ("Evaluasi", bool(fit and fit.ok)), ("Matikan & log book", all(S["off"].values()))]
        st.markdown(" → ".join(chip(("✓ " if ok else "") + n, "ok" if ok else "") for n, ok in flow), unsafe_allow_html=True)
    with c2:
        st.plotly_chart(model3d_fig(lid=False, xray=False, labels=False, beam=False, height=420), key="home3d")
        st.caption("Model 3D skematis dengan tata letak mengikuti unit di laboratorium; bukan gambar teknik pabrikan.")
    with st.expander("Spesifikasi ringkas UV-1700 (indikatif)"):
        st.table(pd.DataFrame(list(UV1700.items()), columns=["Parameter", "Keterangan"]))


def model3d_fig(lid=False, xray=True, labels=False, beam=True, exploded=False, focus=None, height=600, cam="iso"):
    lam = S["lcd"]["lam"]
    smp_col = "#d9f1f6"
    if S["cell"]["smp"]["kind"] in ("standar", "sampel"):
        smp_col = AN()["warna"]
    parts = instrument_parts(lid, exploded, HW["silika"], lam, HW["lamp_switch"], smp_col)
    fig = go.Figure()
    centers = {}
    for info, grp, v, col, op, emi in parts:
        o = op
        if xray and grp in ("rumah", "atas"):
            o = op * 0.13
        if grp == "dalam" and not (xray or exploded):
            continue
        if grp == "dalam_k" and not (xray or exploded or lid):
            continue
        if focus and info != focus:
            o = o * 0.25
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=_TRI_I, j=_TRI_J, k=_TRI_K, color=col, opacity=o,
                                flatshading=True, hoverinfo="text", hovertext=info, name=info, showscale=False,
                                lighting=dict(ambient=0.55 + emi, diffuse=0.75, specular=0.15, roughness=0.8),
                                lightposition=dict(x=60, y=-80, z=120)))
        centers.setdefault(info, []).append(v.mean(axis=0))
    if beam and (xray or lid or exploded):
        for a, b, c in beam_segments(lam, HW["lamp_switch"]):
            if not (xray or exploded) and min(a[0], b[0]) >= -2:
                continue                                  # hanya ruas di dalam kompartemen yang terlihat saat tutup dibuka
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                                       line=dict(color=c, width=7), hoverinfo="skip", showlegend=False))
    if labels:
        xs, ys, zs, tx = [], [], [], []
        for k, vs in centers.items():
            c = np.mean(vs, axis=0)
            xs.append(c[0]); ys.append(c[1]); zs.append(c[2] + 6); tx.append(k)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="text+markers", text=tx, marker=dict(size=3, color="#b8860b"),
                                   textfont=dict(size=11, color="#13233f"), hoverinfo="skip", showlegend=False))
    eyes = {"iso": dict(x=-1.5, y=-1.7, z=1.1), "depan": dict(x=0, y=-2.4, z=0.5), "kanan": dict(x=2.4, y=0, z=0.6),
            "kiri": dict(x=-2.4, y=0, z=0.6), "belakang": dict(x=0, y=2.4, z=0.6), "atas": dict(x=0.01, y=-0.01, z=2.6)}
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 aspectmode="data", camera=dict(eye=eyes.get(cam, eyes["iso"]))))
    return fig


def page_model3d():
    c1, c2 = st.columns([2.3, 1], gap="large")
    with c2:
        with st.container(border=True):
            cam = st.selectbox("Sudut pandang", ["iso", "depan", "kanan", "kiri", "belakang", "atas"],
                               format_func=lambda c: {"iso": "Isometrik", "depan": "Depan", "kanan": "Sisi kanan (lampu, IC card)",
                                                      "kiri": "Sisi kiri (POWER)", "belakang": "Belakang", "atas": "Atas"}[c])
            lid = st.toggle("Buka tutup kompartemen", False)
            xray = st.toggle("Tembus pandang", True)
            expl = st.toggle("Angkat penutup (terurai)", False)
            beam = st.toggle("Jalur cahaya", True)
            labels = st.toggle("Label komponen", False)
            S["lcd"]["lam"] = float(st.slider("λ instrumen (nm)", 190, 1100, int(S["lcd"]["lam"])))
            lam = S["lcd"]["lam"]
            st.caption(f"Lampu aktif: **{'D₂' if lam < HW['lamp_switch'] else 'WI'}** · {('UV' if lam < 400 else 'tampak' if lam <= 780 else 'NIR')} · "
                       f"warna komplementer larutan yang menyerap: {complementary_hint(lam)}")
            focus = st.selectbox("Sorot komponen", ["—"] + list(KOMPONEN))
        if focus != "—":
            st.info(f"**{focus}.** {KOMPONEN[focus]}")
    with c1:
        st.plotly_chart(model3d_fig(lid, xray, labels, beam, expl, None if focus == "—" else focus, 620, cam), key="m3d")
        st.caption("Seret untuk memutar, gulir/cubit untuk zoom; arahkan kursor ke bagian alat untuk nama komponen. "
                   "Berkas: putih = polikromatis sebelum kisi; berwarna = monokromatis sesuai λ. Proporsi perkiraan.")
    with st.expander("Fungsi seluruh komponen"):
        for k, v in KOMPONEN.items():
            st.markdown(f"**{k}.** {v}")


def page_prep():
    st.markdown(f"Mengikuti **{IK_META['kode']} bagian A (Persiapan)**. Setiap langkah harus diselesaikan berurutan.")
    left, right = st.columns([1.25, 1], gap="large")
    done = {"kabel": HW["kabel"], "ups": HW["ups"], "silika": (not HW["silika"]) and HW["monitor_alat"],
            "power": HW["power"] and HW["selfcheck"], "warmup": HW["power"] and t_warm() >= 15, "pc": HW["connected"]}
    cur = next((k for k, _ in IK_PERSIAPAN if not done[k]), None)
    with left:
        for i, (k, txt) in enumerate(IK_PERSIAPAN):
            cls = "done" if done[k] else ("now" if k == cur else "")
            st.markdown(f"<div class='k-step {cls}'><b>A.{i + 1}</b> {txt}</div>", unsafe_allow_html=True)
        st.divider()
        if cur == "kabel":
            if st.button("Hubungkan kabel ke sumber listrik", type="primary"):
                HW["kabel"] = True; event("Kabel terminal utama terhubung."); st.rerun()
        elif cur == "ups":
            if st.button("Tekan POWER UPS", type="primary"):
                HW["ups"] = True; event("UPS menyala."); st.rerun()
        elif cur == "silika":
            a, b = st.columns(2)
            if HW["silika"] and a.button("Keluarkan silika gel", type="primary"):
                HW["silika"] = False; event("Silika gel dikeluarkan."); st.rerun()
            if not HW["monitor_alat"] and b.button("Buka (angkat) layar UV-1700"):
                HW["monitor_alat"] = True; event("Layar alat dibuka."); st.rerun()
            if HW["silika"]:
                st.caption("Anda juga dapat melewati langkah ini untuk melihat akibatnya pada pengecekan energi berkas.")
                if st.button("Lewati dan nyalakan alat"):
                    HW["monitor_alat"] = True; HW["power"] = True; HW["t_on"] = time.time(); st.rerun()
        elif cur == "power":
            if not HW["power"] and st.button("Tekan POWER (sisi kiri UV-1700)", type="primary"):
                HW["power"] = True; HW["t_on"] = time.time(); HW["skip"] = 0.0; event("UV-1700 dinyalakan."); st.rerun()
            if HW["power"] and not HW["selfcheck"]:
                with st.status("Pengecekan otomatis UV-1700…", expanded=True) as stt:
                    warn = False
                    for it in SELF_CHECK:
                        time.sleep(0.2)
                        if it == "Energi berkas" and HW["silika"]:
                            st.write("⚠ " + it + " — energi berkas sampel rendah (periksa kompartemen)")
                            warn = True
                        else:
                            st.write("✓ " + it)
                    stt.update(label="Pengecekan selesai" + (" dengan peringatan" if warn else ""), state="error" if warn else "complete")
                HW["selfcheck"] = True; HW["selfcheck_warn"] = warn
                event("Pengecekan otomatis selesai" + (" (peringatan energi berkas)." if warn else "."))
                st.rerun()
        elif cur == "warmup":
            st.info(f"Pemanasan: {f(t_warm(), 1)} dari 15 menit.")
            st.progress(min(1.0, t_warm() / 15))
            a, b = st.columns(2)
            if a.button("Percepat waktu simulasi +5 menit"):
                HW["skip"] += 5; st.rerun()
            if b.button("Perbarui"):
                st.rerun()
        elif cur == "pc":
            a, b, c = st.columns(3)
            if not HW["pc_control"] and a.button("Tekan F4 (PC Control)", type="primary"):
                HW["pc_control"] = True; event("F4: PC Control."); st.rerun()
            if not HW["pc_on"] and b.button("Nyalakan CPU, monitor, printer"):
                HW["pc_on"] = True; event("Komputer dan printer menyala."); st.rerun()
            if HW["pc_on"] and not HW["uvprobe"] and c.button("Buka UVProbe (klik 2×)"):
                HW["uvprobe"] = True; event("UVProbe dibuka."); st.rerun()
            if HW["uvprobe"] and st.button("Connect", type="primary", disabled=not HW["pc_control"]):
                HW["connected"] = True; event("UVProbe terhubung ke UV-1700."); st.rerun()
            if HW["uvprobe"] and not HW["pc_control"]:
                st.warning("Connect gagal: alat belum dalam mode PC Control (tekan F4).")
        else:
            st.success("Persiapan selesai: alat siap dikendalikan dari UVProbe.")
        if HW["selfcheck_warn"] and HW["silika"]:
            st.error("Silika gel masih di kompartemen; pembacaan akan tidak bermakna.")
            if st.button("Keluarkan silika gel sekarang"):
                HW["silika"] = False; st.rerun()
    with right:
        st.markdown(lcd_html(), unsafe_allow_html=True)
        st.write("")
        with st.expander("Mengapa pemanasan 15 menit?"):
            tt = np.linspace(0, 40, 161)
            d = 0.25 * np.array([drift_curve(t) for t in tt])
            rate = np.gradient(d, tt) * 1e3
            fig = fig_base("Waktu sejak lampu menyala (menit)", "Laju drift (mA/menit)", 260)
            fig.add_trace(go.Scatter(x=tt, y=rate, mode="lines", line=dict(color=PAL[0]), name="laju drift"))
            fig.add_vline(x=15, line_dash="dash", line_color=PAL[1], annotation_text="15 menit")
            st.plotly_chart(fig, key="driftchart")
            st.caption("Model simulator: laju drift setelah kompensasi berkas ganda turun ±10× dalam 15 menit pertama.")
        with st.expander("Log aktivitas"):
            st.code("\n".join(S["events"]) or "Belum ada aktivitas.", language=None)


def page_panel():
    st.markdown("Operasi **standalone** dari panel UV-1700 (tanpa PC). Saat **PC Control** aktif, panel hanya menampilkan data; tekan **MODE** untuk keluar.")
    left, right = st.columns([1, 1.25], gap="large")
    with right:
        compartment_panel("panel")
    with left:
        slot = st.empty()
        keys = [["F1", "F2", "F3", "F4"], ["RETURN", "MODE", "PRINT", "◁(−)"], ["7", "8", "9", "▷"], ["4", "5", "6", "GOTO WL"],
                ["1", "2", "3", "AUTO ZERO"], ["0", ".", "ENTER", "START/STOP"]]
        pressed = None
        for r, row in enumerate(keys):
            cols = st.columns(4)
            for c, k in enumerate(row):
                if cols[c].button(k, key=f"kp_{r}_{c}"):
                    pressed = k
        L = S["lcd"]
        if pressed:
            if not ready():
                st.warning("Alat belum menyala/siap.")
            elif pressed == "F4":
                HW["pc_control"] = True; event("F4: PC Control.")
            elif pressed == "MODE":
                if HW["pc_control"]:
                    HW["pc_control"] = False; HW["connected"] = False; event("MODE: keluar dari PC Control; UVProbe terputus.")
            elif HW["pc_control"]:
                st.info("Dalam PC Control; gunakan UVProbe atau tekan MODE.")
            elif pressed in "0123456789." and len(pressed) == 1:
                L["buf"] += pressed
            elif pressed == "◁(−)":
                L["buf"] = L["buf"][:-1]
            elif pressed == "RETURN":
                L["buf"], L["await"] = "", None
            elif pressed == "GOTO WL":
                L["await"], L["buf"] = "wl", ""
            elif pressed == "ENTER":
                if L["await"] == "wl":
                    v = num(L["buf"])
                    if 190 <= v <= 1100:
                        L["lam"] = round(v, 1)
                        L["A"] = float(read(L["lam"])["A"][0])
                        event(f"GOTO WL {f(v, 1)} nm.")
                    else:
                        st.error("λ harus 190–1100 nm.")
                L["buf"], L["await"] = "", None
            elif pressed == "AUTO ZERO":
                do_zero(L["lam"])
            elif pressed == "START/STOP":
                r = read(L["lam"])
                L["A"] = float(r["A"][0])
                event(f"Baca {f(L['lam'], 1)} nm: A = {L['A']:.3f} ({S['cell']['label']}).")
                if not r["corrected"][0]:
                    st.warning("Belum ada autozero pada λ ini; pembacaan mengandung offset instrumen.")
                if r["low_energy"][0]:
                    st.error("Energi berkas sangat rendah (kuvet/kompartemen?); nilai tidak bermakna.")
            elif pressed == "PRINT":
                event(f"PRINT: {f(L['lam'], 1)} nm, {L['A']:.3f} ABS.")
            elif pressed == "▷":
                pass
        slot.markdown(lcd_html(), unsafe_allow_html=True)
        st.caption("Contoh: GOTO WL → 5 1 0 → ENTER; isi sel sampel dengan blanko → AUTO ZERO; ganti larutan → START/STOP.")
        with st.expander("Log aktivitas"):
            st.code("\n".join(S["events"][:25]) or "—", language=None)


def page_spectrum():
    st.markdown("Meniru **UVProbe → Window → Spectrum** (IK bagian B, *Mencari λ maksimum*).")
    with st.expander("Langkah IK — Mencari λ maksimum", expanded=False):
        for i, s in enumerate(IK_SPEKTRUM):
            st.markdown(f"{i + 1}. {s}")
    left, right = st.columns([1, 2], gap="large")
    sp = S["sp"]
    with left:
        compartment_panel("spektrum")
        with st.container(border=True):
            st.markdown("**Edit → Method**")
            a, b = st.columns(2)
            sp["start"] = a.number_input("Start (λ tertinggi, nm)", 190.0, 1100.0, float(sp["start"]), 10.0)
            sp["end"] = b.number_input("End (λ terendah, nm)", 190.0, 1100.0, float(sp["end"]), 10.0)
            iv = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            sp["interval"] = a.selectbox("Sampling interval (nm)", iv, index=iv.index(sp["interval"]))
            sv = ["Fast", "Medium", "Slow"]
            sp["speed"] = b.selectbox("Scan speed", sv, index=sv.index(sp["speed"]))
            sp["mode"] = st.radio("Measuring mode", ["Abs", "%T"], index=0 if sp["mode"] == "Abs" else 1, horizontal=True,
                                  format_func=lambda m: "Absorbance" if m == "Abs" else "Transmittance")
            if sp["start"] < sp["end"]:
                st.error("Start harus λ tertinggi dan End λ terendah (IK B.2.a).")
            lo, hi = min(sp["start"], sp["end"]), max(sp["start"], sp["end"])
            st.caption(f"{int(round((hi - lo) / sp['interval'])) + 1} titik · lampu berganti pada {f(HW['lamp_switch'], 1)} nm")
            if lo < KUVET[CFG["kuvet"]]["cutoff"]:
                st.error(f"Kuvet {KUVET[CFG['kuvet']]['nama']} menyerap kuat di bawah ±{KUVET[CFG['kuvet']]['cutoff']} nm.")
    with right:
        ok = need_ready("pemindaian") and need_uvprobe()
        b1, b2, b3 = st.columns(3)
        bl = S["baseline"]
        bl_ok = bool(bl and bl["key"] == CFG["analit"] and bl["kuvet"] == CFG["kuvet"] and bl["lo"] <= lo and bl["hi"] >= hi)
        if b1.button("Baseline", disabled=not ok or sp["start"] < sp["end"]):
            if S["cell"]["smp"]["kind"] != "blanko" or S["cell"]["ref"]["kind"] != "blanko":
                st.warning("IK B.3: blanko harus di kedua kompartemen; baseline tetap direkam dengan isi saat ini.")
            lam = np.round(np.arange(lo, hi + sp["interval"] / 2, sp["interval"]), 3)
            r = raw_reading(lam, CFG["analit"], S["cell"]["smp"], S["cell"]["ref"], cfg_now(), t_warm(), rng(), "Slow")
            S["baseline"] = {"lo": lo, "hi": hi, "lam": lam, "A": r["A"] + 0.3 * r["noise"], "key": CFG["analit"],
                             "kuvet": CFG["kuvet"], "b": CFG["b"], "t": time.time(), "isi": S["cell"]["label"]}
            with st.spinner("Baseline…"):
                time.sleep(0.6)
            event(f"Baseline {hi:.0f}→{lo:.0f} nm ({S['cell']['label']}).")
            st.rerun()
        start = b2.button("Start", type="primary", disabled=not ok or sp["start"] < sp["end"])
        b3.markdown(chip("Baseline berlaku" if bl_ok else "Baseline belum/tidak berlaku", "ok" if bl_ok else "sim"), unsafe_allow_html=True)
        if start:
            n = int(round((hi - lo) / sp["interval"])) + 1
            lam = np.round(hi - np.arange(n) * sp["interval"], 3)          # λ panjang → pendek
            bar = st.progress(0.0, f"Scanning {hi:.0f} → {lo:.0f} nm")
            for k in range(1, 11):
                bar.progress(k / 10, f"Scanning… {hi - (hi - lo) * k / 10:.0f} nm")
                time.sleep({"Fast": 0.03, "Medium": 0.07, "Slow": 0.12}[sp["speed"]])
            r = read(lam, sp["speed"])
            spec = {"id": nid(), "name": f"{file_stub()}_{S['cell']['label']}", "label": S["cell"]["label"], "src": "simulasi",
                    "lam": lam.tolist(), "A": np.round(r["A"], 5).tolist(), "time": f"{datetime.now():%d/%m %H.%M}",
                    "analit": CFG["analit"], "baseline": bool(r["corrected"].all()), "low_energy": bool(r["low_energy"].any()),
                    "speed": sp["speed"], "interval": sp["interval"]}
            S["spectra"].append(spec); S["spectra"] = S["spectra"][-30:]; S["active"] = spec["id"]
            event(f"Scan {spec['name']}" + ("" if spec["baseline"] else " (tanpa baseline berlaku)") + ".")
            st.rerun()
        with st.expander("Impor spektrum dari UVProbe (CSV/TXT: λ; Abs)"):
            up = st.file_uploader("Berkas data spektrum", type=["csv", "txt", "tsv"], key="spimp")
            if up is not None and st.button("Tambahkan spektrum impor"):
                res = parse_table(up.getvalue().decode("utf-8", errors="replace"))
                if not res["ok"]:
                    st.error(res["reason"])
                else:
                    for c in range(1, len(res["cols"])):
                        x, y = res["cols"][0], res["cols"][c]
                        m = np.isfinite(x) & np.isfinite(y)
                        if m.sum() < 7:
                            continue
                        nm = res["header"][c] if res["header"] and c < len(res["header"]) else f"{up.name} #{c}"
                        S["spectra"].append({"id": nid(), "name": nm, "label": nm, "src": "impor", "lam": x[m].tolist(), "A": y[m].tolist(),
                                             "time": f"{datetime.now():%d/%m %H.%M}", "analit": None, "baseline": True, "low_energy": False})
                        S["active"] = S["spectra"][-1]["id"]
                    st.rerun()
        if not S["spectra"]:
            st.info("Belum ada spektrum. Lakukan Baseline dengan blanko, isi sel sampel dengan standar tertinggi, lalu Start.")
            return
        names = {s["id"]: f"{s['name']} [{s['src']}]" for s in S["spectra"]}
        sel_ids = st.multiselect("Spektrum ditampilkan (overlay)", list(names), default=[S["active"]] if S["active"] in names else [S["spectra"][-1]["id"]],
                                 format_func=names.get)
        sel = [s for s in S["spectra"] if s["id"] in sel_ids]
        proc = st.radio("Tampilan", ["Asli", "Savitzky–Golay", "Turunan pertama"], horizontal=True)
        sp["thr"] = st.number_input("Threshold peak pick (A)", 0.001, 0.5, float(sp["thr"]), 0.001, format="%.3f")
        yt = {"Turunan pertama": "dA/dλ (nm⁻¹)"}.get(proc, "Transmitansi (%)" if sp["mode"] == "%T" else "Absorbansi")
        fig = fig_base("Panjang gelombang (nm)", yt, 420)
        for i, s in enumerate(sel):
            x = np.asarray(s["lam"]); y = np.asarray(s["A"]); o = np.argsort(x); x, y = x[o], y[o]
            if proc == "Savitzky–Golay":
                y = smooth(y, 9)
            if proc == "Turunan pertama":
                y = np.gradient(smooth(y, 11), x)
            elif sp["mode"] == "%T":
                y = 100 * 10 ** -y
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s["name"], line=dict(color=PAL[i % len(PAL)], width=2)))
        tgt = sel[-1] if sel else None
        peaks = peak_pick(tgt["lam"], tgt["A"], sp["thr"]) if tgt else []
        for p in [p for p in peaks if p["Jenis"] == "Puncak"][:3]:
            fig.add_vline(x=p["λ (nm)"], line_dash="dash", line_color=PAL[1], annotation_text=f(p["λ (nm)"], 1) + " nm")
        st.plotly_chart(fig, key="specplot")
        if tgt:
            st.markdown(src_chip(tgt["src"]) + (chip("tanpa baseline", "bad") if not tgt["baseline"] else "") +
                        (chip("energi rendah", "bad") if tgt["low_energy"] else ""), unsafe_allow_html=True)
            st.markdown(f"**Operation → Peak Pick — {tgt['name']}**")
            if peaks:
                pdf_ = pd.DataFrame(peaks)
                show = pdf_.copy()
                for c in ["λ (nm)", "FWHM (nm)"]:
                    show[c] = show[c].map(lambda v: f(v, 1))
                for c in ["Abs", "Prominensi"]:
                    show[c] = show[c].map(lambda v: f(v, 4))
                st.dataframe(show, hide_index=True)
                top = next((p for p in peaks if p["Jenis"] == "Puncak"), None)
                if top:
                    exp = ANALIT[tgt["analit"]]["lam"] if tgt.get("analit") else None
                    st.markdown(f"λmaks = **{f(top['λ (nm)'], 1)} nm**, A = {f(top['Abs'], 4)}" +
                                (f" (acuan metode {exp} nm; selisih {f(top['λ (nm)'] - exp, 1)} nm)" if exp else ""))
                    if top["Abs"] > 1.5:
                        st.warning("A puncak > 1,5: rasio sinar sesat dan galat fotometrik meningkat; gunakan standar lebih encer untuk menentukan bentuk pita.")
                    if st.button(f"Gunakan {f(top['λ (nm)'], 1)} nm sebagai WL1 di mode fotometrik"):
                        S["lcd"]["lam"] = round(top["λ (nm)"], 1)
                        S["pho_wl_hint"] = round(top["λ (nm)"], 1)
                        st.toast(f"WL1 = {f(top['λ (nm)'], 1)} nm")
                c1, c2 = st.columns(2)
                pdf_button(c1, "Print tabel peak pick (PDF)", "peak", lambda: build_pdf({
                    "title": "PEAK PICK", "subtitle": tgt["name"], "simulated": tgt["src"] != "impor",
                    "blocks": [{"t": "img", "png": spectrum_png([tgt], [p["λ (nm)"] for p in peaks if p["Jenis"] == "Puncak"]), "cap": tgt["name"]},
                               {"t": "table", "head": list(show.columns), "rows": show.values.tolist()}]}),
                    f"{tgt['name']}_peakpick.pdf")
            else:
                st.info("Tidak ada puncak di atas threshold.")
            dfo = pd.DataFrame({"Panjang gelombang (nm)": tgt["lam"], "Abs": tgt["A"]})
            st.download_button("File → Save As (CSV)", csv_bytes(dfo), f"{tgt['name']}.csv", "text/csv")
            if st.button("Hapus spektrum yang ditampilkan"):
                S["spectra"] = [s for s in S["spectra"] if s["id"] not in sel_ids]; st.rerun()


def page_photometric():
    st.markdown("Meniru **UVProbe → Window → Photometric** (IK bagian B, *Analisa Sample*).")
    with st.expander("Langkah IK — Analisa sampel", expanded=False):
        for i, s in enumerate(IK_FOTOMETRIK):
            st.markdown(f"{i + 1}. {s}")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        compartment_panel("fotometrik")
        with st.container(border=True):
            st.markdown("**Edit → Method (wizard)**")
            pm = S["pho"] or {}
            wl_default = S.get("pho_wl_hint") or pm.get("wl") or float(AN()["lam"])
            wl = st.number_input("Wavelength (nm) → Add", 190.0, 1100.0, float(wl_default), 0.1)
            types = ["Multi Point", "Single Point", "K-Factor"]
            typ = st.selectbox("Type", types, index=types.index(pm.get("type", "Multi Point")))
            units = ["mg/L", "ppm", "µg/mL", "µg/L", "mmol/L", "%"]
            un = st.selectbox("Unit", units, index=units.index(pm.get("unit", "mg/L")))
            kf = st.number_input("K-factor (konsentrasi per A)", 0.0001, 1e6, float(pm.get("k", 5.0))) if typ == "K-Factor" else None
            zero = st.checkbox("Zero Interception", pm.get("zero", True), help="IK B: Zero Interception dicentang. Kesahihannya diuji di Evaluasi Kalibrasi.")
            fname = st.text_input("File name", pm.get("file", file_stub() + "_photometric"))
            if st.button("Finish", type="primary"):
                S["pho"] = {"wl": round(wl, 1), "type": typ, "unit": un, "zero": zero, "file": fname, "k": kf, "analit": CFG["analit"]}
                S.pop("pho_wl_hint", None)
                if not S["std"]:
                    set_std([{"ID Sampel": f"Std{i + 1}", "Konsentrasi": c, "Abs": None, "Pakai": True} for i, c in enumerate(AN()["std"])], "simulasi")
                event(f"Metode fotometrik: {typ}, WL1 {f(wl, 1)} nm, unit {un}, zero interception {'ya' if zero else 'tidak'}.")
                st.rerun()
    with right:
        if not S["pho"]:
            st.info("Selesaikan wizard **Edit → Method** lalu klik **Finish**.")
            return
        pm = S["pho"]
        st.markdown(f"<div class='k-eq'>{pm['file']} · WL1 {f(pm['wl'], 1)} nm · {pm['type']} · {pm['unit']} · Zero Interception: {'ON' if pm['zero'] else 'OFF'}</div>", unsafe_allow_html=True)
        ok_hw = ready() and HW["connected"]
        if not ok_hw and S["ui"]["mode"] == "latihan":
            need_ready() and need_uvprobe()
        z = S["zero"]
        zok = bool(z and abs(z["lam"] - pm["wl"]) < 0.05 and z["key"] == CFG["analit"])
        c1, c2, c3 = st.columns([1, 1, 1.4])
        if c1.button("Autozero", disabled=not ok_hw):
            S["lcd"]["lam"] = pm["wl"]; do_zero(pm["wl"]); st.rerun()
        c2.markdown(chip("Autozero " + (z["isi"] if zok else "belum"), "ok" if zok and z["isi"] == "Blanko" else ("bad" if zok else "sim")), unsafe_allow_html=True)
        if zok:
            age = (time.time() - z["t"]) / 60 + 0.0
            c3.caption(f"Autozero {f(age, 1)} menit lalu" + (" — pertimbangkan autozero ulang." if age > 20 else ""))

        st.markdown("**Standard Table** " + src_chip(S["src"]), unsafe_allow_html=True)
        df = std_df()
        edited = st.data_editor(df, key=f"std_{S['std_ver']}", num_rows="dynamic", hide_index=True,
                                column_config={"Konsentrasi": st.column_config.NumberColumn(f"Conc ({pm['unit']})", format="%.4f"),
                                               "Abs": st.column_config.NumberColumn(f"WL{f(pm['wl'], 1)} (Abs)", format="%.4f"),
                                               "Pakai": st.column_config.CheckboxColumn("Pakai", default=True)})
        if not edited.reset_index(drop=True).astype(str).equals(df.reset_index(drop=True).astype(str)):
            S["std"] = [{"ID Sampel": r["ID Sampel"] if isinstance(r["ID Sampel"], str) else "", "Konsentrasi": None if pd.isna(r["Konsentrasi"]) else float(r["Konsentrasi"]),
                         "Abs": None if pd.isna(r["Abs"]) else float(r["Abs"]), "Pakai": bool(r["Pakai"]) if not pd.isna(r["Pakai"]) else True}
                        for r in edited.to_dict("records")]
            S["std_ver"] += 1
            if S["src"] in ("simulasi", "contoh"):
                S["src"] = "campuran"
        rows = S["std"]
        a1, a2, a3 = st.columns([1.3, 1, 1])
        if rows:
            ri = a1.selectbox("Baris standar", list(range(len(rows))), format_func=lambda i: f"{rows[i]['ID Sampel']} · {g(rows[i]['Konsentrasi'], 3)}")
            if a2.button("Read Std", disabled=not ok_hw):
                lab = S["cell"]["label"]
                expect = f"Standar {cf(rows[ri]['Konsentrasi'])} {AN()['unit']}"
                if lab != expect:
                    st.warning(f"Sel sampel berisi '{lab}', bukan {expect}.")
                A = float(read(pm["wl"])["A"][0])
                rows[ri]["Abs"] = round(A, 4); S["lcd"]["A"] = A; S["lcd"]["lam"] = pm["wl"]
                set_std(rows); event(f"Read Std {rows[ri]['ID Sampel']}: {A:.4f}."); st.rerun()
            if a3.button("Baca seluruh deret", disabled=not ok_hw, help="Mengulang langkah 12 untuk semua standar (ganti larutan → Read Std)."):
                r_ = rng()
                for row in rows:
                    if row["Konsentrasi"] is None:
                        continue
                    lab = f"Standar {cf(row['Konsentrasi'])} {AN()['unit']}"
                    set_cell("standar", prep_conc(row["Konsentrasi"], r_), lab)
                    row["Abs"] = round(float(read(pm["wl"])["A"][0]), 4)
                set_std(rows); event("Seluruh deret standar dibaca."); st.rerun()
        paste = st.text_area("Tempel data alat nyata (Conc lalu Abs per baris)", height=80, placeholder="0,5\t0,101\n1,0\t0,199")
        p1, p2, p3 = st.columns(3)
        if p1.button("Masukkan tempelan") and paste.strip():
            res = parse_table(paste)
            if res["ok"]:
                set_std([{"ID Sampel": f"Std{i + 1}", "Konsentrasi": float(c), "Abs": float(a), "Pakai": True}
                         for i, (c, a) in enumerate(zip(res["cols"][0], res["cols"][1])) if np.isfinite(c)], "pengguna"); st.rerun()
            else:
                st.error(res["reason"])
        if S["ui"]["mode"] == "latihan" and p2.button("Data contoh Fe"):
            set_std([{"ID Sampel": f"Std{i + 1}", "Konsentrasi": c, "Abs": a, "Pakai": True}
                     for i, (c, a) in enumerate(zip([0.5, 1, 2, 3, 4, 5], [0.102, 0.201, 0.398, 0.601, 0.794, 0.992]))], "contoh")
            S["smp"].append(new_sample("Air sumur A (contoh)", "0,452; 0,448; 0,455", "contoh"))
            st.rerun()
        if p3.button("Kosongkan tabel"):
            set_std([], "pengguna"); st.rerun()

        fit = current_fit()
        if fit and fit.ok:
            fig = fig_base(f"Konsentrasi ({pm['unit']})", "Absorbansi", 330)
            fig.update_layout(hovermode="closest")
            p = std_points()
            fig.add_trace(go.Scatter(x=p["Konsentrasi"], y=p["Abs"], mode="markers", name="Standar", marker=dict(size=10, color=PAL[0])))
            xs = np.array([0, fit.xmax * 1.05])
            fig.add_trace(go.Scatter(x=xs, y=fit.a * xs + fit.b, mode="lines", name=fit.equation(4), line=dict(color=PAL[1])))
            for s in S["smp"]:
                r = sample_res(s, fit)
                if r:
                    fig.add_trace(go.Scatter(x=[r["c_meas"]], y=[r["A_mean"]], mode="markers+text", text=[s["name"]], textposition="top right",
                                             name=s["name"], marker=dict(symbol="diamond", size=12, color=PAL[2])))
            st.plotly_chart(fig, key="phocal")
            st.markdown(f"<div class='k-eq'>{fit.equation()} · r = {f(fit.r, 5)} · R² = {f(fit.r2, 5)}</div>", unsafe_allow_html=True)
        elif fit:
            st.info(fit.reason)

        st.markdown("**Sample Table**")
        t1, t2 = st.columns(2)
        if t1.button("+ Sampel (data sendiri)"):
            S["smp"].append(new_sample(f"Sampel {len(S['smp']) + 1}")); st.rerun()
        if S["ui"]["mode"] == "latihan" and t2.button("+ Sampel X (konsentrasi tersembunyi)"):
            k = sum(1 for s in S["smp"] if s.get("hidden") is not None)
            hc = hidden_conc(CFG["analit"], S["seed"], k)
            s = new_sample(f"Sampel X{k + 1}", "", "simulasi", hc, matriks=0.004 * (k % 3))
            S["smp"].append(s); event(f"{s['name']} ditambahkan; masukkan ke sel sampel lalu Read Unk."); st.rerun()
        for idx, s in enumerate(list(S["smp"])):
            with st.expander(s["name"], expanded=True):
                a, b = st.columns([1, 1.1], gap="medium")
                with a:
                    s["name"] = st.text_input("Sample ID", s["name"], key=f"n{s['id']}")
                    s["reps"] = st.text_input("Abs ulangan (pisahkan ;)", s["reps"], key=f"r{s['id']}_{s['v']}", placeholder="0,412; 0,415; 0,409")
                    if s.get("hidden") is not None:
                        if st.button("Read Unk", key=f"ru{s['id']}", disabled=not ok_hw):
                            if S["cell"]["label"] != s["name"]:
                                st.warning(f"Sel sampel berisi '{S['cell']['label']}'. Pilih '{s['name']}' pada Isi sel sampel.")
                            else:
                                A = float(read(pm["wl"])["A"][0])
                                s["reps"] = (s["reps"] + "; " if s["reps"].strip() else "") + f(A, 4)
                                s["v"] += 1; S["lcd"]["A"] = A; event(f"Read Unk {s['name']}: {A:.4f}."); st.rerun()
                    st.caption("Pengenceran: V pipet → V labu (FP = Π V labu/V pipet).")
                    steps = st.data_editor(pd.DataFrame(s["steps"], columns=["V pipet (mL)", "V labu (mL)"], dtype=float), num_rows="dynamic",
                                           hide_index=True, key=f"st{s['id']}_{s['v']}")
                    new_steps = steps.dropna().to_dict("records")
                    if new_steps != s["steps"]:
                        s["steps"] = new_steps
                        s["v"] += 1
                    m1, m2 = st.columns(2)
                    s["mass"] = m1.text_input("Massa sampel padat (g)", s["mass"], key=f"m{s['id']}")
                    s["vol"] = m2.text_input("Volume larutan awal (mL)", s["vol"], key=f"v{s['id']}")
                    if st.button("Hapus sampel", key=f"d{s['id']}"):
                        S["smp"].pop(idx); st.rerun()
                with b:
                    st.markdown(src_chip(s["src"]), unsafe_allow_html=True)
                    try:
                        reps = parse_list(s["reps"])
                    except ValueError as e:
                        st.error(str(e)); continue
                    if not reps:
                        st.info("Belum ada absorbansi.")
                        continue
                    u = pm["unit"]
                    if pm["type"] != "Multi Point":
                        d = describe(reps)
                        if pm["type"] == "K-Factor":
                            c = pm["k"] * d["mean"]
                        else:
                            p = std_points()
                            c = d["mean"] * float(p["Konsentrasi"].iloc[0]) / float(p["Abs"].iloc[0]) if len(p) else NAN
                        st.markdown(f"Ā = {f(d['mean'], 4)} → C = **{g(c)} {u}** × FP {g(sample_fp(s))} = **{g(c * sample_fp(s))} {u}**")
                        st.caption(f"{pm['type']}: tanpa estimasi ketidakpastian regresi.")
                        continue
                    r = sample_res(s, fit)
                    if not r:
                        st.info("Kurva kalibrasi belum valid.")
                        continue
                    st.markdown(f"Ā (n = {r['n']}) = **{f(r['A_mean'], 4)}**" + (f" · RSD {f(r['rsd'], 2)} %" if r["n"] > 1 else ""))
                    st.markdown(f"C terukur = **{g(r['c_meas'])} ± {g(r['ci'], 2)} {u}** (SK 95 %)")
                    st.markdown(f"#### C sampel = {g(r['c_final'])} ± {g(r['ci_final'], 2)} {u}")
                    if r["mass_ok"]:
                        st.markdown(f"**Kadar = {g(r['kadar'])} mg/kg**")
                    for fl in r["flags"]:
                        st.warning(fl)
                    if not r["flags"]:
                        st.success("Dalam rentang kalibrasi dan daerah kerja absorbansi.")
                    if s.get("hidden") is not None and S["ui"]["mode"] == "latihan" and st.button("Ungkap nilai model", key=f"rv{s['id']}"):
                        st.info(f"Nilai model {g(s['hidden'], 3)} {u}; perolehan kembali {f(r['c_meas'] / s['hidden'] * 100, 1)} %.")
        if fit and fit.ok:
            res_rows = []
            for s in S["smp"]:
                r = sample_res(s, fit)
                if r:
                    res_rows.append({"Sample ID": s["name"], "n": r["n"], "Abs rerata": round(r["A_mean"], 4), "RSD %": round(r["rsd"], 2) if r["n"] > 1 else None,
                                     f"Conc ({pm['unit']})": round(r["c_meas"], 5), "FP": round(r["fp"], 3), f"C sampel ({pm['unit']})": round(r["c_final"], 5),
                                     "SK95 ±": round(r["ci_final"], 5)})
            pt = std_points()
            d1, d2, d3 = st.columns(3)
            std_rows = [[r["ID Sampel"], g(r["Konsentrasi"]), f(r["Abs"], 4)] for r in S["std"] if r["Abs"] is not None]
            pdf_button(d1, "Print Standard Table (PDF)", "std", lambda: build_pdf({
                "title": "STANDARD TABLE", "subtitle": f"{pm['file']} · WL1 {f(pm['wl'], 1)} nm", "simulated": S["src"] != "pengguna",
                "blocks": [{"t": "table", "head": ["Sample ID", f"Conc ({pm['unit']})", "Abs"], "rows": std_rows},
                           {"t": "img", "png": calib_png(fit, pm["unit"]), "cap": "Kurva kalibrasi dan residual."},
                           {"t": "kv", "rows": [("Persamaan", fit.equation()), ("r / R²", f"{f(fit.r, 5)} / {f(fit.r2, 5)}"), ("Zero interception", "Ya" if fit.zero else "Tidak")]}]}),
                f"{pm['file']}_standard.pdf")
            if res_rows:
                pdf_button(d2, "Print Sample Table (PDF)", "smp", lambda: build_pdf({
                    "title": "SAMPLE TABLE", "subtitle": f"{pm['file']} · WL1 {f(pm['wl'], 1)} nm", "simulated": S["src"] != "pengguna",
                    "blocks": [{"t": "table", "head": list(res_rows[0].keys()), "rows": [[g(v) if isinstance(v, float) else v for v in r.values()] for r in res_rows]}]}),
                    f"{pm['file']}_sample.pdf")
            d3.download_button("File → Save (CSV)", csv_bytes(pd.concat([std_df().assign(Jenis="Standard"),
                               pd.DataFrame(res_rows).assign(Jenis="Unknown")], ignore_index=True)), f"{pm['file']}.csv", "text/csv")


def calib_interpretation(fit: Fit, u: str) -> list[str]:
    out = [f"**Linearitas.** {fit.equation()} dengan r = {f(fit.r, 5)} dan R² = {f(fit.r2, 5)}; kriteria R² ≥ {f(CR['r2_min'], 3)} "
           f"{'terpenuhi' if fit.r2 >= CR['r2_min'] else 'tidak terpenuhi'}. R² hanya mengukur proporsi variansi yang dijelaskan, bukan bukti linearitas.",
           f"**Kepekaan.** Kemiringan {g(fit.a)} ± {g(fit.ci_a, 2)} A per {u} (SK 95 %). Untuk b = 1 cm dan satuan mg/L, nilai ini adalah absorptivitas a (L·mg⁻¹·cm⁻¹)."]
    if not fit.zero:
        out.append(f"**Intersep.** b = {g(fit.b, 3)} ± {g(fit.ci_b, 2)}; " + (
            "berbeda nyata dari nol — telusuri serapan reagen, blanko, atau autozero; Zero Interception tidak dibenarkan." if fit.b_significant else
            "tidak berbeda nyata dari nol (α = 0,05), sehingga Zero Interception dapat dibenarkan secara statistik."))
    out.append(f"**Batas deteksi.** s_y/x = {g(fit.syx, 3)}; LOD = 3,3·s_y/x/a = {g(fit.lod, 3)} {u}; LOQ = 10·s_y/x/a = {g(fit.loq, 3)} {u}. "
               "Perkiraan dari kurva; penetapan formal memerlukan ulangan blanko (n ≥ 10).")
    big = int(np.sum(np.abs(fit.sres) > CR["sres_max"]))
    if big:
        out.append(f"**Residual.** {big} titik memiliki |residual terstandar| > {f(CR['sres_max'], 1)}; periksa preparasi/pembacaan sebelum mengeluarkannya.")
    runs = residual_runs(fit.res)
    if fit.n >= 5 and runs <= 2:
        out.append(f"**Pola residual.** Hanya {runs} kelompok tanda — indikasi kelengkungan sistematis.")
    if fit.n < 5:
        out.append(f"**Jumlah titik.** {fit.n} titik; pedoman validasi umumnya mensyaratkan ≥ 5 tingkat konsentrasi.")
    return out


def page_calibration():
    if not S["pho"]:
        st.info("Buat metode fotometrik dan isi Standard Table terlebih dahulu (atau tempel data nyata di halaman Mode Fotometrik).")
        return
    u = unit()
    f0, f1 = current_fit(zero=False), current_fit(zero=True)
    if not f0 or not f0.ok:
        st.info(f0.reason if f0 else "Minimal 3 standar terbaca.")
        return
    use = f1 if S["pho"]["zero"] else f0
    st.markdown(f"Model aktif mengikuti pengaturan metode: **{'melalui nol (Zero Interception)' if S['pho']['zero'] else 'dengan intersep'}**.")
    c = st.columns(4)
    c[0].metric("R² (aktif)", f(use.r2, 5), "memenuhi" if use.r2 >= CR["r2_min"] else "di bawah kriteria", delta_color="normal" if use.r2 >= CR["r2_min"] else "inverse")
    c[1].metric("Kemiringan", g(use.a), f"± {g(use.ci_a, 2)}", delta_color="off")
    c[2].metric("LOD", f"{g(use.lod, 3)} {u}")
    c[3].metric("LOQ", f"{g(use.loq, 3)} {u}")
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        fig = fig_base(f"Konsentrasi ({u})", "Absorbansi", 360)
        fig.update_layout(hovermode="closest")
        fig.add_trace(go.Scatter(x=f0.x, y=f0.y, mode="markers", name="Standar", marker=dict(size=10, color=PAL[0])))
        xs = np.array([0, f0.xmax * 1.05])
        fig.add_trace(go.Scatter(x=xs, y=f0.a * xs + f0.b, mode="lines", name="Dengan intersep: " + f0.equation(4), line=dict(color=PAL[1])))
        fig.add_trace(go.Scatter(x=xs, y=f1.a * xs, mode="lines", name="Melalui nol: " + f1.equation(4), line=dict(color=PAL[2], dash="dash")))
        st.plotly_chart(fig, key="calfit")
        fig = fig_base(f"Konsentrasi ({u})", "Residual terstandar", 250)
        fig.update_layout(hovermode="closest")
        fig.add_trace(go.Scatter(x=use.x, y=use.sres, mode="markers", marker=dict(size=10, color=PAL[2]), name="e/s_y/x"))
        fig.add_hline(y=0, line_color=PAL[1])
        for sg in (1, -1):
            fig.add_hline(y=sg * CR["sres_max"], line_dash="dot", line_color="#8a99b5")
        st.plotly_chart(fig, key="calres")
    with b:
        tab = pd.DataFrame({"Parameter": ["Persamaan", "r", "R²", "s_y/x", "Kemiringan ± SK95", "Intersep ± SK95", "LOD", "LOQ", "Rentang", "n"],
                            "Dengan intersep": [f0.equation(), f(f0.r, 5), f(f0.r2, 5), g(f0.syx, 3), f"{g(f0.a)} ± {g(f0.ci_a, 2)}", f"{g(f0.b, 3)} ± {g(f0.ci_b, 2)}",
                                                g(f0.lod, 3), g(f0.loq, 3), f"{g(f0.xmin, 3)}–{g(f0.xmax, 3)}", f0.n],
                            "Melalui nol": [f1.equation(), "—", f(f1.r2, 5), g(f1.syx, 3), f"{g(f1.a)} ± {g(f1.ci_a, 2)}", "0", g(f1.lod, 3), g(f1.loq, 3),
                                            f"{g(f1.xmin, 3)}–{g(f1.xmax, 3)}", f1.n]})
        st.dataframe(tab.astype(str), hide_index=True)
        tb = abs(f0.b) / f0.sb if f0.sb > 0 else NAN
        st.markdown(f"**Uji intersep:** t = |b|/s_b = {f(tb, 2)} vs t(0,05; {f0.df}) = {f(f0.tc, 2)} → " +
                    ("**intersep nyata**; Zero Interception menimbulkan bias." if f0.b_significant else "intersep tidak nyata; Zero Interception dapat diterima."))
        mt = mandel_test(f0.x, f0.y)
        if mt:
            st.markdown(f"**Uji Mandel (α = 0,01):** F = {f(mt['F'], 2)} vs F(1; {mt['df2']}) = {f(mt['Fcrit'], 2)} → " +
                        ("model kuadratik lebih sesuai (**ketidaklinearan nyata**)." if mt["nonlinear"] else "tidak ada bukti ketidaklinearan."))
        lf = lack_of_fit(f0.x, f0.y)
        if lf:
            st.markdown(f"**Lack-of-fit:** F = {f(lf['F'], 2)} vs F{lf['df']} = {f(lf['Fcrit'], 2)} → " + ("tidak sesuai." if lf["lof"] else "sesuai."))
        A = np.linspace(0.05, 2.0, 120)
        fig = fig_base("Absorbansi", "Galat relatif C (%)", 220)
        fig.add_trace(go.Scatter(x=A, y=ringbom_error(A), mode="lines", line=dict(color=PAL[0]), name="ΔT = 0,5 %"))
        fig.add_vrect(x0=CR["a_min"], x1=CR["a_max"], fillcolor=PAL[3], opacity=0.1, line_width=0)
        fig.add_trace(go.Scatter(x=f0.y, y=ringbom_error(np.clip(f0.y, 0.01, None)), mode="markers", marker=dict(color=PAL[2]), name="standar"))
        st.plotly_chart(fig, key="ringbom")
    with st.container(border=True):
        st.markdown("**Interpretasi otomatis**")
        for p in calib_interpretation(use, u):
            st.markdown("- " + p)


def precision():
    v = S["valid"]
    try:
        vals = parse_list(v["reps"])
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    d = describe(vals)
    hw = None
    if v["kind"] == "konsentrasi":
        lvl = num(v["level"], d["mean"]) if v["level"] else d["mean"]
        fac = {"mg/L": 1e-6, "ppm": 1e-6, "µg/mL": 1e-6, "µg/L": 1e-9, "%": 1e-2}.get(v["level_unit"])
        if fac and lvl > 0:
            pr = horwitz_prsd(lvl * fac)
            hw = {"level": lvl, "prsd": pr, "ref": 0.67 * pr, "horrat": d["rsd"] / (0.67 * pr)}
    return {"vals": vals, "d": d, "g": grubbs(vals), "pass": d["rsd"] <= CR["rsd_max"], "hw": hw}


def recovery():
    v = S["valid"]
    if v["rec_mode"] == "spike":
        cs, css, cad = num(v["cs"]), num(v["css"]), num(v["cadd"])
        if any(math.isnan(x) for x in (cs, css, cad)) or cad <= 0:
            return None
        R = (css - cs) / cad * 100
        return {"mode": "spike", "R": R, "pass": CR["rec_min"] <= R <= CR["rec_max"], "txt": f"({g(css)} − {g(cs)}) / {g(cad)} × 100"}
    pr = precision()
    cr = num(v["cref"])
    if not pr or not cr > 0:
        return None
    R = pr["d"]["mean"] / cr * 100
    return {"mode": "acuan", "R": R, "pass": CR["rec_min"] <= R <= CR["rec_max"], "txt": f"{g(pr['d']['mean'])} / {g(cr)} × 100", "t": t_test_mean(pr["vals"], cr)}


def page_validation():
    with st.container(border=True):
        st.markdown("**Kriteria penerimaan** (sesuaikan dengan SOP/metode baku; tidak ada batas universal).")
        c = st.columns(7)
        CR["r2_min"] = c[0].number_input("R² min", 0.0, 1.0, CR["r2_min"], 0.001, format="%.3f")
        CR["rsd_max"] = c[1].number_input("RSD maks (%)", 0.0, 50.0, CR["rsd_max"], 0.5)
        CR["rec_min"] = c[2].number_input("Recovery min (%)", 0.0, 200.0, CR["rec_min"], 1.0)
        CR["rec_max"] = c[3].number_input("Recovery maks (%)", 0.0, 200.0, CR["rec_max"], 1.0)
        CR["sres_max"] = c[4].number_input("|e/s| maks", 0.5, 5.0, CR["sres_max"], 0.5)
        CR["a_min"] = c[5].number_input("A min kerja", 0.0, 3.0, CR["a_min"], 0.05)
        CR["a_max"] = c[6].number_input("A maks kerja", 0.0, 3.0, CR["a_max"], 0.05)
    v = S["valid"]
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.subheader("Presisi (keterulangan)")
        v["kind"] = st.radio("Jenis data", ["konsentrasi", "absorbansi"], index=0 if v["kind"] == "konsentrasi" else 1, horizontal=True)
        fit = current_fit()
        cand = [s for s in S["smp"] if sample_res(s, fit)] if fit and fit.ok else []
        if cand and st.button("Ambil ulangan dari sampel pertama"):
            s = cand[0]
            reps = parse_list(s["reps"])
            v["reps"] = "; ".join(f(((a - fit.b) / fit.a) * sample_fp(s), 4) for a in reps) if v["kind"] == "konsentrasi" else s["reps"]
            st.rerun()
        v["reps"] = st.text_area("Nilai ulangan (pisahkan ;)", v["reps"], placeholder="2,251; 2,238; 2,264; 2,247; 2,255; 2,243")
        if v["kind"] == "konsentrasi":
            a, b = st.columns(2)
            v["level"] = a.text_input("Kadar acuan Horwitz (kosong = rerata)", v["level"])
            uu = ["mg/L", "ppm", "µg/mL", "µg/L", "%"]
            v["level_unit"] = b.selectbox("Satuan kadar", uu, index=uu.index(v["level_unit"]) if v["level_unit"] in uu else 0)
        pr = precision()
        if pr:
            d = pr["d"]
            st.dataframe(pd.DataFrame({"Parameter": ["n", "Rerata", "s", "RSD (%)", "SK 95 % rerata", "Rentang"],
                                       "Nilai": [str(d["n"]), g(d["mean"], 5), g(d["sd"], 3), f(d["rsd"], 2), "± " + g(d["ci95"], 3), f"{g(d['min'])} – {g(d['max'])}"]}),
                         hide_index=True)
            (st.success if pr["pass"] else st.error)(f"RSD {f(d['rsd'], 2)} % {'≤' if pr['pass'] else '>'} {f(CR['rsd_max'], 2)} %.")
            gr = pr["g"]
            if gr["ok"]:
                (st.warning if gr["outlier"] else st.info)(f"Grubbs (α = 0,05): G = {f(gr['G'], 3)}, G kritis = {f(gr['Gcrit'], 3)} → " +
                                                         (f"{g(gr['value'])} terindikasi pencilan; telusuri penyebabnya sebelum dikeluarkan." if gr["outlier"] else "tidak ada pencilan."))
            if pr["hw"]:
                h = pr["hw"]
                st.caption(f"Horwitz pada {g(h['level'], 3)} {v['level_unit']}: PRSD_R = {f(h['prsd'], 2)} %; batas keterulangan 0,67·PRSD_R = {f(h['ref'], 2)} %; "
                           f"HorRat_r = {f(h['horrat'], 2)} (wajar 0,3–1,3). Pembanding, bukan pengganti kriteria SOP.")
        else:
            st.info("Masukkan minimal 2 nilai.")
    with p2:
        st.subheader("Akurasi")
        v["rec_mode"] = st.radio("Pendekatan", ["spike", "acuan"], index=0 if v["rec_mode"] == "spike" else 1, horizontal=True,
                                 format_func=lambda m: "Adisi standar (spike)" if m == "spike" else "Nilai acuan / CRM (pakai ulangan presisi)")
        if v["rec_mode"] == "spike":
            a, b, c3 = st.columns(3)
            v["cs"] = a.text_input("C sampel", v["cs"])
            v["css"] = b.text_input("C sampel + spike", v["css"])
            v["cadd"] = c3.text_input("C ditambahkan", v["cadd"])
        else:
            v["cref"] = st.text_input("Nilai acuan (satuan sama dengan ulangan)", v["cref"])
        rc = recovery()
        if rc:
            st.markdown(f"<div class='k-eq'>%R = {rc['txt']} = <b>{f(rc['R'], 2)} %</b></div>", unsafe_allow_html=True)
            (st.success if rc["pass"] else st.error)(f"Recovery {'dalam' if rc['pass'] else 'di luar'} rentang {f(CR['rec_min'], 0)}–{f(CR['rec_max'], 0)} %.")
            if rc.get("t"):
                t = rc["t"]
                st.markdown(f"Uji t rerata vs acuan: t = {f(t['t'], 2)}, t kritis = {f(t['tcrit'], 2)}, p = {f(t['p'], 3)} → " +
                            ("**bias nyata** (α = 0,05)." if t["significant"] else "bias tidak nyata."))
        else:
            st.info("Lengkapi nilai untuk menghitung % recovery.")


def page_timescan():
    st.markdown('<div class="k-banner"><b>Model generik.</b> A(t) = A<sub>p</sub>(1 − e<sup>−k₁t</sup>)e<sup>−k₂t</sup> menggambarkan pembentukan lalu pemudaran warna; '
                'konstanta laju adalah nilai ilustratif per metode, bukan nilai literatur tersertifikasi. Data nyata dapat ditempel.</div>', unsafe_allow_html=True)
    ts = S["ts"]
    left, right = st.columns([1, 1.8], gap="large")
    with left:
        src = st.radio("Sumber data", ["simulasi", "pengguna"], horizontal=True, format_func=lambda s: "Simulasi" if s == "simulasi" else "Data pengguna")
        if src == "simulasi":
            st.markdown(f"Metode aktif: **{AN()['nama']}** pada {AN()['lam']} nm")
            ts["dur"] = st.number_input("Durasi (menit)", 5.0, 240.0, float(ts["dur"]), 5.0)
            ts["dt"] = st.number_input("Interval (menit)", 0.1, 10.0, float(ts["dt"]), 0.1)
            conc = st.number_input(f"Konsentrasi standar ({AN()['unit']})", 0.0, 1e4, float(AN()["std"][3]))
            if st.button("Start time scan", type="primary", disabled=not ready()):
                t = np.arange(0, ts["dur"] + ts["dt"] / 2, ts["dt"])
                Ap = conc * float(absorptivity(CFG["analit"], AN()["lam"])) * CFG["b"]
                ts["data"] = list(zip(t.tolist(), np.round(color_kinetics(CFG["analit"], t, Ap, rng()), 4).tolist()))
                event(f"Time scan {AN()['nama']} {f(ts['dur'], 0)} menit."); st.rerun()
            if not ready():
                st.caption("Nyalakan alat (Persiapan) untuk menjalankan time scan simulasi.")
        else:
            txt = st.text_area("Tempel: waktu (menit) dan Abs per baris", height=160, placeholder="0;0,012\n2;0,301\n5;0,412")
            if st.button("Muat data") and txt.strip():
                res = parse_table(txt)
                if res["ok"]:
                    ts["data"] = sorted((float(a), float(b)) for a, b in zip(res["cols"][0], res["cols"][1]) if np.isfinite(a) and np.isfinite(b))
                    st.rerun()
                else:
                    st.error(res["reason"])
        ts["tol"] = st.number_input("Toleransi kestabilan (% dari maksimum)", 0.1, 10.0, float(ts["tol"]), 0.1)
    with right:
        if len(ts["data"]) < 7:
            st.info("Minimal 7 titik waktu.")
            return
        t = np.array([p[0] for p in ts["data"]]); A = np.array([p[1] for p in ts["data"]])
        w = stable_window(t, A, ts["tol"])
        fig = fig_base("Waktu (menit)", "Absorbansi", 380)
        fig.add_trace(go.Scatter(x=t, y=A, mode="markers", name="A(t)", marker=dict(size=5, color=PAL[0])))
        fig.add_trace(go.Scatter(x=t, y=smooth(A, 7), mode="lines", name="dihaluskan", line=dict(color=PAL[1])))
        if w:
            fig.add_vrect(x0=w["t0"], x1=w["t1"], fillcolor=PAL[3], opacity=0.12, line_width=0, annotation_text="jendela stabil")
        st.plotly_chart(fig, key="tsplot")
        if w:
            st.markdown(f"A maksimum {f(w['Amax'], 4)} pada t = {f(w['tmax'], 1)} menit. Jendela dengan A ≥ {f(100 - ts['tol'], 1)} % maksimum: "
                        f"**{f(w['t0'], 1)}–{f(w['t1'], 1)} menit**. Standar dan sampel sebaiknya dibaca dalam jendela ini dengan selang waktu reaksi yang sama.")
        st.download_button("Unduh CSV", csv_bytes(pd.DataFrame(ts["data"], columns=["Waktu (menit)", "Abs"])), f"{file_stub()}_timescan.csv", "text/csv")


def page_shutdown():
    st.markdown(f"Mengikuti **{IK_META['kode']} bagian C (Mematikan Alat) dan D (Mengeringkan Kuvet)**.")
    left, right = st.columns([1.1, 1], gap="large")
    off = S["off"]
    with right:
        st.subheader("Log Book Alat")
        with st.form("logbook", clear_on_submit=True):
            a, b = st.columns(2)
            nama = a.text_input("Nama jelas")
            nim = b.text_input("NIM")
            mulai = a.text_input("Jam mulai", value=datetime.fromtimestamp(HW["t_on"]).strftime("%H.%M") if HW["t_on"] else "")
            selesai = b.text_input("Jam selesai", value=datetime.now().strftime("%H.%M"))
            kondisi = st.selectbox("Kondisi alat", ["Baik", "Ada kendala (jelaskan)"])
            kejadian = st.text_area("Kejadian / catatan", value="; ".join(e.split("  ", 1)[-1] for e in S["events"] if "PERINGATAN" in e))
            paraf = st.checkbox("Saya menandatangani entri ini dengan nama jelas")
            if st.form_submit_button("Simpan entri log book"):
                if not nama or not paraf:
                    st.error("Nama jelas dan tanda tangan wajib (IK C.1).")
                else:
                    S["logbook"].append({"Tanggal": S["ui"]["tanggal"], "Nama": nama, "NIM": nim, "Kelas/Kelompok": f"{S['ui']['kelas']}/{S['ui']['kelompok']}",
                                         "Mulai": mulai, "Selesai": selesai, "Metode": AN()["nama"], "Kondisi": kondisi, "Kejadian": kejadian, "Tanda tangan": nama})
                    off["logbook"] = True
                    event("Log book diisi.")
                    st.rerun()
        if S["logbook"]:
            lb = pd.DataFrame(S["logbook"])
            st.dataframe(lb, hide_index=True)
            st.download_button("Unduh log book (CSV)", csv_bytes(lb), f"{file_stub()}_logbook.csv", "text/csv")
    with left:
        cur = next((k for k, _ in IK_MATIKAN if not off[k]), None)
        for i, (k, txt) in enumerate(IK_MATIKAN):
            cls = "done" if off[k] else ("now" if k == cur else "")
            sec = "D.1" if k == "kuvet" else f"C.{i + 1}"
            st.markdown(f"<div class='k-step {cls}'><b>{sec}</b> {txt}</div>", unsafe_allow_html=True)
        st.divider()
        if cur and cur != "logbook":
            labels = {"disconnect": "Disconnect", "exit": "File → Exit", "shutdown_pc": "Shutdown Computer", "monitor": "Matikan monitor & printer",
                      "mode": "Tekan MODE", "power_off": "POWER OFF + tutup layar + silika gel", "ups_off": "UPS OFF", "cabut": "Cabut kabel (kecuali UPS)",
                      "rapi": "Area kerja & kursi rapi", "kuvet": "Bilas etanol → aseton, balik kuvet"}
            if st.button(labels[cur], type="primary"):
                off[cur] = True
                if cur == "disconnect":
                    HW["connected"] = False
                elif cur == "exit":
                    HW["uvprobe"] = False
                elif cur == "shutdown_pc":
                    HW["pc_on"] = False
                elif cur == "mode":
                    HW["pc_control"] = False
                elif cur == "power_off":
                    HW.update(power=False, selfcheck=False, t_on=None, silika=True, monitor_alat=False, lid_open=False)
                    S["zero"] = None; S["baseline"] = None
                elif cur == "ups_off":
                    HW["ups"] = False
                elif cur == "cabut":
                    HW["kabel"] = False
                event(f"Matikan alat: {labels[cur]}.")
                st.rerun()
        elif cur == "logbook":
            st.info("Isi Log Book Alat di kolom kanan terlebih dahulu.")
        else:
            st.success("Seluruh prosedur mematikan alat dan pengeringan kuvet selesai.")
            if st.button("Mulai sesi baru (reset status alat)"):
                S["off"] = {k: False for k, _ in IK_MATIKAN}
                S["hw"] = _defaults()["hw"]; st.rerun()


def page_learn():
    tabs = st.tabs(list(LESSONS) + ["Simulasi Lambert–Beer", "Kuis"])
    for i, (k, v) in enumerate(LESSONS.items()):
        with tabs[i]:
            st.markdown(v)
            if k == "Prinsip & optik UV-1700":
                st.plotly_chart(model3d_fig(False, True, True, True, False, None, 460), key="learn3d")
    with tabs[len(LESSONS)]:
        st.latex(r"A_{\mathrm{terukur}} = -\log_{10}\frac{10^{-\varepsilon b C} + s}{1 + s}")
        a, b, c = st.columns(3)
        eps = a.slider("ε (L·mol⁻¹·cm⁻¹)", 1000, 50000, 11100, 100)
        s_ = b.select_slider("Sinar sesat s", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, format_func=lambda x: f"{x * 100:g} %")
        cmax = c.slider("C maks (µmol/L)", 50, 600, 300, 10)
        C = np.linspace(0, cmax, 120)
        A0 = eps * C * 1e-6
        Am = -np.log10((10 ** -A0 + s_) / (1 + s_))
        fig = fig_base("C (µmol/L)", "Absorbansi", 320)
        fig.add_trace(go.Scatter(x=C, y=A0, mode="lines", name="Ideal (A = εbC)", line=dict(color=PAL[0], dash="dash")))
        fig.add_trace(go.Scatter(x=C, y=Am, mode="lines", name="Dengan sinar sesat", line=dict(color=PAL[2])))
        st.plotly_chart(fig, key="beer")
        dev = np.where(A0 > 0, (A0 - Am) / np.where(A0 > 0, A0, 1) * 100, 0)
        idx = np.argmax(dev > 1) if np.any(dev > 1) else None
        st.markdown(f"Penyimpangan negatif > 1 % mulai pada A ideal ≈ **{f(A0[idx], 2)}**." if idx else "Penyimpangan < 1 % di seluruh rentang.")
    with tabs[-1]:
        score = 0
        for i, (q, opts, ans, exp) in enumerate(QUIZ):
            st.markdown(f"**{i + 1}. {q}**")
            ch = st.radio(f"Jawaban soal {i + 1}", list(range(len(opts))), index=None, key=f"quiz{i}",
                          format_func=lambda j, o=opts: f"{chr(65 + j)}. {o[j]}", label_visibility="collapsed")
            if ch is not None:
                (st.success if ch == ans else st.error)(("Benar. " if ch == ans else "Kurang tepat. ") + exp)
                score += int(ch == ans)
        st.metric("Skor", f"{score}/{len(QUIZ)}")


def report_model() -> dict:
    rp = S["report"]
    u = unit()
    srcs = set()
    B = [{"t": "h", "text": "1. Identitas"},
         {"t": "kv", "rows": [("Judul", rp["judul"] or "—"), ("Praktikan / NIM", f"{rp['praktikan'] or '—'} / {rp['nim'] or '—'}"),
                              ("Kelas / Kelompok", f"{S['ui']['kelas'] or '—'} / {S['ui']['kelompok'] or '—'}"), ("Dosen / asisten", rp["dosen"] or "—"),
                              ("Sampel", rp["sampel"] or "—"), ("Tanggal", S["ui"]["tanggal"]), ("Instruksi kerja", f"{IK_META['kode']} (berlaku {IK_META['berlaku']})")]},
         {"t": "h", "text": "2. Instrumen dan kondisi pengukuran"},
         {"t": "kv", "rows": [("Instrumen", "Shimadzu UV-1700 PharmaSpec, berkas ganda, lebar pita 1 nm (pembacaan dari instrumen virtual)"),
                              ("Metode", AN()["nama"]), ("Kuvet / tebal", f"{KUVET[CFG['kuvet']]['nama']} / {f(CFG['b'], 1)} cm"),
                              ("Blanko", "Blanko reagen" if CFG["blanko"] == "reagen" else "Pelarut/akuades"),
                              ("λ pengukuran (WL1)", f"{f(S['pho']['wl'], 1)} nm" if S["pho"] else "—"),
                              ("Tipe kalibrasi", f"{S['pho']['type']}, Zero Interception {'ON' if S['pho']['zero'] else 'OFF'}" if S["pho"] else "—")]}]
    sp = next((s for s in S["spectra"] if s["id"] == S["active"]), S["spectra"][-1] if S["spectra"] else None)
    n = 3
    if sp:
        srcs.add(sp["src"])
        pk = [p for p in peak_pick(sp["lam"], sp["A"], S["sp"]["thr"]) if p["Jenis"] == "Puncak"]
        B += [{"t": "h", "text": f"{n}. Spektrum absorpsi dan λmaks"},
              {"t": "img", "png": spectrum_png([sp], [p["λ (nm)"] for p in pk[:2]]), "cap": f"Gambar 1. Spektrum {sp['name']}."},
              {"t": "p", "text": (f"λmaks teramati {f(pk[0]['λ (nm)'], 1)} nm (A = {f(pk[0]['Abs'], 4)}, FWHM {f(pk[0]['FWHM (nm)'], 1)} nm). "
                                  "Pengukuran kuantitatif pada λmaks meminimalkan sensitivitas terhadap ketidaktepatan panjang gelombang karena dA/dλ ≈ 0.") if pk else "Puncak tidak terdeteksi."}]
        n += 1
    fit = current_fit()
    sres = []
    if fit and fit.ok:
        srcs.add(S["src"])
        sres = [(s, sample_res(s, fit)) for s in S["smp"]]
        sres = [(s, r) for s, r in sres if r]
        B += [{"t": "h", "text": f"{n}. Kurva kalibrasi"},
              {"t": "table", "head": ["No", f"C ({u})", "A", "Â", "Residual"],
               "rows": [[i + 1, g(x), f(y, 4), f(fit.yhat[i], 4), f(fit.res[i], 4)] for i, (x, y) in enumerate(zip(fit.x, fit.y))]},
              {"t": "img", "png": calib_png(fit, u, [(s["name"], r["c_meas"], r["A_mean"]) for s, r in sres]), "cap": "Gambar 2. Kurva kalibrasi dan residual."},
              {"t": "kv", "rows": [("Persamaan", fit.equation()), ("r / R²", f"{f(fit.r, 5)} / {f(fit.r2, 5)}"), ("s_y/x", g(fit.syx, 3)),
                                   ("LOD / LOQ", f"{g(fit.lod, 3)} / {g(fit.loq, 3)} {u}"), ("Rentang", f"{g(fit.xmin, 3)}–{g(fit.xmax, 3)} {u}")]}]
        for p in calib_interpretation(fit, u):
            B.append({"t": "p", "text": re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)})
        n += 1
        if sres:
            for s, _ in sres:
                srcs.add(s["src"])
            B += [{"t": "h", "text": f"{n}. Hasil penetapan kadar"},
                  {"t": "table", "head": ["Sampel", "n", "Ā", "RSD %", "C terukur", "FP", "C sampel ± SK95", "mg/kg"],
                   "rows": [[s["name"], r["n"], f(r["A_mean"], 4), f(r["rsd"], 2) if r["n"] > 1 else "-", g(r["c_meas"]), g(r["fp"]),
                             f"{g(r['c_final'])} ± {g(r['ci_final'], 2)} {u}", g(r["kadar"]) if r["mass_ok"] else "-"] for s, r in sres]}]
            for s, r in sres:
                if r["flags"]:
                    B.append({"t": "note", "text": s["name"] + ": " + " ".join(r["flags"])})
            n += 1
    pr, rc = precision(), recovery()
    rows = []
    if fit and fit.ok:
        rows.append(("Linearitas", f"R² = {f(fit.r2, 4)} (kriteria ≥ {f(CR['r2_min'], 3)}): {'memenuhi' if fit.r2 >= CR['r2_min'] else 'tidak memenuhi'}"))
    if pr:
        rows.append(("Presisi", f"n = {pr['d']['n']}, RSD = {f(pr['d']['rsd'], 2)} % (≤ {f(CR['rsd_max'], 2)} %): {'memenuhi' if pr['pass'] else 'tidak memenuhi'}"))
    if rc:
        rows.append(("Akurasi", f"%R = {f(rc['R'], 2)} % ({f(CR['rec_min'], 0)}–{f(CR['rec_max'], 0)} %): {'memenuhi' if rc['pass'] else 'tidak memenuhi'}"))
    if rows:
        B += [{"t": "h", "text": f"{n}. Evaluasi mutu data"}, {"t": "kv", "rows": rows}]
        n += 1
    concl = [f"Kadar {AN()['nama']} dalam {s['name']} = {g(r['c_final'])} ± {g(r['ci_final'], 2)} {u} (SK 95 %, n = {r['n']})"
             + (f", setara {g(r['kadar'])} mg/kg." if r["mass_ok"] else ".") for s, r in sres]
    if not concl:
        concl = ["Data belum cukup untuk kesimpulan kuantitatif."]
    simulated = bool(srcs & {"simulasi", "contoh", "campuran"})
    if simulated:
        concl.append("Sebagian/seluruh data berasal dari model simulasi atau data contoh; hasil merupakan latihan.")
    B += [{"t": "h", "text": f"{n}. Kesimpulan"}] + [{"t": "p", "text": c} for c in concl]
    if rp["catatan"]:
        B += [{"t": "h", "text": "Catatan praktikan"}, {"t": "p", "text": rp["catatan"]}]
    if S["logbook"]:
        lb = S["logbook"][-1]
        B += [{"t": "h", "text": "Log book"}, {"t": "kv", "rows": [(k, v) for k, v in lb.items()]}]
    B.append({"t": "note", "text": "Sumber data: " + (", ".join(sorted(srcs)) or "—") + "."})
    return {"title": "LAPORAN ANALISIS SPEKTROFOTOMETRI UV-VIS", "subtitle": f"{rp['judul']} · {S['ui']['tanggal']}", "blocks": B,
            "simulated": simulated, "author": rp["praktikan"]}


def page_report():
    rp = S["report"]
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        for k, lab in [("judul", "Judul percobaan"), ("praktikan", "Nama praktikan"), ("nim", "NIM"), ("dosen", "Dosen / asisten"), ("sampel", "Identitas sampel")]:
            rp[k] = st.text_input(lab, rp[k])
        rp["catatan"] = st.text_area("Catatan / pembahasan tambahan", rp["catatan"])
        if st.button("Buat laporan PDF", type="primary"):
            with st.spinner("Menyusun laporan…"):
                S["_pdf"] = build_pdf(report_model())
        if S.get("_pdf"):
            st.download_button("Unduh laporan PDF", S["_pdf"], f"{file_stub()}_laporan.pdf", "application/pdf")
    with right:
        st.markdown("**Pratinjau isi**")
        for b in report_model()["blocks"]:
            if b["t"] == "h":
                st.markdown(f"#### {b['text']}")
            elif b["t"] == "p":
                st.markdown(b["text"].replace("<b>", "**").replace("</b>", "**"))
            elif b["t"] == "note":
                st.caption(b["text"])
            elif b["t"] == "kv":
                st.dataframe(pd.DataFrame(b["rows"], columns=["", "Nilai"]).astype(str), hide_index=True)
            elif b["t"] == "table":
                st.dataframe(pd.DataFrame(b["rows"], columns=b["head"]).astype(str), hide_index=True)
            elif b["t"] == "img":
                st.image(b["png"], caption=b["cap"])


def page_project():
    st.markdown("Simpan seluruh sesi (metode, standar, sampel, spektrum, log book) sebagai JSON dan muat kembali kapan saja.")

    def _ser(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    proj = {k: v for k, v in S.items() if not k.startswith("_")}
    st.download_button("Ekspor proyek (.json)", json.dumps(proj, default=_ser, ensure_ascii=False).encode("utf-8"), f"{file_stub()}_proyek.json", "application/json")
    up = st.file_uploader("Impor proyek (.json)", type=["json"])
    if up is not None and st.button("Muat proyek"):
        try:
            o = json.loads(up.getvalue())
            new = _defaults()
            for k, v in o.items():
                if k in new:
                    new[k] = v
            new["hw"]["t_on"] = None; new["hw"]["power"] = False; new["hw"]["selfcheck"] = False
            new["zero"] = None; new["baseline"] = None
            st.session_state.S = new
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Impor gagal: {e}")
    st.divider()
    st.markdown("**Log aktivitas sesi**")
    st.code("\n".join(S["events"]) or "—", language=None)
    if st.button("Reset seluruh sesi"):
        st.session_state.S = _defaults(); st.rerun()


st.title(page.split(" · ")[-1])
{"Beranda": page_home, "Model 3D Instrumen": page_model3d, "1 · Persiapan Alat": page_prep, "2 · Panel UV-1700": page_panel,
 "3 · Mode Spektrum (λmaks)": page_spectrum, "4 · Mode Fotometrik": page_photometric, "5 · Evaluasi Kalibrasi": page_calibration,
 "6 · Validasi Metode": page_validation, "7 · Time Scan (Kestabilan Warna)": page_timescan,
 "8 · Mematikan Alat & Log Book": page_shutdown, "Pembelajaran & Kuis": page_learn, "Laporan PDF": page_report,
 "Proyek & Data": page_project}[page]()
