"""Download the spectral templates used by the calculator (Pickles stars, Kinney-Calzetti galaxies, QSO composite,
Nugent supernovae at B maximum) and store them as CSV in data/templates with a templates.json manifest.

Run:  python scripts/fetch_templates.py
"""
import gzip, io, json, re, urllib.request
from pathlib import Path
import numpy as np
from astropy.io import fits

OUT = Path(__file__).resolve().parent.parent / 'data' / 'templates'
CDBS = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid'
NUGENT = 'https://c3.lbl.gov/nugent/templates/'
manifest = {}


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (7DT calculator template fetch)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fits_table(url):
    with fits.open(io.BytesIO(get(url))) as h:
        d = h[1].data
        return np.asarray(d['WAVELENGTH'], float), np.asarray(d['FLUX'], float)


def write(name, wave, flux, meta):
    good = np.isfinite(wave) & np.isfinite(flux) & (flux >= 0)
    wave, flux = wave[good], flux[good]
    order = np.argsort(wave); wave, flux = wave[order], flux[order]
    keep = np.concatenate([[True], np.diff(wave) > 0]); wave, flux = wave[keep], flux[keep]
    if meta.get('extend_to') and wave[-1] < meta['extend_to']:
        # extend redward with constant f_nu (f_lambda ~ lambda^-2), anchored on the median of the last 200 A
        anchor = np.median(flux[wave > wave[-1] - 200])
        ext = np.arange(wave[-1] + 10, meta['extend_to'] + 10, 10.0)
        wave = np.concatenate([wave, ext]); flux = np.concatenate([flux, anchor * (wave[-len(ext) - 1] / ext) ** 2])
        meta['note'] = meta.get('note', '') + f' Extended beyond {int(wave[-len(ext) - 1])} A with constant f_nu.'
    path = OUT / meta['file']
    hdr = f"{name}: {meta['source']}. {meta.get('note', '')}\nwavelength[A] f_lambda[relative]"
    np.savetxt(path, np.column_stack([wave, flux]), header=hdr, fmt='%.2f %.6e')
    manifest[name] = {k: v for k, v in meta.items() if k != 'extend_to'}
    manifest[name]['lam_min'] = float(wave[0]); manifest[name]['lam_max'] = float(wave[-1])
    print(f'{name:24s} {wave[0]:8.0f}-{wave[-1]:6.0f} A  n={len(wave)}')


# ---- Pickles (1998) UVK stellar library
with fits.open(io.BytesIO(get(f'{CDBS}/pickles/dat_uvk/pickles_uk.fits'))) as h:
    idx = {str(r['SPTYPE']).strip().lower(): str(r['FILENAME']).strip() for r in h[1].data}
print('Pickles index entries:', len(idx), sorted(idx)[:8], '...')
for sp in ['o5v', 'b0v', 'a0v', 'f0v', 'g2v', 'k0v', 'm0v']:
    fname = idx[sp]
    w, f = fits_table(f'{CDBS}/pickles/dat_uvk/{fname}.fits')
    write(f'Star: {sp.upper()}', w, f, {'file': f'pickles_{sp}.csv', 'category': 'Star', 'source': 'Pickles (1998) UVK stellar library, STScI CDBS',
                                         'note': f'{sp.upper()} main-sequence star.'})

# ---- Kinney-Calzetti (1996) galaxy templates
for key, label in [('elliptical', 'Elliptical'), ('s0', 'S0'), ('sa', 'Sa'), ('sb', 'Sb'), ('sc', 'Sc'), ('starb1', 'Starburst')]:
    w, f = fits_table(f'{CDBS}/kc96/{key}_template.fits')
    write(f'Galaxy: {label}', w, f, {'file': f'kc96_{key}.csv', 'category': 'Galaxy', 'source': 'Kinney et al. (1996) galaxy templates, STScI CDBS',
                                       'note': ('Starburst SB1 (E(B-V) < 0.10).' if key == 'starb1' else f'{label} galaxy.'), 'extend_to': 11000})

# ---- QSO: Vanden Berk et al. (2001) SDSS median composite (rest frame 800-8555 A), the template used by the Gemini ITC
IOP = 'https://content.cld.iop.org/journals/1538-3881/122/2/549/revision1/datafile1.txt'
rows = []
for line in get(IOP).decode('utf-8', 'replace').splitlines():
    parts = line.split()
    if len(parts) < 3:
        continue
    try:
        rows.append((float(parts[0]), float(parts[1])))
    except ValueError:
        continue   # header lines
vb = np.array(rows)
write('QSO (SDSS, Vanden Berk 2001)', vb[:, 0], vb[:, 1], {'file': 'qso_vandenberk2001.csv', 'category': 'AGN',
      'source': 'Vanden Berk et al. (2001) SDSS median composite quasar spectrum (AJ 122, 549, electronic Table 1)',
      'note': 'Rest-frame 800-8555 A, relative f_lambda; set z.', 'extend_to': 11000})

# ---- QSO: Selsing et al. (2016) X-shooter composite (rest frame 1000-11350 A), through the VizieR TSV service
VIZIER = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/585/A87/spectrum&-out.max=unlimited&-out.all'
rows = []
for line in get(VIZIER).decode('utf-8', 'replace').splitlines():
    parts = line.split('\t')
    if line.startswith('#') or len(parts) < 3:
        continue
    try:
        rows.append((float(parts[1]), float(parts[2])))
    except ValueError:
        continue   # header, unit and separator lines
qso = np.array(rows)
write('QSO (X-shooter, Selsing 2016)', qso[:, 0], qso[:, 1], {'file': 'qso_selsing2016.csv', 'category': 'AGN',
                                    'source': 'Selsing et al. (2016) X-shooter composite of luminous 1<z<2.1 SDSS quasars, VizieR J/A+A/585/A87',
                                    'note': 'Rest-frame 1000-11350 A, f_lambda normalised at 6800 A; set z.'})

# ---- Nugent SN templates at maximum light
import urllib.error
for names, label, note in [(['sn1a_flux.v1.2.dat'], 'SN Ia (max)', 'Normal SN Ia, phase 0 d (Nugent, Kim & Perlmutter 2002).'),
                           (['sn1bc_flux.v1.1.dat', 'sn1bc_flux.v1.2.dat'], 'SN Ib/c (max)', 'SN Ib/c, phase 0 d (Levan et al. 2005).'),
                           (['sn2p_flux.v1.2.dat', 'sn2p_flux.v1.1.dat'], 'SN II-P (max)', 'SN II-P, phase 0 d (Gilliland, Nugent & Phillips 1999).')]:
    raw = None
    for fname in names:
        try:
            raw = get(NUGENT + fname); break
        except urllib.error.HTTPError as exc:
            print('not found', fname, exc.code)
    if raw is None:
        continue
    pat = fname.split('_')[0] + '_flux'
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    data = np.loadtxt(io.StringIO(raw.decode()))
    phases = data[:, 0]
    # the file's epoch 0 is the first template day (about 20 d before maximum for SNe Ia):
    # pick the epoch of maximum B-band synthetic flux as 'maximum light' (the usual SN phase reference)
    vcurve = np.loadtxt(OUT.parent / 'transmission' / 'B.csv', delimiter=',', skiprows=1)
    best, best_flux = None, -1
    for ph in np.unique(phases):
        sel = phases == ph
        w, f = data[sel, 1], data[sel, 2]
        t = np.interp(w, vcurve[:, 0], vcurve[:, 1], left=0, right=0)
        vflux = np.trapezoid(f * w * t, w)
        if vflux > best_flux:
            best, best_flux = ph, vflux
    sel = phases == best
    print(f'{label}: B-band peak at file epoch {best:g} d (grid {phases.min():g}-{phases.max():g} d)')
    write(label, data[sel, 1], data[sel, 2], {'file': f'nugent_{pat.split("_")[0]}_max.csv', 'category': 'Supernova',
                                              'source': f'Nugent template {fname} (https://c3.lbl.gov/nugent/templates/), epoch {best:g} d of the file = B-band maximum',
                                              'note': note.replace('phase 0 d', 'at maximum light'), 'extend_to': 11000})

with open(OUT / 'templates.json', 'w') as fh:
    json.dump(manifest, fh, indent=2)
print('manifest entries:', list(manifest))
