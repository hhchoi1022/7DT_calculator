"""
7DT Exposure time / SNR: expected SNR of the targets in every filter of an observation mode, or the
number of frames needed to reach a target SNR.
Run with:  streamlit run etc_app.py
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from astropy.time import Time

from sevendt_calc import config, etc
from sevendt_calc import templates as tpl
from sevendt_calc.photometry import MAG_INPUT_FILTERS
from sevendt_calc.targets import Target
from sevendt_calc.photometry import Spectrum
from ui_common import FLUX_UNIT_LABELS, SPECTRUM_FILE_TYPES, live_config, obsmode_selector

st.set_page_config(page_title='7DT Exposure time / SNR', page_icon='📈', layout='wide')

TYPE_OPTIONS = ['Point source', 'Extended source']
SPECTRUM_CATEGORIES = ['Uploaded spectrum', 'Star', 'Galaxy', 'QSO', 'SN', 'Blackbody', 'Power law']
SPECTRUM_CHOICES = {                        # category -> {what the user sees: template key}
    'Star': {'O5V': 'Star: O5V', 'B0V': 'Star: B0V', 'A0V': 'Star: A0V', 'F0V': 'Star: F0V', 'G2V': 'Star: G2V', 'K0V': 'Star: K0V', 'M0V': 'Star: M0V'},
    'Galaxy': {'Spiral': 'Galaxy: Sb', 'Elliptical': 'Galaxy: Elliptical'},
    'SN': {'Ia (max)': 'SN Ia (max)', 'Ib/c (max)': 'SN Ib/c (max)', 'II-P (max)': 'SN II-P (max)'},
}
SPECTRUM_SINGLE = {'Uploaded spectrum': tpl.UPLOAD_KEY, 'QSO': 'QSO (SDSS, Vanden Berk 2001)', 'Blackbody': 'BB', 'Power law': 'PL'}
ANALYTIC_KEYS = ('BB', 'PL')


def analytic_spectrum(kind: str, value: float) -> Spectrum:
    """A blackbody of temperature `value` [K] or a power law S_lambda ~ lambda**value, on 3000-11000 A (shape only)."""
    w = np.arange(3000.0, 11000.0, 5.0)
    if kind == 'BB':
        h, c, k = 6.62607e-27, 2.99792458e10, 1.380649e-16
        lam = w * 1e-8
        with np.errstate(over='ignore'):
            f = 1.0 / (lam ** 5 * np.expm1(h * c / (lam * k * float(value))))
        return Spectrum.from_arrays(w, f / f.max(), 'flam', source=f'blackbody {value:g} K')
    f = (w / 5500.0) ** float(value)
    return Spectrum.from_arrays(w, f, 'flam', source=f'power law S_lambda ~ lambda^{value:g}')
# Sky and seeing choices offered instead of a date (Moon separation and hours since sunset are fixed)
SKY_PRESETS = {'20%/Darkest': 0.0, '50%/Dark': 0.5, '80%/Grey': 0.8, 'Any/Bright': 1.0}      # Moon illumination
SEEING_PRESETS = {'20%/Best': 1.8, '70%/Good': 2.6, '85%/Poor': 3.0, 'Any': 3.5}                 # seeing [arcsec]
MOON_SEPARATION = 90.0
HOURS_SINCE_SUNSET = 5.0
SAMPLE_SPECTRUM = config.DATA_DIR / 'sample_spectrum.txt'


@st.cache_resource(show_spinner='Loading the depth model...')
def get_calculator() -> etc.ExposureCalculator:
    return etc.ExposureCalculator(etc.DepthModel())


@st.cache_data(show_spinner=False)
def typical_seeing() -> float:
    """Median seeing of the frames the depth model was fitted on (the auto value of the seeing input)."""
    model = etc.DepthModel().model
    values = [m['medians'][3] for m in model.values() if len(m.get('medians', [])) > 3]
    return round(float(np.median(values)), 1) if values else 2.0


def target_inputs() -> Target | None:
    """One target: type, magnitude or surface brightness with its filter, and the spectrum."""
    c1, c2, c3, cz = st.columns([1.2, 1, 0.8, 0.8])
    ttype = c1.selectbox('Target type', TYPE_OPTIONS, key='etc_type')
    extended = ttype.startswith('Extended')
    if extended:
        mag = c2.number_input('Surface brightness [mag/arcsec²]', 5.0, 35.0, 22.0, 0.1, key='etc_sb')
    else:
        mag = c2.number_input('Magnitude', -2.0, 30.0, 19.0, 0.1, key='etc_mag')
    band = c3.selectbox('Filter of the magnitude', list(MAG_INPUT_FILTERS), index=list(MAG_INPUT_FILTERS).index('r'), key='etc_band',
                        help='B V R I: Vega; g r i: AB')
    redshift = cz.number_input('Redshift', 0.0, 12.0, 0.0, 0.01, format='%.3f', key='etc_z',
                               help='The spectrum (template, blackbody, power law or uploaded file) is taken as rest frame and shifted to this redshift; '
                                    'the magnitude is kept as the observed value in its filter.')
    c4, c5 = st.columns([1.2, 1.8])
    category = c4.selectbox('Spectrum', SPECTRUM_CATEGORIES, index=SPECTRUM_CATEGORIES.index('Star'), key='etc_spectrum')
    spec = None
    if category in SPECTRUM_CHOICES:
        choices = SPECTRUM_CHOICES[category]
        default = 'G2V' if category == 'Star' else list(choices)[0]
        pick = c5.selectbox(f'{category} type', list(choices), index=list(choices).index(default), key=f'etc_spectrum_{category}')
        key = choices[pick]
    else:
        key = SPECTRUM_SINGLE[category]
    if key == 'BB':
        temp = c5.number_input('Blackbody temperature [K]', 1000.0, 200000.0, 10000.0, 500.0, key='etc_bb_temp')
        spec = analytic_spectrum('BB', temp)
    elif key == 'PL':
        alpha = c5.number_input('Power-law index α  (S_λ ∝ λ^α)', -6.0, 6.0, -3.0, 0.1, key='etc_pl_alpha')
        spec = analytic_spectrum('PL', alpha)
    if key == tpl.UPLOAD_KEY:
        with c5:
            u1, u2 = st.columns([2, 1.2])
            upload = u1.file_uploader('Spectrum file (two columns: wavelength [Å], flux)', type=SPECTRUM_FILE_TYPES, key='etc_upload')
            unit = u2.radio('Flux unit', FLUX_UNIT_LABELS, key='etc_unit')
            if SAMPLE_SPECTRUM.exists():
                st.download_button('Download a sample spectrum', SAMPLE_SPECTRUM.read_bytes(), file_name='sample_spectrum.txt',
                                   mime='text/plain', key='etc_dl_sample')
        if upload is None:
            st.error('Upload the spectrum file (or choose a template).')
            return None
        try:
            spec = Spectrum.from_text(upload.getvalue(), 'fnu' if unit.startswith('f_nu') else 'flam', source=upload.name)
        except ValueError as exc:
            st.error(f'Could not read the spectrum: {exc}')
            return None
    return Target(name='target', mag=float(mag), mag_filter=band, spectrum_type=None if key in ANALYTIC_KEYS else key,
                  spectrum=spec, extended=extended, redshift=float(redshift))


def conditions_widgets(prefix: str):
    """Sky background and image quality categories; 'Any' image quality takes a seeing value (0 = typical)."""
    sky = st.radio('Sky Background', list(SKY_PRESETS), horizontal=True, key=f'{prefix}_sky')
    c1, c2 = st.columns([2, 1], vertical_alignment='center')
    see = c1.radio('Image Quality', list(SEEING_PRESETS), index=1, horizontal=True, key=f'{prefix}_seeing_choice')
    seeing = SEEING_PRESETS[see]
    slot = c2.empty()                  # a placeholder at a fixed position: sent at once, so the seeing input vanishes immediately
    if see == 'Any':                   # instead of at the end of the rerun (when Streamlit removes stale widgets)
        typed = slot.number_input('Seeing [arcsec] (0 = typical)', 0.0, 6.0, 0.0, 0.1, key=f'{prefix}_seeing_any')
        seeing = typed if typed > 0 else typical_seeing()
    return etc.conditions_for(seeing=seeing, moon_phase=SKY_PRESETS[sky], moon_separation=MOON_SEPARATION, hour_since_sunset=HOURS_SINCE_SUNSET)


def settings_block(prefix: str, cfg: dict, calc_mode: str):
    if calc_mode == 'snr':
        c1, c2 = st.columns(2)
        exptime = c1.number_input('Exposure time per frame [s]', 1.0, 7200.0, 100.0, 10.0, key=f'{prefix}_exp')
        value = int(c2.number_input('Frames per filter and unit', 1, 999, 3, key=f'{prefix}_cnt',
                                    help='Filters observed by several units at once get count x units frames stacked'))
        if exptime < etc.SHORT_EXPTIME:
            st.warning(etc.SHORT_EXPTIME_NOTE)
    else:
        exptime = 100.0                       # the standard 7DT frame, used only to express the total time in frames
        value = st.number_input('Target SNR', 1.0, 1000.0, 10.0, 1.0, key=f'{prefix}_snr')
    om = obsmode_selector(prefix, cfg)
    return {'exptime': exptime, 'value': value, 'obsmode': om}


def figures(target, results, cond, mode_key: str, value: float, seed) -> tuple:
    """Both figures as interactive plotly charts (drawn in the browser, so no server-side rendering cost)."""
    label = 'Extended source' if target.extended else 'Point source'
    if mode_key == 'snr':
        fig1 = etc.plot_snr_vs_exptime_interactive(target, results, cond)
        title = f'{label}: expected 7DT photometry'
    else:
        fig1 = etc.plot_snr_vs_total_time_interactive(target, results, cond, value)
        title = f'{label}: photometry with the frames needed for SNR {value:g}'
    fig2 = etc.plot_spectrum_points_interactive(target, results, noise_seed=seed, title=title)
    return fig1, fig2


def main():
    st.title('7DT Exposure time / SNR')
    st.markdown('The predictions are based on real 7DT observations taken in 2025 with 100 s exposures; '
                'for much shorter exposures the SNR estimate may be inaccurate.')
    cfg = live_config()
    calc = get_calculator()

    target = target_inputs()
    cond = conditions_widgets('etc')
    calc_mode = st.radio('Calculation', ['SNR from exposure time', 'Exposure time from SNR'], horizontal=True, key='etc_calc')
    mode_key = 'snr' if calc_mode.startswith('SNR') else 'time'
    s = settings_block('etc', cfg, mode_key)
    noise = st.checkbox('Draw a random noise realisation of the expected photometry', value=True, key='etc_noise')
    if 'etc_seed' not in st.session_state:                 # one realisation per session, so the figure cache can be reused
        st.session_state.etc_seed = int(Time.now().unix) % 10000
    if target is None or s['obsmode'] is None or not s['obsmode'].filters:
        return

    filters = s['obsmode'].filters
    note = target.saturation_note(filters, extended_shift=etc.extended_saturation_shift(cond.seeing))
    if note:
        st.warning(note)
    units = s['obsmode'].filter_multiplicity()
    if mode_key == 'snr':
        results = calc.snr(target, filters, s['exptime'], s['value'], cond, units=units)
        df = pd.DataFrame([r.row() for r in results])
    else:
        results = calc.exptime_needed(target, filters, s['exptime'], s['value'], cond, units=units)
        df = pd.DataFrame([r.row_time_needed() for r in results])
    if target.extended:
        st.caption('Extended source: mag_AB is the surface brightness [mag/arcsec²], UL5_single / UL5_stacked are the 5σ '
                   'surface-brightness limits over 1 arcsec², and SNR_single / SNR_stacked and mag_err are per arcsec².')
    st.dataframe(df.round(3), hide_index=True, width='stretch')
    st.download_button('Download table (CSV)', df.to_csv(index=False).encode('utf-8'), file_name=f'{mode_key}.csv', mime='text/csv', key='dl_table')
    seed = st.session_state.etc_seed if noise else None
    fig1, fig2 = figures(target, results, cond, mode_key, s['value'], seed)
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig1, width='stretch', key='etc_fig1',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'snr_vs_exptime' if mode_key == 'snr' else 'snr_vs_total_time', 'scale': 3}})
    c2.plotly_chart(fig2, width='stretch', key='etc_fig2',
                    config={'displaylogo': False, 'toImageButtonOptions': {'filename': 'photometry', 'scale': 3}})
    unavailable = [r.filter for r in results if not r.available]
    if unavailable:
        st.warning('Not evaluated (no depth model or magnitude): ' + ', '.join(unavailable))


if __name__ == '__main__':
    main()
