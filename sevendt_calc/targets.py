"""Targets: coordinates (degrees or sexagesimal), magnitudes, optional spectra, name resolving."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord

from .photometry import MAG_INPUT_FILTERS, Spectrum, get_filter_curve, magnitude_system, to_ab
from . import templates as tpl

SATURATION_MAG = 11.0     # sources brighter than this (any band) are flagged as saturation risks

_NUMBER_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def _is_number(text: str) -> bool:
    return bool(_NUMBER_RE.match(text.strip()))


def _clean(text) -> str:
    if text is None:
        return ''
    text = str(text).strip()
    if text.lower() in ('nan', 'none', 'null'):
        return ''
    # unicode degree / minute / second marks -> letters astropy understands
    return text.replace('°', 'd').replace('′', 'm').replace('″', 's').replace("'", 'm').replace('"', 's')


def parse_ra(text) -> float:
    """RA in degrees from '123.45', '11:02:03.4', '11h02m03.4s' or '11 02 03.4' (sexagesimal = hours)."""
    text = _clean(text)
    if not text:
        raise ValueError('RA is empty')
    if _is_number(text):
        value = float(text)
        if not 0 <= value < 360:
            value = value % 360.0
        return value
    try:
        angle = Angle(text, unit=u.hourangle)
    except Exception as exc:
        raise ValueError(f'Cannot parse RA {text!r}: {exc}') from exc
    return float(angle.wrap_at(360 * u.deg).deg)


def parse_dec(text) -> float:
    """Dec in degrees from '-30.5', '-30:12:11', '-30d12m11s' or '+30 12 11'."""
    text = _clean(text)
    if not text:
        raise ValueError('Dec is empty')
    if _is_number(text):
        value = float(text)
    else:
        try:
            value = float(Angle(text, unit=u.deg).deg)
        except Exception as exc:
            raise ValueError(f'Cannot parse Dec {text!r}: {exc}') from exc
    if not -90 <= value <= 90:
        raise ValueError(f'Dec {value} is outside [-90, 90]')
    return value


def parse_coordinates(ra_text, dec_text) -> tuple[float, float]:
    """Parse an (RA, Dec) pair. A single field holding both ('11 02 03.4 -30 12 11') is split."""
    ra_text, dec_text = _clean(ra_text), _clean(dec_text)
    if ra_text and not dec_text:
        parts = ra_text.split()
        if len(parts) == 2:
            ra_text, dec_text = parts
        elif len(parts) == 6:
            ra_text, dec_text = ' '.join(parts[:3]), ' '.join(parts[3:])
    return parse_ra(ra_text), parse_dec(dec_text)


def resolve_name(name: str) -> tuple[float, float]:
    """
    Resolve a name to RA/Dec [deg]: a 7DS tile ID (T01234) gives the tile centre from the tile
    catalogue, any other name goes through Sesame (SIMBAD / NED / VizieR; needs internet access).
    """
    name = str(name).strip()
    if not name:
        raise ValueError('Empty name')
    from . import tiles                              # lazy: tiles imports this module
    centre = tiles.tile_centre(name)
    if centre is not None:
        return centre
    try:
        coord = SkyCoord.from_name(name)
    except Exception as exc:
        raise ValueError(f"Could not resolve '{name}': {exc}") from exc
    return float(coord.ra.deg), float(coord.dec.deg)


def to_sexagesimal(ra: float, dec: float, precision: int = 2) -> tuple[str, str]:
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    return (str(c.ra.to_string(unit=u.hourangle, sep=':', precision=precision, pad=True)),
            str(c.dec.to_string(unit=u.deg, sep=':', precision=precision - 1, alwayssign=True, pad=True)))


@dataclass
class Target:
    name: str = ''
    ra: float | None = None            # deg
    dec: float | None = None           # deg
    mag: float | None = None           # magnitude in `mag_filter`
    mag_filter: str | None = None      # one of MAG_INPUT_FILTERS
    spectrum: Spectrum | None = None   # uploaded spectrum (absolute flux) or a template shape
    spectrum_type: str | None = None   # template key, tpl.FLAT_KEY, tpl.UPLOAD_KEY or None (= flat)
    spectrum_file: str = ''            # file name an Upload row refers to (bulk import)
    redshift: float = 0.0
    extended: bool = False             # True: `mag` is a surface brightness [mag/arcsec^2]
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.name = str(self.name or '').strip()
        self.spectrum_file = str(self.spectrum_file or '').strip()
        self.redshift = 0.0 if self.redshift is None or (isinstance(self.redshift, float) and math.isnan(self.redshift)) else float(self.redshift)
        if self.spectrum_type is not None:
            self.spectrum_type = str(self.spectrum_type).strip() or None
        if self.spectrum_type is not None and self.spectrum_type not in tpl.template_keys():
            raise ValueError(f'Unknown spectrum type {self.spectrum_type!r}')
        if self.spectrum is None and tpl.is_template(self.spectrum_type):
            self.spectrum = tpl.load_template(self.spectrum_type, self.redshift)
        for attr in ('ra', 'dec', 'mag'):
            value = getattr(self, attr)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                setattr(self, attr, None)
            else:
                setattr(self, attr, float(value))
        if self.mag_filter is not None:
            self.mag_filter = str(self.mag_filter).strip() or None
        if self.mag_filter is not None and self.mag_filter not in MAG_INPUT_FILTERS:
            raise ValueError(f'Magnitude filter must be one of {MAG_INPUT_FILTERS}, got {self.mag_filter!r}')
        self._effective_spectrum = None

    # ----------------------------------------------------------------- state
    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.has_coord:
            return f'({self.ra:.4f}, {self.dec:+.4f})'
        return 'unnamed target'

    @property
    def has_coord(self) -> bool:
        return self.ra is not None and self.dec is not None

    @property
    def has_mag(self) -> bool:
        return self.mag is not None and self.mag_filter is not None

    @property
    def has_spectrum(self) -> bool:
        return self.spectrum is not None

    @property
    def uses_template(self) -> bool:
        """A template only gives the spectral shape: the magnitude is needed to set its level."""
        return tpl.is_template(self.spectrum_type)

    @property
    def has_photometry(self) -> bool:
        if self.uses_template:
            return self.has_mag
        return self.has_mag or self.has_spectrum

    @property
    def spectrum_label(self) -> str:
        if self.uses_template:
            return self.spectrum_type + (f' (z = {self.redshift:g})' if self.redshift else '')
        if self.has_spectrum:
            return self.spectrum.source or 'uploaded spectrum'
        return 'flat (constant f_nu)'

    @property
    def coord(self) -> SkyCoord | None:
        if not self.has_coord:
            return None
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')

    @property
    def ra_hms(self) -> str:
        return to_sexagesimal(self.ra, self.dec)[0] if self.has_coord else ''

    @property
    def dec_dms(self) -> str:
        return to_sexagesimal(self.ra, self.dec)[1] if self.has_coord else ''

    @property
    def mag_ab(self) -> float | None:
        """The input magnitude converted to AB (Vega bands B, V, R, I get the Blanton & Roweis offsets)."""
        return to_ab(self.mag, self.mag_filter) if self.has_mag else None

    @property
    def mag_system(self) -> str:
        return magnitude_system(self.mag_filter) if self.mag_filter else ''

    def missing(self, need_coord: bool = True, need_photometry: bool = False) -> list[str]:
        """Names of required input fields that are empty."""
        out = []
        if need_coord:
            if self.ra is None:
                out.append('ra')
            if self.dec is None:
                out.append('dec')
        if need_photometry and (self.uses_template or not self.has_spectrum):
            if self.mag is None:
                out.append('mag')
            if self.mag_filter is None:
                out.append('mag_filter')
        return out

    # ------------------------------------------------------------- photometry
    def effective_spectrum(self) -> Spectrum | None:
        """
        The spectrum used for synthetic photometry: the input spectrum, rescaled to the input
        magnitude when both are given. None when no spectrum was supplied.
        """
        if not self.has_spectrum:
            return None
        if self.uses_template and not self.has_mag:
            return None
        if self._effective_spectrum is None:
            spec = self.spectrum
            if self.has_mag:
                curve = get_filter_curve(self.mag_filter)
                if curve is None:
                    self.notes.append(f'No transmission curve for {self.mag_filter}; spectrum kept at its absolute flux')
                else:
                    try:
                        spec = spec.scaled_to(self.mag_ab, curve)
                    except ValueError:
                        self.notes.append(f'Spectrum does not cover the {self.mag_filter} band; spectrum kept at its absolute flux')
            self._effective_spectrum = spec
        return self._effective_spectrum

    def ab_magnitude(self, filter_name: str) -> tuple[float, str]:
        """
        AB magnitude of the target in a 7DT filter and how it was obtained:
        'spectrum' (synthetic photometry), 'flat' (same AB magnitude in every band) or 'none'.
        """
        spec = self.effective_spectrum()
        if spec is not None:
            curve = get_filter_curve(filter_name)
            if curve is not None:
                mag = spec.synthetic_ab_mag(curve)
                if math.isfinite(mag):
                    return mag, 'spectrum'
            if self.has_mag:
                return self.mag_ab, 'flat (spectrum does not cover this band)'
            return float('nan'), 'none (spectrum does not cover this band)'
        if self.has_mag:
            return self.mag_ab, 'flat'
        return float('nan'), 'none'

    def saturation_note(self, filters=(), limit: float = SATURATION_MAG, extended_shift: float = 0.0) -> str | None:
        """
        A warning string when the target is brighter than `limit` in the input band or any listed filter.
        `limit` is the point-source limit (peak pixel of a star). For an extended source the surface
        brightness is compared with limit + extended_shift, where extended_shift = 2.5 log10(2 pi sigma^2)
        with sigma the seeing in arcsec / 2.355: the surface brightness that puts the same flux into a
        pixel as the peak pixel of a star of magnitude `limit`.
        """
        shift = extended_shift if self.extended else 0.0
        bright = []
        if self.has_mag and self.mag - shift < limit:
            unit = '/arcsec²' if self.extended else ''
            bright.append(f'{self.mag_filter} = {self.mag:.2f} ({self.mag_system}{unit})')
        in_filters = []
        if self.has_spectrum:
            for f in filters:
                mag, source = self.ab_magnitude(f)
                if source == 'spectrum' and mag - shift < limit:
                    in_filters.append((mag, f))
        if in_filters:
            in_filters.sort()
            shown = ', '.join(f'{f} = {m:.2f} AB' for m, f in in_filters[:3])
            more = f' and {len(in_filters) - 3} more filters' if len(in_filters) > 3 else ''
            bright.append(f'{len(in_filters)} of {len(list(filters))} 7DT filters (brightest {shown}{more})')
        if not bright:
            return None
        return f'Saturation risk: brighter than {limit:g} mag in ' + '; '.join(bright)

    def summary(self) -> dict:
        return {
            'name': self.label,
            'ra_deg': self.ra, 'dec_deg': self.dec,
            'ra_hms': self.ra_hms, 'dec_dms': self.dec_dms,
            'mag': self.mag, 'mag_filter': self.mag_filter, 'mag_ab': self.mag_ab,
            'spectrum': self.spectrum_label, 'redshift': self.redshift,
        }
