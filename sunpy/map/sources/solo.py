"""
Solar Orbiter Map subclass definitions.
"""

import warnings
import numpy as np

import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from astropy.visualization import AsinhStretch, ImageNormalize

from sunpy.coordinates import HeliocentricInertial
from sunpy.map import GenericMap
from sunpy.map.sources.source_type import source_stretch

__all__ = ['EUIMap', 'METISMap']


class EUIMap(GenericMap):
    """
    EUI Image Map

    The Extreme Ultraviolet Imager (EUI) is a remote sensing instrument onboard the
    Solar Orbiter (SolO) spacecraft. EUI has three telescopes that image the Sun in
    Lyman-alpha (1216 Å) and the EUV (174 Å and 304 Å). The three telescopes are the
    Full Sun Imager (FSI) and two High Resolution Imagers (HRI). The FSI images the
    whole Sun in both 174 Å and 304 Å. The EUV and Lyman-alpha HRI telescopes image a
    1000"-by-1000" patch in 174 Å and 1216 Å, respectively.

    References
    ----------
    * `Solar Orbiter Mission Page <https://sci.esa.int/web/solar-orbiter/>`__
    * `EUI Instrument Page <https://www.sidc.be/EUI/about/instrument>`__
    * Instrument Paper: :cite:t:`rochus_solar_2020`
    """

    def __init__(self, data, header, **kwargs):
        super().__init__(data, header, **kwargs)
        self._nickname = self.detector
        self.plot_settings['norm'] = ImageNormalize(
            stretch=source_stretch(self.meta, AsinhStretch(0.01)), clip=False)

    @property
    def _rotation_matrix_from_crota(self):
        return super()._rotation_matrix_from_crota(crota_key='CROTA')

    @property
    def processing_level(self):
        if self.meta.get('level'):
            # The level number is prepended by the letter L
            return int(self.meta.get('level')[1:])

    @property
    def waveunit(self):
        # EUI JP2000 files do not have the WAVEUNIT key in the metadata.
        # However, the FITS files do.
        # The EUI metadata spec says the WAVELNTH key is always expressed
        # in Angstroms so we assume this if the WAVEUNIT is missing.
        return super().waveunit or u.Angstrom

    @property
    def _supported_observer_coordinates(self):
        return [(('hcix_obs', 'hciy_obs', 'hciz_obs'),
                 {'x': self.meta.get('hcix_obs'),
                  'y': self.meta.get('hciy_obs'),
                  'z': self.meta.get('hciz_obs'),
                  'unit': u.m,
                  'representation_type': CartesianRepresentation,
                  'frame': HeliocentricInertial})] + super()._supported_observer_coordinates

    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """Determines if header corresponds to an EUI image"""
        is_solo = 'solar orbiter' in str(header.get('obsrvtry', '')).lower()
        is_eui = str(header.get('instrume', '')).startswith('EUI')
        return is_solo and is_eui



class METISMap(GenericMap):
    """
    Metis Image Map

    Metis is the multi-wavelength coronagraph for the Solar Orbiter mission that
    investigates the global corona in polarized visible light and in ultraviolet
    light. In the ultraviolet band, the coronagraph obtains monochromatic images
    in the Lyman-alpha line, at 121.6 nm, emitted by the few neutral-hydrogen
    atoms surviving in the hot corona.

    By simulating a solar eclipse, Metis observes the faint coronal light in an
    annular zone 1.6-2.9 deg wide, around the disk center. When Solar Orbiter
    is at its closest approach to the Sun, at the minimum perihelion of 0.28
    astronomical units, the annular zone is within 1.7 and 3.1 solar radii from
    disk center.

    Solar Orbiter was successfully launched on February 10th, 2020.

    References
    ----------
    * `Solar Orbiter Mission Page <https://sci.esa.int/web/solar-orbiter/>`_
    * `Metis Instrument Page <https://metis.oato.inaf.it/index.html>`_
    * Instrument Paper: Antonucci et al. 2020, A&A, 642, A10
    """

    def __init__(self, data, header, **kwargs):
        """
        Initialize the METISMap class with the provided data and header.
        Validate that the header contains the required parameters.
        """

        if 'RSUN_OBS' in header or 'SOLAR_R' in header or 'RADIUS' in header:
            pass
        else:
            header['RSUN_OBS'] = header['RSUN_ARC']

        # Call the superclass (GenericMap) to initialize the map
        super().__init__(data, header, **kwargs)

        self._nickname = f"{self.instrument}/{self.meta['filter']}"
        self._prodtype = self.get_prodtype()
        self._contr_cut = self.get_contr_cut()
        self.plot_settings['cmap'] = self._get_cmap_name()

    def get_prodtype(self):
        """
        Define the type of the Metis data product.

        Returns
        -------
        prodtype : `str`
            Name of the Metis data product.

        """

        btype_suff_dict = {
            'VL total brightness':             ('-TB', '-TB'),
            'VL polarized brightness':         ('-PB', '-PB'),
            'VL fixed-polarization intensity': ('-FP', '-Fix. Pol.'),
            'VL polarization angle':           ('-PA', '-Pol. Angle'),
            'Stokes I':                        ('-SI', '-Stokes I'),
            'Stokes Q':                        ('-SQ', '-Stokes Q'),
            'Stokes U':                        ('-SU', '-Stokes U'),
            'Pixel quality':                   ('-PQ', '-Pixel quality'),
            'Absolute error':                  ('-AE', '-Abs. err.'),
            'UV Lyman-alpha intensity':        ('', ''),
        }

        btype = self.meta['btype']
        prodtype = self.meta['filter']

        if btype in btype_suff_dict:
            suff, nickname_add = btype_suff_dict[btype]
            prodtype += suff
            self._nickname += nickname_add
        else:
            raise ValueError(
                f"Error. self.meta['btype']='{btype}' is not known."
            )

        return prodtype


    @property
    def prodtype(self):
        return self._prodtype


    @prodtype.setter
    def prodtype(self, value):
        raise AttributeError('Cannot manually set prodtype for METISMap')


    def get_contr_cut(self):
        """
        Define the contrast of the Metis data product.

        Returns
        -------
        contr_cut : `float` or `None`
            Contrast of the Metis data product.

        """
        if 'L2' in self.meta['level']:
            if self.prodtype == 'VL-TB' or self.prodtype == 'VL-SI':
                contr_cut = 0.05
            elif self.prodtype == 'VL-PB':
                contr_cut = 0.005
            elif self.prodtype == 'VL-FP':
                contr_cut = 0.01
            elif self.prodtype == 'VL-PA':
                contr_cut = 0.01
            elif self.prodtype == 'VL-SQ':
                contr_cut = 0.01
            elif self.prodtype == 'VL-SU':
                contr_cut = 0.01
            elif self.prodtype == 'UV':
                contr_cut = 0.05  # 0.03
            elif self.prodtype == 'VL-PQ' or self.prodtype == 'UV-PQ':
                contr_cut = None
            elif self.prodtype == 'VL-AE' or self.prodtype == 'UV-AE':
                contr_cut = 0.1
            else:
                contr_cut = None
        else:
            contr_cut = None
        return contr_cut


    @property
    def contr_cut(self):
        return self._contr_cut


    @contr_cut.setter
    def contr_cut(self, value):
        self._contr_cut = value


    @classmethod
    def is_datasource_for(cls, data, header, **kwargs):
        """
        Determine whether the data is a Metis product.

        Returns
        -------
        `bool`
            ``True`` if data corresponds to a Metis product, otherwise
            ``False``.

        """
        instrume = header.get('INSTRUME', '').strip().upper()
        obsrvtry = header.get('OBSRVTRY', '').strip().upper()
        return ('METIS' in instrume) and ('SOLAR ORBITER' in obsrvtry)


    def get_fov_rsun(self):
        """
        Return the Metis field of view in solar radii.

        Returns
        -------
        `tuple` : `(float, float, float)`
            Inner, outer radii of the field, determined by the internal
            occulter, field stop and detector size, respectively.

        """
        rsun_deg = self.rsun_obs.value / 3600.0  # in deg
        rmin_rsun = self.meta['inn_fov'] / rsun_deg  # in rsun
        rmax_rsun = self.meta['out_fov'] / rsun_deg  # in rsun
        board_deg = 2.9  # deg
        board_rsun = board_deg / rsun_deg  # in rsun
        return rmin_rsun, rmax_rsun, board_rsun


    def mask_occs(self, mask_val=np.nan):
        """
        Mask the data in regions obscured by internal and external occulters.

        Parameters
        ----------
        mask_val : `float`, optional
            The values of masked pixels (outside the field of view). Default is
            ``np.nan``.

        """
        if self.meta['cdelt1'] != self.meta['cdelt2']:
            warnings.warn('Warning: CDELT1 != CDELT2 for {fname}'.format(
                fname=self.meta['filename'])
            )
            print('\t>>> exiting mask_occs method.')
            return

        inn_fov = self.meta['inn_fov'] * 3600 / self.meta['cdelt1']  # in pix
        out_fov = self.meta['out_fov'] * 3600 / self.meta['cdelt2']  # in pix
        x = np.arange(0, self.meta['naxis1'], 1)
        y = np.arange(0, self.meta['naxis2'], 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        dist_suncen = np.sqrt(
            (xx-self.meta['sun_xcen'])**2 + (yy-self.meta['sun_ycen'])**2
        )
        dist_iocen = np.sqrt(
            (xx-self.meta['io_xcen'])**2 + (yy-self.meta['io_ycen'])**2
        )
        self.data[dist_iocen < inn_fov] = mask_val
        self.data[dist_suncen > out_fov] = mask_val


    def mask_bad_pix(self, qmat, mask_val=np.nan):
        """
        Mask bad-quality pixels in the Metis image.

        Parameters
        ----------
        qmat : `numpy.ndarray`
            Pixel quality matrix with the size of original image and the
            following values: ``1`` - linear range (good-quality), ``0`` -
            close to 0 counts/close to saturation (bad-quality) and ``np.nan`` -
            0 count/saturated pixels (bad-quality).
        mask_val : `float`, optional
            The values of masked pixels. Default is ``np.nan``.

        """

        if qmat.shape != self.data.shape:
            warnings.warn('Warning: Pixel quality matrix and the METISMap data have different size')
            print('\t>>> exiting mask_bad_pix method.')
            return
        qmat_mask = qmat == 1
        self.data[~qmat_mask] = mask_val


    def _get_cmap_name(self):
        """
        Override the default implementation to handle Metis color maps.

        Returns
        -------
        cmap_string : `str`
            Name of the Metis data product.

        """

        cmap_string = '{obsv}{instr}{prod}'.format(
            obsv=self.observatory, # self.observatory.replace(' ', '_'),
            instr=self.instrument,
            prod=self.prodtype
        )
        cmap_string = cmap_string.lower()
        return cmap_string


    def get_img_vlim(self):
        """
        Return the intensity limits of the Metis image.

        Returns
        -------
        `tuple` : `(float, float)`
            The minimum and maximum intensity values

        """

        vlim = AsymmetricPercentileInterval(
            self.contr_cut*100, (1-self.contr_cut)*100
        ).get_limits(self.data)
        return vlim


    def plot(self, **kwargs):
        """
        Override the default implementation to handle Metis color maps and
        contrast.

        """

        if self.contr_cut is None:
            clip_interval = None
        else:
            clip_interval = [self.contr_cut*100, (1-self.contr_cut)*100] * u.percent
        return super().plot(clip_interval=clip_interval, **kwargs)
