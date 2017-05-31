#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Geographical interpolation (lon/lats).
"""
import tempfile

from numpy import arccos, sign, rad2deg, sqrt, arcsin, memmap, float64, radians, cos, sin, where, logical_and, less, \
    greater, diff, log2, float64, float32

from geotiepoints.interpolator import Interpolator

EARTH_RADIUS = 6370997.0


class GeoInterpolator(Interpolator):
    """
    Handles interpolation of geolocation from a grid of tie points.  It is
    preferable to have tie-points out till the edges if the tiepoint grid, but
    a method is provided to extrapolate linearly the tiepoints to the borders
    of the grid. The extrapolation is done automatically if it seems necessary.

    Uses numpy, scipy, and optionally pyresample

    The constructor takes in the tiepointed data as *data*, the
    *tiepoint_grid* and the desired *final_grid*. As optional arguments, one
    can provide *kx_* and *ky_* as interpolation orders (in x and y directions
    respectively), and the *chunksize* if the data has to be handled by pieces
    along the y axis (this affects how the extrapolator behaves). If
    *chunksize* is set, don't forget to adjust the interpolation orders
    accordingly: the interpolation is indeed done globaly (not chunkwise).
    """

    def __init__(self, lon_lat_data, *args, **kwargs):

        Interpolator.__init__(self, None, *args, **kwargs)
        self.lon_tiepoint = None
        self.lat_tiepoint = None
        try:
            # Maybe it's a pyresample object ?
            self.set_tiepoints(lon_lat_data.lons, lon_lat_data.lats)
            xyz = lon_lat_data.get_cartesian_coords()
            self.tie_data = [xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]]

        except AttributeError:
            self.set_tiepoints(lon_lat_data[0], lon_lat_data[1])
            lons_rad = radians(self.lon_tiepoint)
            lats_rad = radians(self.lat_tiepoint)
            x__ = EARTH_RADIUS * cos(lats_rad) * cos(lons_rad)
            y__ = EARTH_RADIUS * cos(lats_rad) * sin(lons_rad)
            z__ = EARTH_RADIUS * sin(lats_rad)
            self.tie_data = [x__, y__, z__]

        self.new_data = []
        for num in range(len(self.tie_data)):
            self.new_data.append([])

        # Check indices to be an array
        if isinstance(self.row_indices, (tuple, list)):
            self.row_indices = asarray(self.row_indices)
        if isinstance(self.col_indices, (tuple, list)):
            self.col_indices = asarray(self.col_indices)
        if isinstance(self.hrow_indices, (tuple, list)):
            self.hrow_indices = asarray(self.hrow_indices)
        if isinstance(self.hcol_indices, (tuple, list)):
            self.hcol_indices = asarray(self.hcol_indices)

        # convert indices to uint to save up to 20% of memory
        # get max size of row/cell
        max_size = max(self.col_indices.max(), self.row_indices.max())
        max_size_h = max(self.hcol_indices.max(), self.hrow_indices.max())
        # choose best dtype for indices
        data_type = self.choose_dtype(max_size)
        data_type_h = self.choose_dtype(max_size_h)

        self.row_indices = self.row_indices.astype(data_type)
        self.col_indices = self.col_indices.astype(data_type)
        self.hrow_indices = self.hrow_indices.astype(data_type_h)
        self.hcol_indices = self.hcol_indices.astype(data_type_h)

        # TODO: choose best data type depending on indices step
        # check size is enough for diff
        if self.hrow_indices.size > 2 and self.hcol_indices.size > 2:
            max_step = max(diff(self.hrow_indices).min(), diff(self.hcol_indices).min())
            max_size = log2(self.hcol_indices.max() * self.hrow_indices.max())
        else:
            max_step = max_size = 1
        # max image size 3000x3000 corresponds to log2(3000*3000) ~= 23.1
        if max_size > 23 and max_step <= 2:
            for num in range(len(self.tie_data)):
                if self.tie_data[num] is not None:
                    self.tie_data[num] = self.tie_data[num].astype(float64)
        else:
            for num in range(len(self.tie_data)):
                if self.tie_data[num] is not None:
                    self.tie_data[num] = self.tie_data[num].astype(float32)

    def set_tiepoints(self, lon, lat):
        """Defines the lon,lat tie points.
        """
        self.lon_tiepoint = lon
        self.lat_tiepoint = lat

    def interpolate(self):
        newx, newy, newz = Interpolator.interpolate(self)
        # Use memmap for lower memory usage
        shape = newx.shape
        # lat_f = tempfile.NamedTemporaryFile(dir='/home/mag/Documents/repos/solab/PySOL/notebooks/POSADA/')
        lat_f = tempfile.NamedTemporaryFile()
        lat_f.name = '/tmp/.lats.npz'
        lat = memmap(lat_f.name, dtype=newx.dtype, mode='w+', shape=shape)
        # lon_f = tempfile.NamedTemporaryFile(dir='/home/mag/Documents/repos/solab/PySOL/notebooks/POSADA/')
        lon_f = tempfile.NamedTemporaryFile()
        lon_f.name = '/tmp/.lons.npz'
        lon = memmap(lon_f.name, dtype=newx.dtype, mode='w+', shape=shape)

        lon[:] = get_lons_from_cartesian(newx, newy)[:]
        lat[:] = get_lats_from_cartesian(newx, newy, newz)[:]
        # Write any changes in the array to the file on disk
        lat.flush()
        lon.flush()
        return lon, lat


def get_lons_from_cartesian(x__, y__):
    """Get longitudes from cartesian coordinates.
    """
    return rad2deg(arccos(x__ / sqrt(x__ ** 2 + y__ ** 2))) * sign(y__)


def get_lats_from_cartesian(x__, y__, z__, thr=0.8):
    """Get latitudes from cartesian coordinates.
    """
    # if we are at low latitudes - small z, then get the
    # latitudes only from z. If we are at high latitudes (close to the poles)
    # then derive the latitude using x and y:

    lats = where(logical_and(less(z__, thr * EARTH_RADIUS),
                             greater(z__, -1. * thr * EARTH_RADIUS)),
                 90 - rad2deg(arccos(z__ / EARTH_RADIUS)),
                 sign(z__) *
                 (90 - rad2deg(arcsin(sqrt(x__ ** 2 + y__ ** 2)
                                      / EARTH_RADIUS))))
    return lats
