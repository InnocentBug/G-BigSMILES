# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
import pytest
from scipy import stats

import gbigsmiles

EPSILON = 0.15
NSTAT = 2000


def test_empty_serialize():
    vector = gbigsmiles.StochasticDistribution.get_empty_serial_vector()
    instance = gbigsmiles.StochasticDistribution.from_serial_vector(vector)
    assert instance is None


@pytest.mark.parametrize("a", [0.01, 0.05, 0.1, 0.3, 0.5])
def test_flory_schulz(a):
    def mean(a):
        return 2 / a - 1

    def variance(a):
        return (2 - 2 * a) / a**2

    def skew(a):
        return (2 - a) / np.sqrt(2 - 2 * a)

    value = int(1 / a + 1)
    if value % 2:
        flory_schulz = gbigsmiles.StochasticDistribution.make(f"flory_schulz({a})")
    else:
        flory_schulz = gbigsmiles.FlorySchulz.make(f"flory_schulz({a})")

    assert isinstance(flory_schulz, gbigsmiles.FlorySchulz)

    random_mw = flory_schulz.draw_mw()
    assert flory_schulz.prob_mw(random_mw) > 0

    data = np.asarray([flory_schulz.draw_mw() for i in range(4 * NSTAT)])

    assert np.abs((np.mean(data) - mean(a)) / mean(a)) < EPSILON
    assert np.abs((np.var(data) - variance(a)) / variance(a)) < EPSILON
    assert np.abs((stats.skew(data) - skew(a)) / skew(a)) < EPSILON
    assert str(flory_schulz) == f"|flory_schulz({a})|"
    assert flory_schulz.generable

    serial_vector = flory_schulz.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(flory_schulz)


@pytest.mark.parametrize(("mu", "sigma"), [(100.0, 10.0), (200.0, 100.0), (500.0, 1.0), (600.0, 0.0)])
def test_gauss(mu, sigma):
    def mean(mu, sigma):
        return mu

    def variance(mu, sigma):
        return sigma**2

    def skew(mu, sigma):
        return 0

    gauss = gbigsmiles.StochasticDistribution.make(f"gauss({mu}, {sigma})")
    assert isinstance(gauss, gbigsmiles.Gauss)

    example = gauss.draw_mw()
    assert gauss.prob_mw(example) > 0

    data = np.asarray([gauss.draw_mw() for i in range(NSTAT)])

    assert np.abs((np.mean(data) - mean(mu, sigma)) / mean(mu, sigma)) < EPSILON
    if sigma > 0:
        assert np.abs((np.var(data) - variance(mu, sigma)) / variance(mu, sigma)) < EPSILON

    assert str(gauss) == f"|gauss({mu}, {sigma})|"
    assert gauss.generable

    serial_vector = gauss.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(gauss)


@pytest.mark.parametrize(("low", "high"), [(10.0, 100.0), (200.0, 1000.0), (50.0, 100.0), (0.0, 600.0)])
def test_uniform(low, high):
    def mean(low, high):
        return 0.5 * (low + high)

    def variance(low, high):
        return 1 / 12.0 * (high - low) ** 2

    def skew(low, high):
        return 0

    uniform = gbigsmiles.StochasticDistribution.make(f"uniform({low}, {high})")
    assert isinstance(uniform, gbigsmiles.Uniform)

    assert uniform.prob_mw(uniform.draw_mw()) > 0

    data = np.asarray([uniform.draw_mw() for i in range(NSTAT)])

    assert np.abs((np.mean(data) - mean(low, high)) / mean(low, high)) < EPSILON
    assert np.abs((np.var(data) - variance(low, high)) / variance(low, high)) < EPSILON

    assert str(uniform) == f"|uniform({low}, {high})|"
    assert uniform.generable

    serial_vector = uniform.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(uniform)


@pytest.mark.parametrize(("Mw", "factor"), [(11.3e3, 1.5)])
def test_schulz_zimm(Mw, factor):
    def mean(Mn, z):
        return Mn

    def variance(Mn, z):
        return Mn**2 / z

    Mn = Mw / factor
    schulz_zimm = gbigsmiles.StochasticDistribution.make(f"schulz_zimm({Mw}, {Mn})")
    assert isinstance(schulz_zimm, gbigsmiles.SchulzZimm)
    z = schulz_zimm._z

    data = []
    for _i in range(NSTAT):
        data.append(schulz_zimm.draw_mw())
    data = np.asarray(data)

    # x = np.linspace(1e3, 40e3, 1000).astype(int)
    # plt.plot(x, schulz_zimm._distribution.pmf(x, z=schulz_zimm._z, Mn=schulz_zimm._Mn))
    # plt.show()

    assert np.abs((np.mean(data) - mean(Mn, z)) / mean(Mn, z)) < EPSILON
    assert np.abs((np.var(data) - variance(Mn, z)) / variance(Mn, z)) < EPSILON
    assert str(schulz_zimm) == f"|schulz_zimm({Mw}, {Mn})|"
    assert schulz_zimm.generable

    serial_vector = schulz_zimm.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(schulz_zimm)


@pytest.mark.parametrize(("M", "D"), [(11.3e3, 1.1), (5.3e3, 1.5), (20.3e3, 2.0)])
def test_log_normal(M, D):
    def mean(M, D):
        return M

    log_normal = gbigsmiles.StochasticDistribution.make(f"log_normal({M}, {D})")
    assert isinstance(log_normal, gbigsmiles.LogNormal)

    data = []
    for _i in range(NSTAT):
        d = log_normal.draw_mw()
        data.append(d)
    data = np.asarray(data)

    # import matplotlib.pyplot as plt
    # x = np.linspace(1e3, 40e3, 1000)
    # plt.plot(x, log_normal._distribution.pdf(x, M=log_normal._M, D=log_normal._D))
    # plt.show()

    assert np.abs((np.mean(data) - mean(M, D))) / mean(M, D) < EPSILON
    assert str(log_normal) == f"|log_normal({M}, {D})|"
    assert log_normal.generable

    serial_vector = log_normal.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(log_normal)


@pytest.mark.parametrize("M", [11.3e3, 5.3e3, 20.3e3])
def test_poisson(M):
    def mean(M):
        return M

    def variance(M):
        return M

    poisson = gbigsmiles.StochasticDistribution.make(f"poisson({M})")
    assert isinstance(poisson, gbigsmiles.Poisson)

    data = []
    for _i in range(NSTAT):
        d = poisson.draw_mw()
        data.append(d)
    data = np.asarray(data)

    assert np.abs((np.mean(data) - mean(M))) / mean(M) < EPSILON
    assert np.abs((np.var(data) - variance(M))) / variance(M) < EPSILON
    assert str(poisson) == f"|poisson({M})|"
    assert poisson.generable

    serial_vector = poisson.get_serial_vector()
    new_instance = gbigsmiles.StochasticDistribution.from_serial_vector(serial_vector)
    assert str(new_instance) == str(poisson)
