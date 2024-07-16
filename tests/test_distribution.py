# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
from scipy import stats

import gbigsmiles

EPSILON = 0.15
NSTAT = 2000


def test_flory_schulz():
    def mean(a):
        return 2 / a - 1

    def variance(a):
        return (2 - 2 * a) / a**2

    def skew(a):
        return (2 - a) / np.sqrt(2 - 2 * a)

    rng = np.random.default_rng()
    for a in [0.01, 0.05, 0.1, 0.3, 0.5]:
        flory_schulz = gbigsmiles.distribution.get_distribution(f"flory_schulz({a})")

        assert flory_schulz.prob_mw(flory_schulz.draw_mw(rng)) > 0

        data = np.asarray([flory_schulz.draw_mw(rng) for i in range(4 * NSTAT)])

        assert np.abs((np.mean(data) - mean(a)) / mean(a)) < EPSILON
        assert np.abs((np.var(data) - variance(a)) / variance(a)) < EPSILON
        assert np.abs((stats.skew(data) - skew(a)) / skew(a)) < EPSILON
        assert str(flory_schulz) == f"|flory_schulz({a})|"
        assert flory_schulz.generable


def test_gauss():
    def mean(mu, sigma):
        return mu

    def variance(mu, sigma):
        return sigma**2

    def skew(mu, sigma):
        return 0

    rng = np.random.default_rng()
    for mu, sigma in [(100.0, 10.0), (200.0, 100.0), (500.0, 1.0), (600.0, 0.0)]:
        gauss = gbigsmiles.distribution.get_distribution(f"gauss({mu}, {sigma})")

        example = gauss.draw_mw(rng)
        assert gauss.prob_mw(example) > 0

        data = np.asarray([gauss.draw_mw(rng) for i in range(NSTAT)])

        assert np.abs((np.mean(data) - mean(mu, sigma)) / mean(mu, sigma)) < EPSILON
        if sigma > 0:
            assert np.abs((np.var(data) - variance(mu, sigma)) / variance(mu, sigma)) < EPSILON

        assert str(gauss) == f"|gauss({mu}, {sigma})|"
        assert gauss.generable


def test_uniform():
    def mean(low, high):
        return 0.5 * (low + high)

    def variance(low, high):
        return 1 / 12.0 * (high - low) ** 2

    def skew(low, high):
        return 0

    rng = np.random.default_rng()
    for low, high in [(10, 100), (200, 1000), (50, 100), (0, 600)]:
        uniform = gbigsmiles.distribution.get_distribution(f"uniform({low}, {high})")

        assert uniform.prob_mw(uniform.draw_mw(rng)) > 0

        data = np.asarray([uniform.draw_mw(rng) for i in range(NSTAT)])

        assert np.abs((np.mean(data) - mean(low, high)) / mean(low, high)) < EPSILON
        assert np.abs((np.var(data) - variance(low, high)) / variance(low, high)) < EPSILON

        assert str(uniform) == f"|uniform({low}, {high})|"
        assert uniform.generable


def test_schulz_zimm():
    def mean(Mn, z):
        return Mn

    def variance(Mn, z):
        return Mn**2 / z

    rng = np.random.default_rng()
    for Mw in [11.3e3]:
        Mn = Mw / 1.5
        schulz_zimm = gbigsmiles.distribution.get_distribution(f"schulz_zimm({Mw}, {Mn})")
        z = schulz_zimm._z

        data = []
        for i in range(NSTAT):
            data.append(schulz_zimm.draw_mw(rng))
        data = np.asarray(data)

        # x = np.linspace(1e3, 40e3, 1000).astype(int)
        # plt.plot(x, schulz_zimm._distribution.pmf(x, z=schulz_zimm._z, Mn=schulz_zimm._Mn))
        # plt.show()

        assert np.abs((np.mean(data) - mean(Mn, z)) / mean(Mn, z)) < EPSILON
        assert np.abs((np.var(data) - variance(Mn, z)) / variance(Mn, z)) < EPSILON
        assert str(schulz_zimm) == f"|schulz_zimm({Mw}, {Mn})|"
        assert schulz_zimm.generable


def test_log_normal():
    def mean(M, D):
        return M

    rng = np.random.default_rng()
    for M, D in [
        (11.3e3, 1.1),
        (5.3e3, 1.5),
        (20.3e3, 2.0),
    ]:

        log_normal = gbigsmiles.distribution.get_distribution(f"log_normal({M}, {D})")

        data = []
        for i in range(NSTAT):
            d = log_normal.draw_mw(rng)
            data.append(d)
        data = np.asarray(data)

        # import matplotlib.pyplot as plt
        # x = np.linspace(1e3, 40e3, 1000)
        # plt.plot(x, log_normal._distribution.pdf(x, M=log_normal._M, D=log_normal._D))
        # plt.show()

        assert np.abs((np.mean(data) - mean(M, D))) / mean(M, D) < EPSILON
        assert str(log_normal) == f"|log_normal({M}, {D})|"
        assert log_normal.generable


def test_poisson():
    def mean(M):
        return M

    def variance(M):
        return M

    rng = np.random.default_rng()
    for M in [11.3e3, 5.3e3, 20.3e3]:

        poisson = gbigsmiles.distribution.get_distribution(f"poisson({M})")

        data = []
        for i in range(NSTAT):
            d = poisson.draw_mw(rng)
            data.append(d)
        data = np.asarray(data)

        assert np.abs((np.mean(data) - mean(M))) / mean(M) < EPSILON
        assert np.abs((np.var(data) - variance(M))) / variance(M) < EPSILON
        assert str(poisson) == f"|poisson({M})|"
        assert poisson.generable


if __name__ == "__main__":
    # test_flory_schulz()
    # test_gauss()
    # test_uniform()
    test_schulz_zimm()
