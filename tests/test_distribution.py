# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
from scipy import stats

import bigsmiles_gen

EPSILON = 0.15
NSTAT = 20000


def test_flory_schulz():
    def mean(a):
        return 2 / a - 1

    def variance(a):
        return (2 - 2 * a) / a**2

    def skew(a):
        return (2 - a) / np.sqrt(2 - 2 * a)

    rng = np.random.default_rng()
    for a in [0.01, 0.05, 0.1, 0.3, 0.5]:
        flory_schulz = bigsmiles_gen.distribution.get_distribution(f"flory_schulz({a})")

        data = np.asarray([flory_schulz.draw_mw(rng) for i in range(NSTAT)])

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
        gauss = bigsmiles_gen.distribution.get_distribution(f"gauss({mu}, {sigma})")

        data = np.asarray([gauss.draw_mw(rng) for i in range(NSTAT)])

        assert np.abs((np.mean(data) - mean(mu, sigma)) / mean(mu, sigma)) < EPSILON
        if sigma > 0:
            assert np.abs((np.var(data) - variance(mu, sigma)) / variance(mu, sigma)) < EPSILON

        assert str(gauss) == f"|gauss({mu}, {sigma})|"
        assert gauss.generable


if __name__ == "__main__":
    test_flory_schulz()
    test_gauss()
