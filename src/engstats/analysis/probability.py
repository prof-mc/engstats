"""
engstats.analysis.probability
==============================
Probability distribution helpers wrapping scipy.stats.
"""

import numpy as np
from scipy import stats


def normal_prob(x, mean: float = 0, std: float = 1, tail: str = "less") -> float:
    """
    Compute a normal distribution probability P(X ≤ x) or related tails.

    Parameters
    ----------
    x : float
        Value to evaluate.
    mean : float
        Distribution mean (μ). Default 0.
    std : float
        Standard deviation (σ). Default 1.
    tail : {'less', 'greater', 'two'}
        'less'    → P(X ≤ x)
        'greater' → P(X > x)
        'two'     → 2 · P(X ≤ −|x − μ| / σ)  (two-tailed area)

    Returns
    -------
    float

    Examples
    --------
    >>> es.normal_prob(1.96)          # ≈ 0.975
    >>> es.normal_prob(75, mean=70, std=5, tail='greater')
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    dist = stats.norm(loc=mean, scale=std)
    if tail == "less":
        return float(dist.cdf(x))
    elif tail == "greater":
        return float(dist.sf(x))
    elif tail == "two":
        z = abs((x - mean) / std)
        return float(2 * stats.norm.cdf(-z))
    else:
        raise ValueError("tail must be 'less', 'greater', or 'two'.")


def binomial_prob(k: int, n: int, p: float, cumulative: bool = False) -> float:
    """
    Compute a binomial probability P(X = k) or P(X ≤ k).

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    p : float
        Probability of success on each trial (0 ≤ p ≤ 1).
    cumulative : bool
        If True, return P(X ≤ k). Default False.

    Returns
    -------
    float

    Examples
    --------
    >>> es.binomial_prob(3, n=10, p=0.5)        # P(X = 3)
    >>> es.binomial_prob(3, n=10, p=0.5, cumulative=True)  # P(X ≤ 3)
    """
    dist = stats.binom(n=n, p=p)
    return float(dist.cdf(k) if cumulative else dist.pmf(k))


def poisson_prob(k: int, lam: float, cumulative: bool = False) -> float:
    """
    Compute a Poisson probability P(X = k) or P(X ≤ k).

    Parameters
    ----------
    k : int
        Number of events.
    lam : float
        Expected number of events (λ > 0).
    cumulative : bool
        If True, return P(X ≤ k).

    Returns
    -------
    float

    Examples
    --------
    >>> es.poisson_prob(2, lam=3.5)
    """
    if lam <= 0:
        raise ValueError("lam (λ) must be positive.")
    dist = stats.poisson(mu=lam)
    return float(dist.cdf(k) if cumulative else dist.pmf(k))
