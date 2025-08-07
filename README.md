# Convergence of hyperbolic approximations to higher-order PDEs for smooth solutions

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16748605.svg)](https://doi.org/10.5281/zenodo.16748605)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{giesselmann2025convergence,
  title={{C}onvergence of hyperbolic approximations to higher-order {PDEs} for
         smooth solutions},
  author={Giesselmann, Jan and Ranocha, Hendrik},
  year={2025},
  month={08},
  eprint={2508.04112},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{giesselmann2025convergenceRepro,
  title={Reproducibility repository for
         "{C}onvergence of hyperbolic approximations to higher-order {PDEs} for
         smooth solutions"},
  author={Giesselmann, Jan and Ranocha, Hendrik},
  year={2025},
  howpublished={\url{https://github.com/ranocha/2025_relative_energy_hyperbolization}},
  doi={10.5281/zenodo.16748605}
}
```

## Abstract

We prove the convergence of hyperbolic approximations for several classes of
higher-order PDEs, including the Benjamin-Bona-Mahony, Korteweg-de Vries, Gardner,
Kawahara, and Kuramoto-Sivashinsky equations, provided a smooth solution of the
limiting problem exists. We only require weak (entropy) solutions of the hyperbolic
approximations. Thereby, we provide a solid foundation for these approximations,
which have been used in the literature without rigorous convergence analysis.
We also present numerical results that support our theoretical findings.



## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.10.9.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- Jan Giesselmann (TU Darmstadt, Germany)
- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
