# Third-party notices

Everything in this repository that I did not write, what it is, where it came
from, and exactly what was done to it.

## Crystal structure data

All structural data comes from the **Crystallography Open Database** (COD),
<https://www.crystallography.net/cod/>. COD places its contents in the public
domain, and the statement travels inside each CIF file rather than living only
on the website:

> `# All data on this site have been placed in the public domain by the`
> `# contributors.`

| Asset | Source | Licence | Transformation applied |
|---|---|---|---|
| `tests/fixtures/cod-1000041-nacl.cif` | COD entry [1000041](https://www.crystallography.net/cod/1000041.html) | CC0 / public domain | Byte-for-byte copy. Renamed only. |
| `tests/fixtures/cod-9016675-fe3c.cif` | COD entry [9016675](https://www.crystallography.net/cod/9016675.html) | CC0 / public domain | Byte-for-byte copy. Renamed only. |
| `data/survey_cod.csv` | COD, 4,000 entries sampled uniformly | CC0 / public domain | Derived. No structural data is reproduced: each row holds an id, a formula, counts, a cell volume and the filter's verdict. Produced by `scripts/survey_cod.py`. |
| `data/catalog.csv` | COD, entries under 1500 Å³ | CC0 / public domain | Derived, as above, plus symmetry recomputed from the coordinates with pymatgen. Produced by `scripts/build_catalog.py`. |
| `data/cache/` | COD, per-structure REST endpoint | CC0 / public domain | Unmodified CIFs, fetched at runtime. Not committed. |

COD entries are experimental refinements deposited by their original authors.
Where an entry carries a DOI, the underlying publication belongs to its authors
and is not reproduced here.

## Model

**CHGNet** at <https://github.com/CederGroupHub/chgnet>, Modified BSD.
Copyright (c) 2023, The Regents of the University of California, through
Lawrence Berkeley National Laboratory, and the University of California,
Berkeley.

anneal installs CHGNet from PyPI (`chgnet==0.4.2`) and calls the pretrained
weights that package ships (`chgnet_0.3.0_e29f68s314m37.pth.tar`, 412,525
parameters). No weights are committed to this repository, none were retrained,
and no fine-tuning was performed. The model is used exactly as published.

## Libraries

| Package | Licence | Used for |
|---|---|---|
| pymatgen | MIT | CIF parsing, `Structure`, symmetry analysis |
| CHGNet | Modified BSD | interatomic potential |
| PyTorch | BSD-3-Clause | CHGNet's runtime |
| ASE | LGPL-2.1-or-later | the FIRE optimizer driving relaxation |
| NumPy | BSD-3-Clause | arrays |
| requests / urllib3 | Apache-2.0 / MIT | HTTP against COD |

ASE is LGPL. anneal imports it as a library and does not modify or redistribute
it, which the LGPL permits; anneal's own source stays under the licence in
[LICENSE](LICENSE).

## What the derived values are not

The energies, forces and relaxed geometries anneal produces are **predictions
from a machine-learned interatomic potential**. They are not measurements, not
DFT results, and not the deposited experimental data.

- A relaxed structure is not a refinement of the original and must not be cited
  as one. The deposited coordinates are the experimental record; anything anneal
  moves is a model's opinion about where atoms would go.
- Energies are useful for comparing an edit against the structure it started
  from. They are not publication-grade, not benchmark labels, and not
  interchangeable with DFT numbers.
- The filter's verdicts are anneal's own criteria, not a judgement on the
  quality of anyone's crystallographic work. An entry excluded here, whether for
  disorder, for a missing hydrogen or for cell size, is usually a perfectly good
  refinement that this particular model cannot be asked about.
- Symmetry labels in `data/catalog.csv` are recomputed from coordinates with a
  0.1 Å tolerance and may disagree with the space group the depositor recorded.
