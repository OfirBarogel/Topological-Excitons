# Topological-Excitons
Workflow:
1. create config and place in config folder
2. run on cluster, pass relative config file path to run.sh or config_paths.txt and run batch
3. download from cluster the datetime folder with 4 files: config, Exciton_Energys, F[0], Integrated_Fs
4. Analyze data with notebook
Additional:
If multiple configs write relative paths to config_paths.txt and run batch script.
If I want to look at other bands use Fslicer on Fs.npy file by giving the path relative to tobiasholder-storage/Data to Slicer.sh as well as the index of desired band.
Notebook can read folders.txt (which contains all analyzed folders, analyze data to the correct folders and add those folders to folders.txt.

# Scripts:
Topological Excitons.ipynb: contains all the functions, workspace for code development and documentation. This script contains all the graphing.
Topological_Excitons.py: contains functions from the notebook relating to the solving of the BSE. main runs calculations from config.
ExcitonWF.py: contains functions from notebook relating to the exciton wavefunctions (anything relating to dot product)

# Bloch Basis and Reciprocal-Lattice Convention

This code uses a fixed Bloch-basis convention for each lattice. The convention determines how Hamiltonians and their eigenvectors transform when momentum is shifted by a reciprocal lattice vector,

$
\mathbf{k}\rightarrow\mathbf{k}+\mathbf{G}.
$

## Units and lattice constant

Throughout the code, the lattice length scale is set to

$
\boxed{a=1}.
$

All real-space lattice vectors, bond vectors, and intracell orbital positions are therefore expressed in units of (a), while reciprocal-space vectors and momenta are expressed in units of (1/a).

For example, the honeycomb nearest-neighbor vectors are written as

$
\mathbf d_1=(-1,0),
$

$
\mathbf d_2=
\left(
\frac12,-\frac{\sqrt3}{2}
\right),
\qquad
\mathbf d_3=
\left(
\frac12,\frac{\sqrt3}{2}
\right),
$

rather than explicitly carrying a factor of (a).

Similarly, the reciprocal lattice vectors are written as

$
\mathbf b_1=
\left(
\frac{2\pi}{3},
\frac{2\pi}{\sqrt3}
\right),
\qquad
\mathbf b_2=
\left(
-\frac{2\pi}{3},
\frac{2\pi}{\sqrt3}
\right).
$

These expressions assume (a=1).

If the lattice constant were restored, real-space quantities would scale as

$
\mathbf r \rightarrow a\mathbf r,
$

while reciprocal-space quantities would scale as

$
\mathbf k\rightarrow \frac{\mathbf k}{a},
\qquad
\mathbf G\rightarrow\frac{\mathbf G}{a}.
$

Consequently, quantities such as

$
\mathbf k\cdot\mathbf r
\quad\text{and}\quad
\mathbf G\cdot\boldsymbol\tau_\alpha
$

remain dimensionless and are unchanged.

**Convention:** Hamiltonians added to the code should use (a=1) unless the lattice-constant convention is explicitly changed throughout the lattice and grid definitions. Avoid introducing an explicit `a` in an individual Hamiltonian while leaving `cfg["rec_vec"]` or other geometric quantities in (a=1) units, as this would make the real- and reciprocal-space conventions inconsistent.

## Reciprocal lattice

The primitive reciprocal lattice vectors are stored in

```python
cfg["rec_vec"] = [b1, b2]
```

and satisfy the usual reciprocal-lattice condition

$
e^{i\mathbf{G}\cdot\mathbf{R}}=1
$

for every Bravais lattice vector (\mathbf{R}).

Any reciprocal lattice vector can therefore be written as

$
\mathbf{G}=m\mathbf{b}_1+n\mathbf{b}_2,
\qquad m,n\in\mathbb{Z}.
$

## Intracell basis positions

Each orbital or sublattice is assigned an intracell position

$
\boldsymbol{\tau}_\alpha.
$

A lattice site belonging to orbital (\alpha) is therefore located at

$
\mathbf{r}*{\alpha}=\mathbf{R}+\boldsymbol{\tau}*\alpha.
$

Because (\mathbf{G}) is a reciprocal lattice vector,

$
e^{-i\mathbf{G}\cdot(\mathbf{R}+\boldsymbol{\tau}_\alpha)}=
e^{-i\mathbf{G}\cdot\boldsymbol{\tau}_\alpha}.
$

Thus all Bravais copies of the same orbital acquire the same phase under a reciprocal-lattice translation.

The intracell positions used by the code are stored as

```python
cfg["tau"]
```

with the ordering of `tau` matching the ordering of the Hamiltonian basis.

## Bloch-state convention

The cell-periodic part of a Bloch state obeys

$
u_{n,\mathbf{k}+\mathbf{G}}(\mathbf r) =
e^{-i\mathbf{G}\cdot\mathbf r}
u_{n,\mathbf{k}}(\mathbf r).
$

In the tight-binding orbital basis, the real-space operator (e^{-i\mathbf G\cdot\mathbf r}) is represented by the diagonal matrix

$
U_{\mathbf G}=
\operatorname{diag}
\left(
e^{-i\mathbf G\cdot\boldsymbol\tau_1},
e^{-i\mathbf G\cdot\boldsymbol\tau_2},
\ldots
\right).
$

This matrix is used when extending Bloch states across reciprocal-lattice boundaries.

For a two-orbital basis,

$
U_{\mathbf G}=
\begin{pmatrix}
e^{-i\mathbf G\cdot\boldsymbol\tau_A} & 0\
0 & e^{-i\mathbf G\cdot\boldsymbol\tau_B}
\end{pmatrix}.
$

If the origin is chosen such that

$
\boldsymbol\tau_A=0,
$

this simplifies to

$
U_{\mathbf G}=
\begin{pmatrix}
1 & 0\
0 & e^{-i\mathbf G\cdot\boldsymbol\tau_B}
\end{pmatrix}.
$

## Important: this is a basis convention, not a Hamiltonian property

The reciprocal-lattice phase is determined by the **lattice geometry, intracell orbital positions, and Bloch-basis convention**. It is not determined by hopping strengths or other Hamiltonian parameters.

Consequently, different models using the same orbital basis use the same reciprocal-lattice transformation.

For example, graphene and the Haldane model use the same honeycomb (A/B) basis. Therefore they use the same (U_{\mathbf G}), even though the Haldane model contains additional next-nearest-neighbor hopping and onsite terms.

When adding a new Hamiltonian to the code:

1. Determine the Bravais lattice and reciprocal lattice vectors.
2. Specify the intracell positions (\boldsymbol\tau_\alpha) in the same basis convention used to construct (H(\mathbf k)).
3. Store these positions in `cfg["tau"]`.
4. Ensure that the ordering of `cfg["tau"]` is identical to the orbital ordering of the Hamiltonian matrix.
5. Use these positions to construct the reciprocal-lattice phase when extending states outside the fundamental Brillouin zone.

A separate reciprocal-lattice phase rule should **not** be introduced for each Hamiltonian if multiple Hamiltonians use the same lattice and Bloch basis.

## Convention consistency

The definitions of

* the hopping vectors used in (H(\mathbf k)),
* the intracell positions `cfg["tau"]`,
* the ordering of the Hamiltonian basis,
* the reciprocal lattice vectors `cfg["rec_vec"]`, and
* the reciprocal-lattice sewing operation

must all use the same convention.

In particular, the current implementation assumes (a=1) in both the Hamiltonian geometry and `cfg["rec_vec"]`.

Changing the Bloch convention can change the matrix representation of (H(\mathbf k)) and the apparent transformation of its eigenvectors without changing the underlying physical system.

For this reason, new lattice implementations should explicitly define their basis positions and maintain the (a=1) convention rather than infer the reciprocal-lattice phase from the Hamiltonian parameters.
