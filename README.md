# Radiation Transport Solver

We implement a numerical solver for the **radiation transport equation** using **source iteration** and **Diffusion Synthetic Acceleration (DSA)** via the **Petrov-Galerkin finite element method** with SUPG stabilization.

We consider the steady-state **radiation transfer (linear Boltzmann) equation** for a single energy group.
For some $D\subset \mathbb{R}^3$, we want to find $\psi : D \times \mathbb{S}^2 \rightarrow \mathbb{R}$ such that

$$\Omega \cdot \nabla_\mathbf{x} \psi(\mathbf{x}, \Omega) + \sigma^t(\mathbf{x}) \psi(\mathbf{x}, \Omega) = \frac {\sigma^s(\mathbf{x})} {|\mathbb{S}^2|} \int_{\mathbb{S}^2} \psi(\mathbf{x}, \Omega') \ d \Omega' + q(\mathbf{x}), \quad \text{in } D \times \mathbb{S}^2$$

$$\psi(\mathbf{x}, \Omega) =  \alpha^\partial(\mathbf{x}),\quad \text{on } \{\mathbf{x} \in \partial D,\: \Omega \in \mathbb{S}^2 \ | \ n_\mathbf{x} \cdot \Omega < 0 \}.$$

Here $\sigma^t$ is the total cross section, $\sigma^s$ is the scattering cross section, and $\sigma^a = \sigma^t - \sigma^s$ is the absorption cross section, and $q$ is an external source. The function $\psi$ represents the amount of radiation at location $\mathbf{x}\in \mathbb{R}^3$ and in the direction $\Omega\in \mathbb{S}^2$.

---

This solver and its associated presentation were written by Rishi Dadlani, Ryan Divan, Sanandan Ojha, and Gavin Ratcliff. We are grateful to Professor Jean-Luc Guermond and Jordan Hoffart for their guidance in this project, as well as the organizers of the 2026 Texas A&M PDE Summer School.