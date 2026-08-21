# Fast Solution Methods for the CJP Market-Making HJB with Asymmetric Fill-Decay Rates

**Model:** Cartea–Jaimungal–Penalva finite-inventory market-making HJB<br>
**Case of interest:** \(\kappa_+\neq\kappa_-\)<br>
**Primary objective:** reduce latency and total work for \(10^4\)–\(10^5\) parameter solves and for a live loop with \(2q_{\max}+1=13\) inventory states<br>
**Date:** 2026-08-21

---

## Executive summary

For the HJB exactly as written in the prompt, the equal-decay linearizing transform is

\[
\boxed{w_q=\exp(+\kappa h_q),}
\]

not \(\exp(-\kappa h_q)\). The positive sign follows immediately from

\[
e^{\kappa h_q}e^{-\kappa(h_q-h_{q-1})}=e^{\kappa h_{q-1}}.
\]

If the implementation currently uses the negative sign, first check whether its neighbor difference is the reverse of the one in the displayed HJB. This sign audit is important before comparing any solvers.

For \(\kappa_+\ne\kappa_-\), there is no invertible **componentwise** scalar, vector, or matrix-valued Cole–Hopf transform that turns the finite-state system into an affine-linear ODE at an interior state where both sides are active. A short proof is given below. I did not find a published theorem stated specifically for this CJP asymmetric-decay system; the result here is a direct derivation. The market-making literature instead leaves the general asymmetric Hamiltonians nonlinear and uses numerical or approximate methods.

The most useful constructive result is that the two book-side vector fields are **separately and exactly linearizable**, each with its own decay rate. This gives finite, explicit one-sided propagators:

\[
\bigl[\Phi_+^s(H)\bigr]_i
=
\frac1{\kappa_+}
\log\!\left(
\sum_{\ell=0}^{i}
\frac{(s\kappa_+c_+)^\ell}{\ell!}
 e^{\kappa_+H_{i-\ell}}
\right),
\]

\[
\bigl[\Phi_-^s(H)\bigr]_i
=
\frac1{\kappa_-}
\log\!\left(
\sum_{\ell=0}^{n-i}
\frac{(s\kappa_-c_-)^\ell}{\ell!}
 e^{\kappa_-H_{i+\ell}}
\right).
\]

A symmetric composition of these exact subflows and the exact drift flow is a second-order, Newton-free finite-horizon solver for arbitrary asymmetry. It uses only log-sum-exp or short positive convolutions, has no nonlinear convergence tolerance, and is especially attractive for a compiled batch or live implementation.

There is also an **exact arbitrary-asymmetry stationary/ergodic reduction**. On a numerically regular feasible branch, the long-horizon corrector is obtained from one scalar shooting equation with only \(O(N)\) recurrence work per residual evaluation. A deeper conditioning check is essential: if an optimal edge rate is nearly zero, the scalar recurrence subtracts nearly equal numbers and may require more precision than floating point provides. A robust log-edge Newton solve, initialized by homotopy from the symmetric Perron eigenvector, avoids that failure. A spectral-gap test tells whether a 150–600 second horizon is long enough for stationary quotes to be accurate.

For parameter sweeps, the recommended hierarchy is:

| Regime | Preferred method | Status |
|---|---|---|
| Ground-truth finite horizon | Reverse-time adaptive IVP in 12 gaps, analytic tridiagonal Jacobian | Numerical to requested tolerance |
| Arbitrary asymmetry, low latency | Exact-side Strang splitting | Second-order; Newton-free |
| Long horizon relative to inventory relaxation | Fast scalar stationary shooting with residual check; log-edge homotopy fallback | Stationary equations exact; finite-horizon correction approximate |
| Near symmetry and repeated base parameters | First/second-order \(\kappa\)-sensitivity around the symmetric matrix exponential | Semi-analytic perturbation |
| Moderate asymmetry around a useful reference \(\kappa_0\) | Exponential defect correction / ETD around a constant tridiagonal generator | Semi-analytic, nonperturbative iteration |
| Extreme latency constraint | Side-specific quadratic-Hamiltonian Riccati surrogate | Approximate; residual-certify |
| Repeated low-dimensional parameter domain | Offline Chebyshev/sparse-grid surrogate of quote gaps | Approximate; validate against ground truth |

The largest practical gains at 13 states are unlikely to come from replacing a dense \(13\times13\) solve by a tridiagonal solve alone. They come from removing nonlinear iterations, compiling/batching the very small kernels, reusing symmetric sensitivities or stationary roots, avoiding storage of the full path, and dispatching most calls to a certified approximation.

---

## 1. Model, indexing, and sign conventions

Let the admissible inventory states be

\[
q_0<q_1<\cdots<q_n,
\qquad n=12
\]

for the \(13\)-state case \(q\in\{-6,\ldots,6\}\). Reverse time:

\[
\tau=T-t,
\qquad
H_i(\tau)=h(T-\tau,q_i).
\]

Write the HJB as

\[
\boxed{
\dot H_i
=
 d_i
+
\mathbf 1_{i>0}c_+e^{-k_+(H_i-H_{i-1})}
+
\mathbf 1_{i<n}c_-e^{-k_-(H_i-H_{i+1})},
}
\tag{1}
\]

with

\[
H_i(0)=-\alpha q_i^2.
\tag{2}
\]

Here \(k_\pm=\kappa_\pm\), and all non-neighbor terms are placed in \(d_i\). For the equation exactly as supplied in the prompt,

\[
c_+=\frac{\lambda_+}{e k_+},
\qquad
c_-=\frac{\lambda_-}{e k_-}.
\tag{3}
\]

Depending on the adverse-selection convention, the optimized Hamiltonian may instead contain

\[
c_s(k_s)=\frac{\lambda_s}{e k_s}e^{-k_s\epsilon_s}.
\tag{4}
\]

All derivations below remain valid after replacing \(c_s\) by the coefficient used in the actual implementation. The difference matters numerically, so it should be audited once.

Under the usual unconstrained CJP feedback convention, the quote-depth terms are

\[
\delta_{+,i}^*
=
\epsilon_++\frac1{k_+}+H_i-H_{i-1},
\qquad i>0,
\tag{5}
\]

\[
\delta_{-,i}^*
=
\epsilon_-+\frac1{k_-}+H_i-H_{i+1},
\qquad i<n.
\tag{6}
\]

Only value differences are needed for quotes. A common additive shift of all \(H_i\) is irrelevant.

### 1.1 The symmetric sign audit

If \(k_+=k_-=k\), define

\[
w_i=e^{kH_i}.
\tag{7}
\]

Then

\[
\dot w_i
=k d_iw_i
+k c_+w_{i-1}
+k c_-w_{i+1},
\tag{8}
\]

with unavailable boundary terms omitted. Thus

\[
\dot w=Aw,
\qquad
w(0)_i=e^{-k\alpha q_i^2},
\tag{9}
\]

and

\[
H_i(\tau)=\frac1k\log\bigl[(e^{A\tau}w(0))_i\bigr].
\tag{10}
\]

For the neighbor differences in (1), the transform \(e^{-kH_i}\) gives terms proportional to \(w_i^2/w_{i\pm1}\) and is not linear. A negative-exponent transform is correct only under a correspondingly reversed difference/sign convention.

---

## 2. What can and cannot be linearized exactly

### 2.1 No componentwise Cole–Hopf transform for unequal decays

The following theorem formalizes the “one state variable, two incompatible exponential scales” intuition.

**Theorem.** Consider an interior state \(i\) for which both neighbor terms in (1) are present and \(c_+c_->0\). Let

\[
W_i=F_i(\tau,H_i)\in\mathbb R^m
\]

be a \(C^2\), componentwise transform that is injective in \(H_i\). Allow it to be time-dependent, inventory-state-dependent, and vector- or matrix-valued. Suppose all transformed variables satisfy an affine-linear system

\[
\dot W_i=b_i(\tau)+\sum_j B_{ij}(\tau)W_j,
\tag{11}
\]

whose coefficients are independent of \(H\). Then necessarily

\[
\boxed{k_+=k_-.}
\]

**Proof.** Take one scalar component \(F(\tau,H_i)\), and set

\[
x=H_i,
\quad y=H_{i-1},
\quad z=H_{i+1}.
\]

The contribution from the lower-neighbor term is

\[
c_+F_x(\tau,x)e^{-k_+(x-y)}
=
c_+\bigl[F_x(\tau,x)e^{-k_+x}\bigr]e^{k_+y}.
\tag{12}
\]

An affine-linear expression in componentwise transformed variables is a sum of functions of individual state variables. Its mixed derivative with respect to \(x\) and \(y\) is zero. Taking the mixed derivative of (12) therefore yields

\[
\frac{d}{dx}\left(F_x(\tau,x)e^{-k_+x}\right)=0,
\]

so

\[
F_x(\tau,x)=C_+(\tau)e^{k_+x}.
\tag{13}
\]

Applying the same argument to the upper-neighbor term gives

\[
F_x(\tau,x)=C_-(\tau)e^{k_-x}.
\tag{14}
\]

If \(k_+\ne k_-\), both identities can hold for all \(x\) only if \(F_x=0\), contradicting injectivity. ∎

This excludes all of the following exact constructions:

- a scalar statewise Cole–Hopf transform;
- different statewise transforms \(F_i(H_i)\);
- time-dependent statewise transforms;
- a finite vector such as \((e^{k_+H_i},e^{k_-H_i})\);
- a componentwise matrix exponential;
- a dense rather than tridiagonal affine-linear target system.

It does **not** exclude an arbitrary nonlocal transformation whose components mix the entire vector \((H_0,\ldots,H_n)\). I found no useful finite nonlocal linearization of that kind in the market-making literature. General local normal-form coordinates for one fixed analytic vector field are not a practical closed form: constructing them requires a parameter-specific infinite series, and their guaranteed convergence neighborhood need not contain the terminal condition.

### 2.2 A stronger but carefully scoped two-state obstruction

With only two states, the single value gap \(r=H_1-H_0\) satisfies

\[
\dot r=a_0+a_+e^{-k_+r}-a_-e^{k_-r}.
\tag{15}
\]

Define vector fields

\[
X_a=e^{ar}\frac{\partial}{\partial r}.
\]

Their Lie bracket is

\[
[X_a,X_b]=(b-a)X_{a+b}.
\tag{16}
\]

When \(k_+=k_-=k\), the span of \(X_{-k},X_0,X_k\) is three-dimensional and isomorphic to the Riccati/\(\mathfrak{sl}_2\) algebra. Indeed, \(x=e^{kr}\) turns (15) into a Riccati equation.

When \(k_+\ne k_-\), repeated brackets generate infinitely many distinct exponents. This rules out a finite-dimensional, coefficient-uniform Lie/projective lift for the whole family as \(a_0,a_+,a_-\) vary. It does not rule out an implicit quadrature or a transform designed for one single fully fixed scalar vector field.

The practical conclusion is narrower and useful: unequal decay rates destroy the finite Riccati/projective closure already in the smallest nontrivial inventory model.

### 2.3 What the literature says

The primary literature is consistent with this structure:

- Guéant’s exact finite-dimensional linear system assumes the same exponential decay parameter on both sides; see Eq. (3.13) in *Optimal Market Making*.
- Fodra and Labadie explicitly allow asymmetric intensity levels and decays, but note that the symmetric simplification is lost and two different nonlinear exponentials remain.
- Bergault, Evangelista, Guéant, and Vieira obtain closed-form proxies by replacing the side Hamiltonians with quadratic approximations and solving the resulting Riccati system.

I did not locate a published no-go theorem matching the componentwise theorem above. It should therefore be treated as a derivation in this note, not as a literature quotation.

---

## 3. Structural properties that should be exploited numerically

### 3.1 Translation invariance

Let \(\mathbf 1=(1,\ldots,1)^\top\). Equation (1) satisfies

\[
F(H+c\mathbf1)=F(H).
\tag{17}
\]

Consequences:

1. Quotes depend only on gaps.
2. A common level may be removed after every numerical step.
3. The common part of the drift is irrelevant to quotes. Decompose
   \[
   d=\bar d\,\mathbf1+\widetilde d.
   \]
   Solve only with \(\widetilde d\), and add \(\bar d\tau\) later only if the absolute value function is required.
4. The Jacobian has a zero row sum and a neutral common-level mode.

For quote-only code, subtracting \(H_{q=0}\), the mean, or the maximum after every accepted step keeps exponent arguments well scaled.

### 3.2 Cooperative tridiagonal Jacobian

Define the active side contributions

\[
a_i=c_+e^{-k_+(H_i-H_{i-1})},
\qquad i>0,
\tag{18}
\]

\[
b_i=c_-e^{-k_-(H_i-H_{i+1})},
\qquad i<n.
\tag{19}
\]

The nonzero Jacobian entries are

\[
J_{i,i-1}=k_+a_i,
\qquad
J_{i,i+1}=k_-b_i,
\tag{20}
\]

\[
J_{i,i}=-k_+a_i-k_-b_i,
\tag{21}
\]

with missing terms omitted at the boundaries. Thus \(J\) is tridiagonal, Metzler off the diagonal, and has zero row sums.

For backward Euler, Newton’s matrix is

\[
I-hJ.
\tag{22}
\]

Its diagonal exceeds the sum of the absolute off-diagonals by exactly \(1\), so it is a strictly diagonally dominant M-matrix. The analogous BDF2 matrix has the same property with \(3/2\) in place of \(1\). A Thomas solve is stable and requires \(O(N)\) work.

At \(N=13\), however, dense factorization is already very small. The major costs are normally function evaluations, nonlinear iterations, adaptive-solver overhead, allocation, and interpreted-language loops—not the asymptotic linear-algebra count.

### 3.3 Effective stiffness is controlled by fill rates

For the coefficient in (3),

\[
k_+a_i
=
\lambda_+e^{-1-k_+(H_i-H_{i-1})},
\tag{23}
\]

and similarly on the other side. With adverse selection inside the exponential, the corresponding expression contains \(\epsilon_s+H_i-H_j\). These quantities are optimized fill intensities.

Therefore raw values such as \(k=15{,}000\)–\(20{,}000\) per price unit do not by themselves imply stiffness. Stiffness is governed by the realized rates \(k_sa_{s,i}\) and their spread across inventory states.

### 3.4 Gap-coordinate system

Define

\[
r_i=H_i-H_{i-1},
\qquad i=1,\ldots,n.
\tag{24}
\]

Then

\[
\begin{aligned}
\dot r_i={}&d_i-d_{i-1}
+c_+e^{-k_+r_i}
-\mathbf1_{i>1}c_+e^{-k_+r_{i-1}}\\
&+\mathbf1_{i<n}c_-e^{k_-r_{i+1}}
-c_-e^{k_-r_i}.
\end{aligned}
\tag{25}
\]

This removes the neutral level and reduces the 13-state system to 12 variables. The state-count saving is small, but the conditioning and quote-oriented error control are cleaner. It is a good formulation for the ground-truth adaptive IVP.

---

## 4. Exact one-sided flows: the main finite-horizon speedup

Split (1) into

\[
F=D+P+M,
\tag{26}
\]

where

\[
(DH)_i=d_i,
\tag{27}
\]

\[
(PH)_i=
\begin{cases}
 c_+e^{-k_+(H_i-H_{i-1})},&i>0,\\
 0,&i=0,
\end{cases}
\tag{28}
\]

and

\[
(MH)_i=
\begin{cases}
 c_-e^{-k_-(H_i-H_{i+1})},&i<n,\\
 0,&i=n.
\end{cases}
\tag{29}
\]

Each subflow is exactly available.

### 4.1 Drift flow

\[
\boxed{\Phi_D^s(H)=H+s d.}
\tag{30}
\]

### 4.2 Exact lower-neighbor/“plus” flow

For \(\dot H=P(H)\), set

\[
u_i=e^{k_+H_i}.
\]

Then

\[
\dot u_i=k_+c_+u_{i-1},
\qquad i>0,
\qquad
\dot u_0=0.
\tag{31}
\]

Let \(L\) be the lower-shift matrix \((Lu)_i=u_{i-1}\), with \((Lu)_0=0\). Since \(L^{n+1}=0\),

\[
u(s)=e^{s k_+c_+L}u(0)
=
\sum_{\ell=0}^{n}
\frac{(s k_+c_+)^\ell}{\ell!}L^\ell u(0).
\]

Therefore

\[
\boxed{
\bigl[\Phi_+^s(H)\bigr]_i
=
\frac1{k_+}
\log\!\left[
\sum_{\ell=0}^{i}
\frac{(s k_+c_+)^\ell}{\ell!}
 e^{k_+H_{i-\ell}}
\right].
}
\tag{32}
\]

### 4.3 Exact upper-neighbor/“minus” flow

Likewise,

\[
\boxed{
\bigl[\Phi_-^s(H)\bigr]_i
=
\frac1{k_-}
\log\!\left[
\sum_{\ell=0}^{n-i}
\frac{(s k_-c_-)^\ell}{\ell!}
 e^{k_-H_{i+\ell}}
\right].
}
\tag{33}
\]

For the HJB coefficient in (3),

\[
k_+c_+=\frac{\lambda_+}{e},
\qquad
k_-c_-=\frac{\lambda_-}{e}.
\tag{34}
\]

Thus the polynomial weights in the exact side flows do not grow with the large raw \(k_s\). The decay parameters enter through the exponential/log coordinates.

### 4.4 Stable log-sum-exp evaluation

For the plus side, evaluate

\[
\bigl[\Phi_+^s(H)\bigr]_i
=
\frac1{k_+}
\operatorname{LSE}_{0\le\ell\le i}
\left(
 k_+H_{i-\ell}
+\ell\log(sk_+c_+)
-\log\Gamma(\ell+1)
\right).
\tag{35}
\]

The minus side is analogous. Internally subtract a common gauge \(g\) from \(H\), compute with \(k_s(H-g)\), and add \(g\) back.

For a faster batch kernel when the scaled range is moderate:

1. choose \(g=\max_iH_i\);
2. set \(x_i=e^{k_s(H_i-g)}\in(0,1]\);
3. form weights recursively,
   \[
   w_0=1,
   \qquad
   w_\ell=w_{\ell-1}\frac{s k_sc_s}{\ell};
   \]
4. apply a short triangular Toeplitz convolution;
5. return \(g+k_s^{-1}\log y_i\).

At 13 states, each side requires only 91 multiply-add contributions. The kernel can be fully unrolled, vectorized over parameter batches, or expressed as a prebuilt triangular matrix multiplication.

### 4.5 State-dependent intensity levels

If \(c_{+,i}\) varies by inventory state, the transformed plus flow is still linear:

\[
\dot u_i=k_+c_{+,i}u_{i-1}.
\]

The simple factorial convolution is replaced by the exponential of a lower-bidiagonal matrix. The same holds on the minus side. Thus exact one-sided linearizability survives state-dependent intensity levels, although the closed finite sum has nonuniform path products.

### 4.6 Monotonicity and nonexpansiveness

Each row of (32) or (33) is a log-sum-exp of shifted state values. Its gradient consists of nonnegative softmax weights that sum to one. Hence

\[
\|\Phi_\pm^s(H)-\Phi_\pm^s(G)\|_\infty
\le
\|H-G\|_\infty.
\tag{36}
\]

The subflows are also order-preserving and translation-equivariant. These properties make positive-time splitting robust even at large \(k_s\).

---

## 5. A Newton-free arbitrary-asymmetry finite-horizon solver

### 5.1 Symmetric second-order composition

One convenient palindromic step is the sequential application

1. \(\Phi_+^{h/2}\);
2. \(\Phi_D^{h/2}\);
3. \(\Phi_-^{h}\);
4. \(\Phi_D^{h/2}\);
5. \(\Phi_+^{h/2}\).

Denote the resulting map by \(S_h\). It is a Strang-type second-order composition:

\[
S_hH=\Phi_hH+O(h^3),
\tag{37}
\]

where \(\Phi_h\) is the exact full nonlinear flow over one step. Over a fixed horizon, the global error is \(O(h^2)\).

The exact flow also has the nonlinear product representation

\[
\boxed{
H(\tau)=\lim_{m\to\infty}S_{\tau/m}^{\,m}H(0).
}
\tag{38}
\]

A finite composition is not an exact closed form, but every subproblem is solved exactly and no Newton or fixed-point iteration occurs.

The reversed palindrome, with the minus side outside and the plus side in the center, is equally valid. Their error constants differ. For a fixed calibration, benchmark both once and retain the one with the smaller step-doubling defect.

### 5.2 Error estimate in quote units

Compute the endpoint once using step \(h\) and once using two half steps. Let their adjacent-gap vectors be \(r_h\) and \(r_{h/2}\). Since the method is second order, the fine-grid gap error is estimated by

\[
\boxed{
E_{\rm gap}
=
\frac13\|r_{h/2}-r_h\|_\infty.
}
\tag{39}
\]

This is directly in quote-depth units when \(1/k_s\) and \(\epsilon_s\) are kept exact.

A formal fourth-order Richardson gap estimate is

\[
\boxed{
r^{[4]}=\frac{4r_{h/2}-r_h}{3}.}
\tag{40}
\]

Validate it with another refinement before production use. Extrapolation loses the monotonicity property of the underlying positive-flow composition.

Real-coefficient splitting formulas of order higher than two generally require negative substeps. Negative time in (32)–(33) can destroy positivity of the transformed vector and make the logarithm undefined. Step doubling or extrapolation is safer here than a standard fourth-order Yoshida composition.

### 5.3 Fixed-step versus adaptive use

For a live loop with a narrow calibrated domain:

1. determine a fixed \(h\) offline from worst-case step-doubling tests;
2. use exactly \(M=T/h\) steps;
3. compile and unroll the side-flow kernels;
4. return only the required quote gaps.

Adjacent outer half-flows merge by the semigroup property,

\[
\Phi_+^{h/2}\circ\Phi_+^{h/2}=\Phi_+^h.
\]

Thus \(M\) palindromic steps require only \(M+1\) evaluations of the outer side flow rather than \(2M\). The same optimization applies when the minus side is chosen as the outer flow. This removes adaptive branching and makes latency predictable.

For a wide parameter sweep, adapt \(h\) using (39). A local controller should use exponent \(1/3\), because the local defect of a second-order method is \(O(h^3)\).

### 5.4 Optional two-flow drift embedding

Split the drift between the two sides:

\[
A_\theta=\theta D+P,
\qquad
B_\theta=(1-\theta)D+M,
\qquad 0\le\theta\le1.
\tag{41}
\]

Let \(Q=\operatorname{diag}(d_0,\ldots,d_n)\). Then

\[
\boxed{
\Phi_{A_\theta}^s(H)
=
\frac1{k_+}
\log\!\left[
 e^{s(k_+\theta Q+k_+c_+L)}e^{k_+H}
\right],
}
\tag{42}
\]

\[
\boxed{
\Phi_{B_\theta}^s(H)
=
\frac1{k_-}
\log\!\left[
 e^{s(k_-(1-\theta)Q+k_-c_-R)}e^{k_-H}
\right].
}
\tag{43}
\]

The matrices are bidiagonal Metzler matrices. An ABA or BAB Strang composition uses only these two exact flows. This variant is attractive when parameters and \(h\) are fixed and the two matrix exponentials can be cached. For parameters that change every call, the explicit convolution formulas are usually simpler.

---

## 6. Exact stationary/ergodic reduction for arbitrary asymmetry

For long reverse time, seek

\[
H_i(\tau)=\rho\tau+v_i+o(1).
\tag{44}
\]

Then

\[
\rho
=
 d_i
+
\mathbf1_{i>0}c_+e^{-k_+(v_i-v_{i-1})}
+
\mathbf1_{i<n}c_-e^{-k_-(v_i-v_{i+1})}.
\tag{45}
\]

Define edge gaps

\[
r_i=v_i-v_{i-1},
\qquad i=1,\ldots,n,
\tag{46}
\]

and edge contributions

\[
a_i=c_+e^{-k_+r_i},
\qquad i=1,\ldots,n,
\tag{47}
\]

\[
b_{i-1}=c_-e^{k_-r_i},
\qquad i=1,\ldots,n.
\tag{48}
\]

Set \(a_0=0\) and \(b_n=0\). The node equations are

\[
\rho=d_i+a_i+b_i.
\tag{49}
\]

### 6.1 Scalar shooting recurrence

Let

\[
\gamma=\frac{k_+}{k_-}.
\tag{50}
\]

For a trial \(\rho\), start at the lower boundary:

\[
b_0=\rho-d_0.
\tag{51}
\]

Then for \(i=1,\ldots,n\), compute

\[
\boxed{
a_i
=c_+\left(\frac{b_{i-1}}{c_-}\right)^{-\gamma}.
}
\tag{52}
\]

For \(i<n\), continue with

\[
\boxed{b_i=\rho-d_i-a_i.}
\tag{53}
\]

The final scalar residual is

\[
\boxed{F(\rho)=\rho-d_n-a_n(\rho).}
\tag{54}
\]

All intermediate \(b_i\) must be positive. This recurrence is algebraically exact.

### 6.2 Branch monotonicity and the conditioning caveat

On any connected interval of \(\rho\) on which every intermediate \(b_i\) remains positive, differentiate the recurrence:

\[
b_0'=1,
\tag{55}
\]

\[
a_i'
=-\gamma a_i\frac{b_{i-1}'}{b_{i-1}},
\tag{56}
\]

\[
b_i'
=1+\gamma a_i\frac{b_{i-1}'}{b_{i-1}}>0.
\tag{57}
\]

Therefore

\[
F'(\rho)
=1+\gamma a_n\frac{b_{n-1}'}{b_{n-1}}>0.
\tag{58}
\]

Thus **each connected feasible branch contains at most one root**. This is weaker than a global monotonic-root claim. Feasible branches can be separated by singular points at which an intermediate \(b_i\) vanishes. More importantly, when the corrector has an almost-disabled edge, (53) forms a tiny positive number by subtracting two nearly equal \(O(1)\) quantities. The scalar shooting variable \(\rho\) may then need far more than double precision to recover that edge rate, even though the corresponding log-gap is representable.

This is not merely theoretical. Extreme intensity/decay/penalty combinations can produce edge rates below machine-relative precision and stationary gaps of tens of price units in the scaled coordinate. A naive bisection that labels every infeasible point as “below the root” can converge to the edge of the wrong feasible branch.

The scalar recurrence remains an excellent fast path when all reconstructed edge rates are comfortably resolved. It must always be followed by a full node-residual check.

### 6.3 A priori bounds for a finite corrector

If a finite stationary corrector exists, then at a state where \(v_i\) is maximal both exponential factors are at most one, and at a state where \(v_i\) is minimal both are at least one. Therefore

\[
\boxed{
\min_i\left[d_i+\mathbf1_{i>0}c_++\mathbf1_{i<n}c_-\right]
\le \rho \le
\max_i\left[d_i+\mathbf1_{i>0}c_++\mathbf1_{i<n}c_-\right].
}
\tag{59}
\]

These are useful outer bounds, but they do not by themselves identify the correct feasible shooting branch. In a parameter sweep, use continuation from the preceding \(\rho\) and bracket locally inside its feasible branch. For a standalone solve or a failed residual check, use the log-edge formulation below.

### 6.4 Robust log-edge stationary system

Define dimensionless edge variables

\[
y_i=k_-r_i
=\log\frac{b_{i-1}}{c_-},
\qquad i=1,\ldots,n.
\]

Then

\[
b_{i-1}=c_-e^{y_i},
\qquad
a_i=c_+e^{-\gamma y_i}.
\]

The stationary problem becomes \(n+1\) equations for \((y_1,\ldots,y_n,\rho)\):

\[
d_0+c_-e^{y_1}-\rho=0,
\]

\[
d_i+c_+e^{-\gamma y_i}+c_-e^{y_{i+1}}-\rho=0,
\qquad i=1,\ldots,n-1,
\]

\[
d_n+c_+e^{-\gamma y_n}-\rho=0.
\]

Its Jacobian is bidiagonal in the edge variables plus one \(-1\) column for \(\rho\):

\[
\frac{\partial G_0}{\partial y_1}=c_-e^{y_1},
\]

\[
\frac{\partial G_i}{\partial y_i}
=-\gamma c_+e^{-\gamma y_i},
\qquad
\frac{\partial G_i}{\partial y_{i+1}}
=c_-e^{y_{i+1}},
\]

\[
\frac{\partial G_n}{\partial y_n}
=-\gamma c_+e^{-\gamma y_n},
\qquad
\frac{\partial G_i}{\partial\rho}=-1.
\]

At 13 states, a dense Newton solve is already negligible; a bordered-bidiagonal elimination is available if desired. The log variables avoid subtracting to obtain tiny edge rates.

A robust initialization is a homotopy from the symmetric-decay Perron solution:

1. choose \(\bar k>0\), for example \((k_++k_-)/2\);
2. keep \(d,c_+,c_-\) fixed and set \(k_+(0)=k_-(0)=\bar k\);
3. obtain \(\rho(0)\) and \(v(0)\) from the principal eigenpair of the symmetric Metzler matrix;
4. continue \(k_+(\theta),k_-(\theta)\) to their target values in adaptive increments;
5. at each increment, initialize \(y_i=k_-(\theta)r_i\) from the preceding physical gaps and apply Newton;
6. halve the continuation increment on failure.

This continuation is normally overkill for \(k_+/k_-\) near one, but it is the appropriate fallback when scalar shooting fails its residual check.

### 6.5 Recovering and verifying the gaps

Once a valid solution is found,

\[
\boxed{
r_i
=
\frac{y_i}{k_-}
=
\frac1{k_-}\log\frac{b_{i-1}}{c_-}
=
-\frac1{k_+}\log\frac{a_i}{c_+}.
}
\tag{60}
\]

Reconstruct every node value in (45) and require

\[
\max_i\left|
 d_i+a_i+b_i-\rho
\right|
\]

below a scale-aware tolerance. Do not accept a scalar shooting answer solely because its one terminal residual is small.

No gauge value for \(v\) is needed if only quotes are required.

### 6.6 Spectral-gap test for finite horizons

At the stationary corrector, the Jacobian has entries

\[
J_{i,i-1}=k_+a_i,
\quad
J_{i,i+1}=k_-b_i,
\quad
J_{i,i}=-(k_+a_i+k_-b_i).
\tag{61}
\]

It is a birth–death generator, diagonally similar to a symmetric tridiagonal matrix. It has one zero eigenvalue for common shifts and all remaining eigenvalues are real and negative.

Let \(g>0\) be the magnitude of the least-negative nonzero eigenvalue. Then

\[
gT\gg1
\tag{62}
\]

is the appropriate diagnostic for whether finite-horizon quote gaps are close to stationary. Raw horizon length alone is not informative.

A cheap terminal-layer approximation is

\[
H(\tau)
\approx
\rho\tau+v+e^{J\tau}\bigl(H(0)-v\bigr).
\tag{63}
\]

This correction is not exact because the HJB is nonlinear. It should be checked with the residual certificate in Section 10.

---

## 7. Perturbation around the symmetric matrix-exponential solution

Let

\[
k_+=\bar k+\eta,
\qquad
k_-=\bar k-\eta,
\tag{64}
\]

or more generally \(k_s=\bar k+\eta\sigma_s\), with \(\sigma_+=1\), \(\sigma_-=-1\). Use the fixed transform

\[
u_i=e^{\bar kH_i}.
\tag{65}
\]

The exact asymmetric equation is

\[
\dot u_i
=
\bar k d_i(\eta)u_i
+
\bar k\sum_s
c_s(k_s)
 u_i^{1-\rho_s}u_{n_s(i)}^{\rho_s},
\qquad
\rho_s=\frac{k_s}{\bar k}.
\tag{66}
\]

At \(\eta=0\),

\[
\dot u^{(0)}=Au^{(0)},
\qquad
u^{(0)}(0)_i=e^{-\bar k\alpha q_i^2},
\tag{67}
\]

where

\[
A_{ii}=\bar k d_i,
\quad
A_{i,i-1}=\bar k c_+(\bar k),
\quad
A_{i,i+1}=\bar k c_-(\bar k).
\tag{68}
\]

Thus

\[
u^{(0)}(\tau)=e^{A\tau}u^{(0)}(0).
\tag{69}
\]

### 7.1 First-order correction

Write

\[
u=u^{(0)}+\eta u^{(1)}+O(\eta^2).
\tag{70}
\]

Let

\[
\ell_s(k)=\log c_s(k).
\]

Then

\[
\boxed{
\dot u^{(1)}=Au^{(1)}+f^{(1)}(\tau),
\qquad
u^{(1)}(0)=0,
}
\tag{71}
\]

where

\[
\boxed{
\begin{aligned}
f_i^{(1)}={}&
\bar k d_i^{(1)}u_i^{(0)}\\
&+\sum_s
\sigma_s c_s(\bar k)u_{n_s(i)}^{(0)}
\left[
\bar k\ell_s'(\bar k)
+
\log\frac{u_{n_s(i)}^{(0)}}{u_i^{(0)}}
\right].
\end{aligned}
}
\tag{72}
\]

Hence

\[
\boxed{
u^{(1)}(\tau)
=
\int_0^\tau e^{A(\tau-r)}f^{(1)}(r)\,dr.
}
\tag{73}
\]

The value-function sensitivity is

\[
\boxed{
H_i^{(1)}(\tau)
=
\frac{u_i^{(1)}(\tau)}{\bar k u_i^{(0)}(\tau)}.
}
\tag{74}
\]

For the coefficient \(c_s(k)=\lambda_s/(ek)\),

\[
\bar k\ell_s'(\bar k)=-1.
\tag{75}
\]

For \(c_s(k)=\lambda_se^{-k\epsilon_s}/(ek)\),

\[
\bar k\ell_s'(\bar k)=-1-\bar k\epsilon_s.
\tag{76}
\]

Every higher order solves a forced linear ODE with the same homogeneous matrix \(A\). Once \(A\) is diagonalized or Schur-factorized, multiple orders and many values of \(\eta\) are inexpensive.

### 7.2 Quote correction

Keep \(1/k_s\) exact and expand only the value gap:

\[
\widetilde\delta_{s,i}^*
=
\epsilon_s+\frac1{k_s}
+
\Delta_sH_i^{(0)}
+
\eta\Delta_sH_i^{(1)}.
\tag{77}
\]

This avoids an unnecessary Taylor error in a known elementary term.

### 7.3 When is the expansion small?

The relative asymmetry

\[
\frac{|\eta|}{\bar k}
\]

is not sufficient. The exponential mismatch is controlled by

\[
|\eta|\,|H_i-H_j+\epsilon_s|.
\tag{78}
\]

For \(k_+=15{,}000\) and \(k_-=20{,}000\),

\[
\frac{|\eta|}{\bar k}
=
\frac{2{,}500}{17{,}500}
\approx0.143.
\tag{79}
\]

That is not automatically small. Use the residual certificate rather than a universal percentage cutoff.

---

## 8. Exponential defect correction around a reference decay

A useful nonperturbative reformulation is available even when the asymmetry is too large for a low-order Taylor series.

Choose a reference \(k_0>0\) and set

\[
u_i=e^{k_0H_i}.
\tag{80}
\]

Define a constant tridiagonal matrix \(A\) by

\[
A_{ii}=k_0d_i,
\quad
A_{i,i-1}=k_0c_+,
\quad
A_{i,i+1}=k_0c_-.
\tag{81}
\]

The exact equation is

\[
\boxed{\dot u=Au+R(u),}
\tag{82}
\]

with

\[
\boxed{
R_i(u)
=
k_0\sum_s c_s
\left[
 u_i^{1-k_s/k_0}u_{n_s(i)}^{k_s/k_0}
-u_{n_s(i)}
\right].
}
\tag{83}
\]

The variation-of-constants identity is

\[
u(\tau)
=e^{A\tau}u(0)
+
\int_0^\tau e^{A(\tau-r)}R(u(r))\,dr.
\tag{84}
\]

### 8.1 One-shot exponential defect correction

Use

\[
u^{[0]}(r)=e^{Ar}u(0),
\tag{85}
\]

then compute

\[
\boxed{
u^{[1]}(\tau)
=e^{A\tau}u(0)
+
\int_0^\tau e^{A(\tau-r)}R(u^{[0]}(r))\,dr.
}
\tag{86}
\]

A small fixed Gauss–Legendre quadrature evaluates the integral. The same eigendecomposition or Schur form of \(A\) is reused at every node. Further Picard corrections are possible.

This method is:

- exact in the symmetric limit;
- nonperturbative in the numerical value of \(k_+-k_-\);
- potentially very fast when one reference generator is reused;
- not guaranteed accurate without a residual check.

A reasonable default is \(k_0=(k_++k_-)/2\). If a representative solution is available, a better weighted choice minimizes the observed mismatch \((k_0-k_s)\Delta_sH\) over active edges.

### 8.2 ETD time stepping

Equation (82) also supports ETD2 or ETDRK4. The symmetric/reference part is propagated exactly; only the asymmetric defect is sampled. This is particularly attractive when the defect is moderate but a single global Picard correction is not accurate enough.

For 13 states, compare ETD against exact-side splitting rather than assuming it is faster. The splitting method has a simpler positivity structure and avoids fractional powers of very small transformed components.

---

## 9. Side-specific quadratic-Hamiltonian/Riccati approximation

Let the side Hamiltonians be

\[
G_s(p)=c_se^{-k_sp}.
\tag{87}
\]

Approximate each side separately by

\[
\widehat G_s(p)
=a_{0s}+a_{1s}p+\frac12a_{2s}p^2.
\tag{88}
\]

At expansion center \(p_0=0\),

\[
a_{0s}=c_s,
\qquad
a_{1s}=-k_sc_s,
\qquad
a_{2s}=k_s^2c_s.
\tag{89}
\]

For a nonzero center \(p_{0,s}\), use

\[
a_{2s}=G_s''(p_{0,s}),
\tag{90}
\]

\[
a_{1s}=G_s'(p_{0,s})-a_{2s}p_{0,s},
\tag{91}
\]

\[
a_{0s}=G_s(p_{0,s})-G_s'(p_{0,s})p_{0,s}
+\frac12a_{2s}p_{0,s}^2.
\tag{92}
\]

Suppose

\[
d(q)=d_2q^2+d_1q+d_0
\]

and seek

\[
\widehat H(\tau,q)=A(\tau)q^2+B(\tau)q+C(\tau).
\tag{93}
\]

The approximate HJB closes:

\[
\boxed{A'=d_2+2(a_{2+}+a_{2-})A^2,}
\tag{94}
\]

\[
\boxed{
B'
=d_1+2A\left[
 a_{1+}-a_{1-}
 +(a_{2+}+a_{2-})B
 +(a_{2-}-a_{2+})A
\right],
}
\tag{95}
\]

\[
\boxed{
\begin{aligned}
C'={}&d_0+a_{0+}+a_{0-}
+a_{1+}(B-A)-a_{1-}(A+B)\\
&+\frac12a_{2+}(B-A)^2
+\frac12a_{2-}(A+B)^2.
\end{aligned}
}
\tag{96}
\]

with

\[
A(0)=-\alpha,
\qquad B(0)=C(0)=0.
\tag{97}
\]

Equation (94) is scalar Riccati; once \(A\) is known, (95) is linear in \(B\), and (96) is one quadrature. Arbitrary side asymmetry is retained.

This approximation ignores exact finite-inventory boundary effects and replaces the original exponentials. It is not an exact solution of (1).

### 9.1 Choosing expansion centers

Expanding at \(p=0\) is often inferior to expanding around the observed gap range from the symmetric solution or stationary corrector. The Taylor remainder satisfies

\[
\boxed{
|G_s(p)-\widehat G_s(p)|
\le
\frac{k_s^3G_s(p_{0,s})}{6}
 e^{k_s|p-p_{0,s}|}|p-p_{0,s}|^3.
}
\tag{98}
\]

Thus the relevant local variable is \(k_s|p-p_{0,s}|\). Use the original-HJB residual to certify the resulting quote error.

---

## 10. Rigorous residual-to-quote error certificate

Let the exact reverse-time system be

\[
\dot H=F(H),
\qquad H(0)=H_0,
\]

and let \(\widetilde H\) be any differentiable approximation with the same initial data. Define its residual

\[
R(\tau)=\dot{\widetilde H}(\tau)-F(\widetilde H(\tau)).
\tag{99}
\]

Set

\[
E(\tau)=\int_0^\tau\|R(s)\|_\infty\,ds.
\tag{100}
\]

Because the vector field is cooperative and translation invariant,

\[
\widetilde H-E\mathbf1
\]

is a subsolution and

\[
\widetilde H+E\mathbf1
\]

is a supersolution. The comparison principle gives

\[
\boxed{
\|H(\tau)-\widetilde H(\tau)\|_\infty
\le
\int_0^\tau\|R(s)\|_\infty\,ds.
}
\tag{101}
\]

Therefore every neighboring value difference satisfies

\[
\boxed{
|\Delta H_{i}(\tau)-\Delta\widetilde H_{i}(\tau)|
\le
2\int_0^\tau\|R(s)\|_\infty\,ds.
}
\tag{102}
\]

If the approximation has an initial mismatch, add \(\|H_0-\widetilde H(0)\|_\infty\) to the right-hand side of (101).

This certificate is especially useful for:

- first/second-order asymmetry perturbations;
- one-shot exponential defect corrections;
- quadratic-Hamiltonian/Riccati surrogates;
- stationary and terminal-layer approximations;
- offline surrogate validation.

For splitting, the raw continuous residual may be pessimistic because commutator errors cancel over a symmetric step. Step doubling is normally a sharper operational estimate.

---

## 11. Ground-truth finite-horizon solver

A reliable reference implementation should integrate the reverse-time initial-value problem rather than solve a global backward fixed-point system.

### 11.1 Recommended choices

- **Nonstiff or mildly stiff:** DOP853 or another high-order embedded explicit Runge–Kutta method.
- **Stiff optimized fill-rate regimes:** Rosenbrock, Radau, or BDF with the analytic Jacobian.
- **State variables:** 12 adjacent gaps for quote-only calculations, or 13 gauge-normalized values.
- **Tolerances:** set directly from acceptable quote-gap error, not from an arbitrary value-function tolerance.

### 11.2 Existing backward-implicit solver acceleration

If retaining backward Euler or BDF2:

1. use the analytic tridiagonal Jacobian (20)–(21);
2. solve Newton systems with Thomas elimination;
3. use extrapolation from the preceding time point as predictor;
4. use the symmetric or perturbative solution as the initial path for the first asymmetric parameter point;
5. traverse parameter sets by continuation, so each solution initializes a nearby one;
6. use a safeguarded/damped Newton step only when needed;
7. stop Newton from a quote-gap defect criterion rather than a stricter absolute-level criterion.

At 13 states, reducing Newton iterations from four to one is far more valuable than changing dense \(O(N^3)\) algebra to banded \(O(N)\) algebra.

### 11.3 Why global spectral-in-time collocation is usually not first choice

A Chebyshev-in-time discretization can converge rapidly for this analytic finite-dimensional ODE, but it creates a nonlinear system of size \(13M\) with dense temporal coupling. It is attractive for an offline family or when many horizon endpoints are needed from one global representation. For isolated live solves, a high-order IVP or exact-side splitting is usually simpler and faster.

---

## 12. Large-\(k\) scaling: what is and is not small

Choose a reference decay \(k_0\) and a reference event rate \(\beta_0\). Define

\[
y_i=k_0H_i,
\qquad
s=\beta_0\tau.
\]

The dimensionless problem depends on combinations such as

\[
\frac{k_+}{k_0},
\quad
\frac{k_-}{k_0},
\quad
\frac{k_0d_i}{\beta_0},
\quad
k_0\alpha,
\quad
\beta_0T,
\quad
k_s\epsilon_s,
\]

plus relative intensity scales. Raw \(k_s\to\infty\) is not a unique asymptotic limit unless the penalty, terminal condition, impact, and time horizon are scaled at the same time.

This nondimensionalization is valuable for surrogate construction: it reduces the parameter space to ratios and scaled penalties instead of feeding very large dimensional numbers directly into an interpolator.

A continuum/WKB approximation should not be justified merely by large \(k_s\). The inventory-Taylor expansion requires smoothness in \(q\) and small enough \(k_s\Delta H\), and the boundaries at \(q=\pm6\) can matter materially.

---

## 13. Continuum asymmetric Cole–Hopf approximation

For completeness, a single Cole–Hopf exponent reappears after a second-order continuum truncation in inventory.

Let \(G_+(p)=c_+e^{-k_+p}\) and \(G_-(p)=c_-e^{-k_-p}\). Expanding the unit inventory differences and the Hamiltonians to second order gives an approximate PDE

\[
H_\tau
=V(q)+bH_q+\nu H_{qq}+\mu(H_q)^2,
\tag{103}
\]

where

\[
b=G_+'(0)-G_-'(0),
\tag{104}
\]

\[
\nu=-\frac12\left[G_+'(0)+G_-'(0)\right]>0,
\tag{105}
\]

\[
\mu=\frac12\left[G_+''(0)+G_-''(0)\right]>0.
\tag{106}
\]

Then

\[
\Psi=e^{(\mu/\nu)H}
\tag{107}
\]

linearizes the approximate PDE:

\[
\Psi_\tau
=\nu\Psi_{qq}+b\Psi_q+\frac{\mu}{\nu}V(q)\Psi.
\tag{108}
\]

For the coefficient in (3),

\[
\frac{\mu}{\nu}
=
\frac{\lambda_+k_++\lambda_-k_-}{\lambda_++\lambda_-}.
\tag{109}
\]

This weighted decay is an approximation created by the continuum/Taylor truncation. It does not contradict the exact discrete no-go theorem.

For \(q_{\max}=6\), this route should rank below exact-side splitting, stationary reduction, and the quadratic-Hamiltonian approximation unless a residual check demonstrates that it is accurate over the actual gap range.

---

## 14. Production architecture for \(10^4\)–\(10^5\) calls

### 14.1 Offline preparation

For each slowly varying calibration family:

1. normalize parameters into dimensionless groups;
2. compute exact reference solutions with a tight adaptive IVP;
3. compute stationary gaps and spectral gaps;
4. test both Strang orderings over the parameter envelope;
5. choose a fixed step count satisfying the quote tolerance;
6. compute first and second \(\kappa\)-sensitivities if near-symmetric calls are common;
7. fit a low-dimensional surrogate only after the numerical solver is validated;
8. retain an out-of-sample error set and an automatic fallback.

### 14.2 Runtime dispatcher

A practical dispatcher can use:

1. **Stationary branch:** if the stationary node residual is verified, \(gT\) exceeds a calibrated threshold, and the terminal-layer residual is below tolerance.
2. **Perturbation branch:** if the first/second-order approximation passes the residual bound.
3. **Surrogate branch:** if the parameter point is inside the certified interpolation domain.
4. **Exact-side splitting:** default low-latency arbitrary-asymmetry branch.
5. **Adaptive IVP fallback:** for points that fail all fast checks.

The dispatcher should be driven by quote-gap errors, not by value levels.

### 14.3 Batch and systems-level optimization

For this state size:

- batch many parameter vectors over a common step grid;
- compile the RHS, side-flow convolution, and step loop with C++, Rust, Numba, JAX, or an equivalent tool;
- use structure-of-arrays layout for parameters;
- precompute \(\log(\ell!)\), shift indices, and fixed-step weights;
- if \(\lambda_s\) and \(h\) are fixed, reuse side-flow convolution matrices;
- avoid Python object creation inside each solve;
- do not store the full time path if only \(t=0\) quotes are required;
- subtract a gauge every step and track the common offset only if needed;
- sort parameter sweeps so neighboring calls are close, maximizing warm-start and cache utility;
- benchmark wall-clock performance with the actual parameter distribution rather than with one “typical” point.

A 13-state solve is so small that interpreter overhead can dominate the floating-point work. A compiled fixed-step splitting loop can therefore outperform a theoretically higher-order library solver even when it takes more mathematical stages.

---

## 15. Reference Python implementation

The code below prioritizes clarity and numerical safety. For production, compile/vectorize the loops and use the short normalized convolution kernel when its dynamic range is safe.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HJBParams:
    d: FloatArray
    c_plus: float
    c_minus: float
    k_plus: float
    k_minus: float

    def validate(self) -> None:
        if self.d.ndim != 1 or self.d.size < 2:
            raise ValueError("d must be a one-dimensional array with >= 2 states")
        if self.c_plus < 0.0 or self.c_minus < 0.0:
            raise ValueError("c_plus and c_minus must be nonnegative")
        if self.k_plus <= 0.0 or self.k_minus <= 0.0:
            raise ValueError("k_plus and k_minus must be positive")


def hjb_rhs(H: FloatArray, p: HJBParams) -> FloatArray:
    """Reverse-time RHS for equation (1), states ordered increasingly in q."""
    p.validate()
    H = np.asarray(H, dtype=np.float64)
    if H.shape != p.d.shape:
        raise ValueError("H and d must have the same shape")

    out = p.d.astype(np.float64, copy=True)
    out[1:] += p.c_plus * np.exp(
        -p.k_plus * (H[1:] - H[:-1])
    )
    out[:-1] += p.c_minus * np.exp(
        -p.k_minus * (H[:-1] - H[1:])
    )
    return out


def jacobian_bands(
    H: FloatArray, p: HJBParams
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Return lower, diagonal, upper bands of the analytic Jacobian."""
    p.validate()
    H = np.asarray(H, dtype=np.float64)
    n_state = H.size

    lower = np.empty(n_state - 1, dtype=np.float64)
    upper = np.empty(n_state - 1, dtype=np.float64)
    diag = np.zeros(n_state, dtype=np.float64)

    a = p.c_plus * np.exp(-p.k_plus * (H[1:] - H[:-1]))
    b = p.c_minus * np.exp(-p.k_minus * (H[:-1] - H[1:]))

    lower[:] = p.k_plus * a
    upper[:] = p.k_minus * b
    diag[1:] -= lower
    diag[:-1] -= upper
    return lower, diag, upper


def side_flow_logsumexp(
    H: FloatArray,
    dt: float,
    c: float,
    k: float,
    direction: Literal["lower", "upper"],
) -> FloatArray:
    """Exact positive-time one-sided flow, equations (32) and (33)."""
    H = np.asarray(H, dtype=np.float64)
    if dt < 0.0:
        raise ValueError("negative side-flow time can violate positivity")
    if c < 0.0 or k <= 0.0:
        raise ValueError("require c >= 0 and k > 0")
    if dt == 0.0 or c == 0.0:
        return H.copy()

    beta_dt = dt * k * c
    log_beta_dt = np.log(beta_dt)

    # Any common gauge works. The maximum keeps scaled exponentials <= 1.
    gauge = float(np.max(H))
    x = k * (H - gauge)
    out = np.empty_like(H)
    n_state = H.size

    if direction == "lower":
        for i in range(n_state):
            ell = np.arange(i + 1, dtype=np.float64)
            terms = (
                x[i - ell.astype(np.int64)]
                + ell * log_beta_dt
                - gammaln(ell + 1.0)
            )
            out[i] = gauge + logsumexp(terms) / k
    elif direction == "upper":
        for i in range(n_state):
            ell = np.arange(n_state - i, dtype=np.float64)
            terms = (
                x[i + ell.astype(np.int64)]
                + ell * log_beta_dt
                - gammaln(ell + 1.0)
            )
            out[i] = gauge + logsumexp(terms) / k
    else:
        raise ValueError("direction must be 'lower' or 'upper'")

    return out


def side_flow_convolution(
    H: FloatArray,
    dt: float,
    c: float,
    k: float,
    direction: Literal["lower", "upper"],
) -> FloatArray:
    """
    Faster exact one-sided flow using normalized positive convolution.
    Fall back to log-sum-exp if the scaled range is too large.
    """
    H = np.asarray(H, dtype=np.float64)
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    if dt == 0.0 or c == 0.0:
        return H.copy()

    gauge = float(np.max(H))
    scaled = k * (H - gauge)
    if np.min(scaled) < -700.0:
        return side_flow_logsumexp(H, dt, c, k, direction)

    x = np.exp(scaled)
    n_state = H.size
    weights = np.empty(n_state, dtype=np.float64)
    weights[0] = 1.0
    z = dt * k * c
    for ell in range(1, n_state):
        weights[ell] = weights[ell - 1] * z / ell

    if direction == "lower":
        y = np.convolve(x, weights, mode="full")[:n_state]
    elif direction == "upper":
        y = np.convolve(x[::-1], weights, mode="full")[:n_state][::-1]
    else:
        raise ValueError("direction must be 'lower' or 'upper'")

    if np.any(y <= 0.0) or not np.all(np.isfinite(y)):
        return side_flow_logsumexp(H, dt, c, k, direction)
    return gauge + np.log(y) / k


def strang_step(H: FloatArray, dt: float, p: HJBParams) -> FloatArray:
    """One plus-drift-minus-drift-plus symmetric splitting step."""
    x = side_flow_convolution(
        H, 0.5 * dt, p.c_plus, p.k_plus, "lower"
    )
    x = x + 0.5 * dt * p.d
    x = side_flow_convolution(
        x, dt, p.c_minus, p.k_minus, "upper"
    )
    x = x + 0.5 * dt * p.d
    x = side_flow_convolution(
        x, 0.5 * dt, p.c_plus, p.k_plus, "lower"
    )
    return x


def solve_strang_fixed(
    H0: FloatArray,
    horizon: float,
    n_steps: int,
    p: HJBParams,
    gauge_index: int | None = None,
) -> FloatArray:
    """
    Fixed-step quote-safe solver with merged adjacent outer half-flows.
    Removed common offsets are tracked exactly.
    """
    if horizon < 0.0 or n_steps <= 0:
        raise ValueError("horizon must be nonnegative and n_steps positive")
    p.validate()

    x = np.asarray(H0, dtype=np.float64).copy()
    if x.shape != p.d.shape:
        raise ValueError("H0 and d must have the same shape")

    if gauge_index is None:
        gauge_index = x.size // 2
    if not 0 <= gauge_index < x.size:
        raise ValueError("invalid gauge_index")

    dt = horizon / n_steps
    common_offset = 0.0

    # Initial outer half-flow. At interior step boundaries the ending and
    # starting half-flows merge into one full plus-side flow.
    x = side_flow_convolution(
        x, 0.5 * dt, p.c_plus, p.k_plus, "lower"
    )

    for step in range(n_steps):
        x = x + 0.5 * dt * p.d
        x = side_flow_convolution(
            x, dt, p.c_minus, p.k_minus, "upper"
        )
        x = x + 0.5 * dt * p.d

        plus_dt = dt if step < n_steps - 1 else 0.5 * dt
        x = side_flow_convolution(
            x, plus_dt, p.c_plus, p.k_plus, "lower"
        )

        shift = float(x[gauge_index])
        x -= shift
        common_offset += shift

    return x + common_offset


def adjacent_gaps(H: FloatArray) -> FloatArray:
    H = np.asarray(H, dtype=np.float64)
    return H[1:] - H[:-1]


def step_doubling_gap_error(
    H0: FloatArray,
    horizon: float,
    n_steps: int,
    p: HJBParams,
) -> Tuple[FloatArray, float]:
    """Return fine solution and second-order fine-grid gap error estimate."""
    coarse = solve_strang_fixed(H0, horizon, n_steps, p)
    fine = solve_strang_fixed(H0, horizon, 2 * n_steps, p)
    err = np.max(np.abs(adjacent_gaps(fine) - adjacent_gaps(coarse))) / 3.0
    return fine, float(err)
```

### 15.1 Stationary implementations

The first function is the very fast scalar shooting path. It deliberately requires a sign-changing bracket already known to lie in one feasible branch—for example, from the preceding parameter point. It verifies the full node residual before returning.

The second function is the robust log-edge homotopy fallback.

```python
from dataclasses import dataclass

from scipy.linalg import eig
from scipy.optimize import root


@dataclass(frozen=True)
class StationaryResult:
    rho: float
    gaps: FloatArray
    a_edges: FloatArray
    b_edges: FloatArray


def _shooting_eval(
    rho: float,
    d: FloatArray,
    c_plus: float,
    c_minus: float,
    gamma: float,
):
    """Return (terminal residual, a_edges, b_edges), or None if infeasible."""
    n_state = d.size
    a = np.empty(n_state - 1, dtype=np.float64)
    b = np.empty(n_state - 1, dtype=np.float64)

    b_prev = rho - d[0]
    if b_prev <= 0.0 or not np.isfinite(b_prev):
        return None
    b[0] = b_prev

    for i in range(1, n_state):
        a_i = c_plus * (b_prev / c_minus) ** (-gamma)
        if not np.isfinite(a_i):
            return None
        a[i - 1] = a_i

        if i == n_state - 1:
            return float(rho - d[i] - a_i), a, b

        b_prev = rho - d[i] - a_i
        if b_prev <= 0.0 or not np.isfinite(b_prev):
            return None
        b[i] = b_prev

    raise RuntimeError("unreachable")


def _stationary_node_residual(
    rho: float,
    d: FloatArray,
    a_edges: FloatArray,
    b_edges: FloatArray,
) -> FloatArray:
    node = d.astype(np.float64, copy=True)
    node[0] += b_edges[0]
    if node.size > 2:
        node[1:-1] += a_edges[:-1] + b_edges[1:]
    node[-1] += a_edges[-1]
    return node - rho


def stationary_shooting_from_bracket(
    d: FloatArray,
    c_plus: float,
    c_minus: float,
    k_plus: float,
    k_minus: float,
    rho_lo: float,
    rho_hi: float,
    tol: float = 1e-13,
    residual_tol: float = 1e-10,
    max_iter: int = 200,
) -> StationaryResult:
    """
    Fast scalar shooting on one known feasible branch.

    Both endpoints must be feasible and satisfy F(rho_lo) <= 0 <= F(rho_hi).
    Use the log-edge homotopy solver when such a bracket is unavailable.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be one-dimensional with at least two states")
    if min(c_plus, c_minus, k_plus, k_minus) <= 0.0:
        raise ValueError("c and k parameters must be positive")
    if not rho_lo < rho_hi:
        raise ValueError("require rho_lo < rho_hi")

    gamma = k_plus / k_minus
    left = _shooting_eval(rho_lo, d, c_plus, c_minus, gamma)
    right = _shooting_eval(rho_hi, d, c_plus, c_minus, gamma)
    if left is None or right is None:
        raise ValueError("both bracket endpoints must be on one feasible branch")
    if left[0] > 0.0 or right[0] < 0.0:
        raise ValueError("bracket must satisfy F(lo) <= 0 <= F(hi)")

    lo, hi = float(rho_lo), float(rho_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = _shooting_eval(mid, d, c_plus, c_minus, gamma)
        if value is None:
            raise RuntimeError(
                "bracket crossed a singular branch; use log-edge homotopy"
            )
        if value[0] < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    else:
        raise RuntimeError("stationary shooting did not converge")

    rho = 0.5 * (lo + hi)
    value = _shooting_eval(rho, d, c_plus, c_minus, gamma)
    if value is None:
        raise RuntimeError("stationary shooting became infeasible")
    _, a_edges, b_edges = value
    gaps = np.log(b_edges / c_minus) / k_minus

    residual = _stationary_node_residual(rho, d, a_edges, b_edges)
    scale = max(
        1.0,
        abs(rho),
        float(np.max(np.abs(d))),
        float(np.max(a_edges)),
        float(np.max(b_edges)),
    )
    if np.max(np.abs(residual)) > residual_tol * scale:
        raise RuntimeError(
            "scalar shooting failed its full residual check; "
            "use log-edge homotopy"
        )
    return StationaryResult(rho, gaps, a_edges, b_edges)


def _log_stationary_residual_jacobian(
    x: FloatArray,
    d: FloatArray,
    c_plus: float,
    c_minus: float,
    k_plus: float,
    k_minus: float,
) -> Tuple[FloatArray, FloatArray]:
    """Residual and analytic Jacobian in y_i = k_minus * r_i coordinates."""
    n_state = d.size
    y = x[:-1]
    rho = float(x[-1])
    gamma = k_plus / k_minus

    # Clipping is only a guard for failed trial iterates. Continuation should
    # keep accepted iterates far from the clipping threshold.
    b = c_minus * np.exp(np.clip(y, -700.0, 700.0))
    a = c_plus * np.exp(np.clip(-gamma * y, -700.0, 700.0))

    f = np.empty(n_state, dtype=np.float64)
    f[0] = d[0] + b[0] - rho
    if n_state > 2:
        f[1:-1] = d[1:-1] + a[:-1] + b[1:] - rho
    f[-1] = d[-1] + a[-1] - rho

    J = np.zeros((n_state, n_state), dtype=np.float64)
    J[0, 0] = b[0]
    if n_state > 2:
        rows = np.arange(1, n_state - 1)
        J[rows, rows - 1] = -gamma * a[:-1]
        J[rows, rows] = b[1:]
    J[-1, -2] = -gamma * a[-1]
    J[:, -1] = -1.0
    return f, J


def _symmetric_stationary_initial(
    d: FloatArray,
    c_plus: float,
    c_minus: float,
    k_bar: float,
) -> Tuple[FloatArray, float]:
    """Physical gaps and rho from the symmetric-decay Perron eigenpair."""
    n_state = d.size
    A = np.diag(k_bar * d)
    A[np.arange(1, n_state), np.arange(n_state - 1)] = k_bar * c_plus
    A[np.arange(n_state - 1), np.arange(1, n_state)] = k_bar * c_minus

    values, vectors = eig(A)
    idx = int(np.argmax(values.real))
    if abs(values[idx].imag) > 1e-9 * max(1.0, abs(values[idx].real)):
        raise RuntimeError("principal eigenvalue is unexpectedly complex")

    z = vectors[:, idx].real
    if np.sum(z) < 0.0:
        z = -z
    if np.any(z <= 0.0):
        # A positive Perron vector exists; abs handles tiny sign noise.
        z = np.abs(z)
    if np.any(z <= 0.0):
        raise RuntimeError("failed to recover a positive Perron eigenvector")

    z /= np.max(z)
    gaps = np.diff(np.log(z)) / k_bar
    rho = float(values[idx].real / k_bar)
    return gaps, rho


def stationary_gaps_log_homotopy(
    d: FloatArray,
    c_plus: float,
    c_minus: float,
    k_plus: float,
    k_minus: float,
    initial_steps: int = 16,
    residual_tol: float = 1e-10,
    min_theta_step: float = 1e-7,
) -> StationaryResult:
    """
    Robust stationary solve in log-edge variables, continued from equal decay.

    c_plus, c_minus, and d are held fixed along the homotopy; only the two
    decay rates move from k_bar to their target values.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("d must be one-dimensional with at least two states")
    if min(c_plus, c_minus, k_plus, k_minus) <= 0.0:
        raise ValueError("c and k parameters must be positive")
    if initial_steps <= 0:
        raise ValueError("initial_steps must be positive")

    k_bar = 0.5 * (k_plus + k_minus)
    gaps, rho = _symmetric_stationary_initial(
        d, c_plus, c_minus, k_bar
    )

    theta = 0.0
    theta_step = 1.0 / initial_steps
    while theta < 1.0:
        trial = min(1.0, theta + theta_step)
        kp = k_bar + trial * (k_plus - k_bar)
        km = k_bar + trial * (k_minus - k_bar)
        x0 = np.concatenate((km * gaps, np.array([rho])))

        def fun(x: FloatArray) -> FloatArray:
            return _log_stationary_residual_jacobian(
                x, d, c_plus, c_minus, kp, km
            )[0]

        def jac(x: FloatArray) -> FloatArray:
            return _log_stationary_residual_jacobian(
                x, d, c_plus, c_minus, kp, km
            )[1]

        sol = root(fun, x0, jac=jac, method="hybr", options={"xtol": 1e-11})
        f = fun(sol.x)
        scale = max(
            1.0,
            abs(float(sol.x[-1])),
            float(np.max(np.abs(d))),
            c_plus,
            c_minus,
        )
        accepted = (
            sol.success
            and np.all(np.isfinite(sol.x))
            and np.max(np.abs(f)) <= residual_tol * scale
        )

        if accepted:
            theta = trial
            gaps = sol.x[:-1] / km
            rho = float(sol.x[-1])
            theta_step = min(1.5 * theta_step, 1.0 - theta)
            continue

        theta_step *= 0.5
        if theta_step < min_theta_step:
            raise RuntimeError(
                "stationary homotopy failed; use a long finite-horizon "
                "solution as the initial gap vector or increase precision"
            )

    y = k_minus * gaps
    gamma = k_plus / k_minus
    b_edges = c_minus * np.exp(y)
    a_edges = c_plus * np.exp(-gamma * y)

    residual = _stationary_node_residual(rho, d, a_edges, b_edges)
    scale = max(
        1.0,
        abs(rho),
        float(np.max(np.abs(d))),
        float(np.max(a_edges)),
        float(np.max(b_edges)),
    )
    if np.max(np.abs(residual)) > residual_tol * scale:
        raise RuntimeError("log-edge stationary solve failed residual check")

    return StationaryResult(rho, gaps, a_edges, b_edges)
```

### 15.2 Mapping gaps to quotes

For states ordered increasingly in inventory and the convention in (5)–(6):

```python
def quote_depths_from_gaps(
    gaps: FloatArray,
    k_plus: float,
    k_minus: float,
    epsilon_plus: float = 0.0,
    epsilon_minus: float = 0.0,
) -> Tuple[FloatArray, FloatArray]:
    """
    plus_depth[i-1] applies at state i, i=1,...,N-1;
    minus_depth[i] applies at state i, i=0,...,N-2.
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    plus_depth = epsilon_plus + 1.0 / k_plus + gaps
    minus_depth = epsilon_minus + 1.0 / k_minus - gaps
    return plus_depth, minus_depth
```

Check this mapping against the naming convention in the trading code; bid/ask labels are often swapped across texts while the neighbor differences remain the same.

### 15.3 Acceptance tests before production use

1. **Symmetric recovery.** Set \(k_+=k_-\) and compare the full endpoint and all quote gaps against the tridiagonal matrix-exponential solution. This catches the transform-sign and neighbor-index errors immediately.
2. **One-sided derivative.** Verify numerically that \(\bigl(\Phi_+^h(H)-H\bigr)/h\to P(H)\) and similarly for the minus side as \(h\to0\).
3. **Boundary audit.** At the minimum inventory, confirm that the forbidden lower-neighbor term is absent; at the maximum inventory, confirm that the forbidden upper-neighbor term is absent.
4. **Second-order ratio.** Against the ground-truth IVP, halving the splitting step should reduce the asymptotic quote-gap error by approximately a factor of four.
5. **Gauge invariance.** Add a large constant to every initial \(H_i\); the returned quote gaps must be unchanged to roundoff.
6. **Stationary verification.** Reconstruct every node in (45). A small terminal shooting residual alone is not sufficient.
7. **Residual certification.** For every approximation branch, evaluate the original, unapproximated HJB—not the transformed or quadratic surrogate equation.
8. **Parameter-envelope benchmark.** Measure latency and errors at the tails of the real calibration distribution, including the highest optimized fill rates and the largest terminal penalties.

---

## 16. Recommended decision rule

For the stated 13-state application, the following order is practical:

### A. Build and retain one trusted reference solver

Use a tight adaptive IVP in the 12 gaps with the analytic Jacobian. All approximations and fast solvers are validated against it.

### B. Use the verified stationary solver whenever justified

First try the scalar shooting recurrence and verify the reconstructed node residual. If it is ill-conditioned, use the log-edge homotopy solver. Then compute the stationary spectral gap. If the calibrated finite-horizon error is below the quote tolerance, this is the fastest general asymmetric solution.

### C. Use exact-side splitting as the robust live default

It is arbitrary-asymmetry, positive-time stable, Newton-free, easy to compile, and has a direct gap error estimator. Preselect a fixed step count offline for deterministic latency.

### D. Use perturbative or defect-correction branches for large sweeps

When many calls share a symmetric/reference generator, precompute its spectral decomposition and sensitivities. Accept only when the original-HJB residual certifies the quote error.

### E. Use the Riccati surrogate only as a certified approximation

It is nearly free and can be very useful for an initial guess, a preconditioner, or an ultra-low-latency branch. It should not be labeled exact.

### F. Build an offline surrogate if the live parameter domain is low dimensional

A Chebyshev or sparse-grid map from dimensionless parameters to the 24 active quote depths can eliminate nearly all online ODE work. Keep the exact solver as an out-of-domain fallback and monitor interpolation residuals.

---

## 17. Findings classified by certainty

### Exact statements

- For the displayed HJB, the symmetric transform is \(e^{+kH}\).
- No invertible componentwise finite-dimensional transform can affine-linearize both active sides when \(k_+\ne k_-\).
- Each one-sided nonlinear flow is exactly linearizable and has the finite formulas (32)–(33).
- The nonlinear Trotter product converges to the exact asymmetric finite-horizon flow.
- On each connected feasible branch, the stationary shooting residual is strictly increasing; a fast scalar recurrence is available, with a log-edge homotopy solver for ill-conditioned bottlenecks.
- The Jacobian is tridiagonal, cooperative, and has zero row sums.
- The residual bound (101)–(102) converts HJB defect to value and quote-gap error.

### Controlled numerical approximations

- Finite-step Strang splitting is second order.
- First/second-order \(\kappa\) sensitivities are regular finite-horizon perturbations around the symmetric solution.
- Exponential defect correction and ETD propagate a reference symmetric part exactly.
- The stationary terminal-layer formula is a linearization, not an exact finite-horizon solution.
- The side-specific quadratic-Hamiltonian model is a Riccati approximation.
- The continuum asymmetric Cole–Hopf transform linearizes only the truncated PDE.

### Claims not established

- No theorem here rules out every imaginable nonlocal transform tailored to one fully fixed parameter vector.
- No universal relative-asymmetry threshold guarantees quote accuracy.
- Large raw \(k_s\) alone does not justify WKB or continuum asymptotics.
- No wall-clock speedup is guaranteed without benchmarking the actual compiled implementation and parameter distribution.

---

## 18. Primary references

1. Á. Cartea, S. Jaimungal, and J. Penalva, *Algorithmic and High-Frequency Trading*, Cambridge University Press, 2015, Chapter 10, especially Eqs. 10.26–10.29. Official author page: [Algorithmic and High-Frequency Trading](https://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/).

2. O. Guéant, “Optimal Market Making,” *Applied Mathematical Finance* 24(2), 112–154, 2017. The general Hamiltonian system is nonlinear; the exact finite-dimensional linearization in Eq. (3.13) assumes identical exponential bid/ask intensity functions. [arXiv:1605.01862](https://arxiv.org/html/1605.01862v5).

3. P. Fodra and M. Labadie, “High-frequency market-making for multi-dimensional Markov processes,” 2013. The asymmetric-intensity discussion notes that unequal side decays leave two different nonlinear exponentials. [arXiv:1303.7177](https://arxiv.org/html/1303.7177v2).

4. P. Bergault, D. Evangelista, O. Guéant, and D. Vieira, “Closed-form Approximations in Multi-asset Market Making,” *Applied Mathematical Finance* 28(2), 101–142. Side-specific quadratic Hamiltonians lead to Riccati equations and closed-form approximations; later corrections are perturbative. [arXiv:1810.04383](https://arxiv.org/html/1810.04383v5).

5. O. Guéant and I. Manziuk, “Optimal control on graphs: existence, uniqueness, and long-term behavior,” *ESAIM: Control, Optimisation and Calculus of Variations* 26, 2020. Relevant results include comparison, sup-norm nonexpansiveness, the ergodic equation, and long-time convergence of value differences. [arXiv:1902.08926](https://arxiv.org/html/1902.08926v6).

6. S. Blanes, F. Casas, and A. Murua, “Splitting methods for differential equations,” *Acta Numerica*, 2024. Review of Lie–Trotter, Strang, nonlinear flow compositions, error structure, and higher-order constraints. [arXiv:2401.01722](https://arxiv.org/html/2401.01722v3).

7. E. Todorov, “Linearly-solvable Markov decision problems,” *Advances in Neural Information Processing Systems* 19, 2006. Useful background on why a common exponential/KL scale enables a desirability transform. [NeurIPS paper](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems).

---

## 19. Bottom line

There is no finite statewise Cole–Hopf transform that simultaneously absorbs two unequal decay rates. The equal-\(k\) matrix exponential should therefore be viewed as a special algebraic closure, not as a formula waiting for a trivial asymmetric modification.

The asymmetrical system is nevertheless much more structured than a generic nonlinear 13-dimensional ODE:

\[
\boxed{
\begin{aligned}
&\text{each side has an exact finite log-sum-exp propagator;}\\
&\text{their symmetric composition gives a fast Newton-free solver;}\\
&\text{stationary shooting is scalar on each feasible branch, with a log-domain fallback;}\\
&\text{near symmetry, every perturbation order reuses one tridiagonal generator;}\\
&\text{all approximations can be tested in quote units by defect or step doubling.}
\end{aligned}
}
\]

For the stated workload, the first methods to benchmark are:

1. a compiled fixed-step exact-side Strang solver;
2. verified stationary shooting, with log-edge homotopy fallback and a spectral-gap acceptance test;
3. a tight adaptive gap-IVP as ground truth and fallback;
4. cached symmetric sensitivities or exponential defect correction for repeated sweeps.

That combination is more likely to deliver a material wall-clock reduction than further optimization of a small dense Newton solve.
