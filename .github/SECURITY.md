# Security policy

## Supported versions

Security fixes are made for the latest released version of `spectral-init`.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability
reporting: open the **Security** tab of the repository and choose **Report a
vulnerability**, or go to
<https://github.com/TalonT-Org/spectral-init/security/advisories/new>. Do not open a
public issue for a suspected vulnerability.

Include the version, the target and CPU, and an input that reproduces the problem if you
have one.

## What counts

`spectral-init` is a numerical library that takes a sparse graph and returns coordinates.
It does no I/O and reads no environment variables. The following are treated as security
problems:

- undefined behaviour reachable through the safe public API, in particular through the
  AVX2 kernel selected by `ComputeMode::RustNative`;
- a panic in `spectral_init` on any input, which the crate documents as impossible;
- a dense allocation larger than the configured dense memory budget.

Long running times and large memory use within the documented budget on large inputs
are not vulnerabilities; neither are numerical differences from umap-learn.
