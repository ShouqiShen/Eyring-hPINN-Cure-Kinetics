# Constraint and Validity Notes

v2 is more automated than v1, but it should still be treated as a physics-informed model with a defined validity domain.

## Recommended Checks

- Temperature should remain inside or close to the training range.
- Conversion `alpha` should stay in the physical interval `[0, 1]`.
- Heating-rate and isothermal conditions should match the regimes represented in training data.
- Molecular structures should be chemically related to the training set unless uncertainty is explicitly reported.
- Mixture ratios should be interpreted carefully near sparse or extrapolative composition regions.
- Learned thermodynamic quantities should remain within plausible ranges for epoxy cure kinetics.

## v2-Specific Constraints

The Eyring parameterisation relies on physically meaningful activation enthalpy, activation entropy, and Gibbs free energy behaviour. The model can produce strong predictions when these constraints are met, but relaxed or inconsistent constraints may lead to nonphysical rates, unstable extrapolation, or misleading parameter trends.

## Reporting Guidance

When reporting v2 results, include:

- Whether the prediction is interpolation or extrapolation.
- The temperature and conversion range used.
- The model uncertainty band.
- A comparison to the v1 benchmark where possible.
