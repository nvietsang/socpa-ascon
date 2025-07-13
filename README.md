# SOCPA on Ascon

Supporting code for the paper [Practical Second-Order CPA Attack on Ascon with Proper Selection Function](https://cascade-conference.org/Archives/2025/Paper/CASCADE25/final-versions/cascade2025-cycleB/cascade2025b-final31.pdf)

## Outline

- [analysis](./analysis/): source code for the analysis in the paper
- [incremental-cpa](./incremental-cpa/): source code for the incremental second-order CPA
- [protected-bi32-armv6](./protected-bi32-armv6/): traces and nonces for testing purpose

## To run the SOCPA

```
cd incremental-cpa
./socpa_k0z0.sh
```

## Contact

Viet-Sang Nguyen [nvietsang@gmail.com](mailto:nvietsang@gmail.com)