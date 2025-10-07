# Quantization of NanoGPT

## Project for 194.077 Applied Deep Learning 

**Topic:** Post‑training integer weight quantization of a small autoregressive transformer.

**Type:** Bring‑your‑own‑method.

## References

1. LeCun, Denker, Solla — *Optimal Brain Damage* — [NeurIPS 1989](https://papers.neurips.cc/paper_files/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
2. Hassibi & Stork — *Optimal Brain Surgeon* — [NeurIPS 1992](https://papers.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
3. Frantar et al. — *GPTQ: Accurate Post‑Training Quantization for Generative Pre‑trained Transformers* — [ICLR 2023](https://openreview.net/forum?id=tcbBPnfwxS)
3. Karpathy — *nanoGPT* — [GitHub Repo](https://github.com/karpathy/nanoGPT)

## Approach

Re-implement a small autoregressive transformer (NanoGPT, ~1–2 M params), train on Tiny Shakespeare, and implement two post‑training integer (int8) weight quantization techniques.

1. **Naive rounding quantization:** quantize each weight to the nearest representable int8 value.
2. **Hessian‑aware quantization:** iteratively quantize weights and adjust remaining weights using a Hessian approximation computed on a small calibration set.

Both quantized variants will be compared to the floating‑point baseline. All will be done on a GTX 960M (4 GB VRAM).

## Dataset

*Primary dataset:* Tiny Shakespeare (~1M tokens).

*Preprocessing:* Simple byte/char tokenization and a shuffled train/validation split.

## Work breakdown

1. Literature & repo setup (4 h)
2. Dataset collection & preprocessing (4 h)
3. Implement NanoGPT & sanity training runs (6 h)
4. Full baseline training (12 h)
5. Implement naive weight quantization (4 h)
6. Implement Hessian-aware weight quantization (8 h)
7. Evaluate model variants and generate plots (7 h)
8. Report writing & presentation prep (10 h)

Total: 55 h
