

TODO:

- add the following decoding methods:
    - Vanilla (w/ temperature 0.0, 0.1, 0.2)
    - Top-K
    - Top-P
    - DoLa-static (layer 8, 16)
    - DoLa (4, 8, 12, 16, ... 28)


- train model to predict best treatment to maximize logprobs
- 



python dlproj_mc_eval.py --top_k=0 --top_p=1.0 --early-exit-layers 0,32 --temperature=1.0 --output-path="./results/eval3/dola_s0_t1.0"



python dlproj_mc_eval.py --top_k=0 --top_p=1.0 --early-exit-layers 8,32 --temperature=1.0 --output-path="./results/mc_evals/adaptive_dola"