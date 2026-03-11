# Working paper draft

Title: Information Geometry of Context Windows in Open-Source LLMs

Status: living draft driven by artifact-backed results from this repo.

## Research scope

This repo is intended to reproduce the computational claims of the attached paper and extend them with:

- estimator stability checks,
- model-family comparisons,
- context-length sensitivity,
- prompt-template sensitivity,
- generation-length sensitivity,
- seed sensitivity.

## Claim tracker

| Claim | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Information Profile can be estimated from Monte Carlo samples of output continuations. | pending |  |  |
| The profile at the needle position correlates with NIAH retrieval. | pending |  |  |
| A lost-in-the-middle pattern appears in the profile or EIP for some models. | pending |  |  |
| The first-token / sink position carries unusually high sensitivity in some models. | pending |  |  |
| The conclusions are stable to MC sample count and random seed. | pending |  |  |
| The conclusions generalize across more than one open-source model family. | pending |  |  |

## Experimental setup

### Models

| Model | Backend | Context tokens | Status | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Prompt / haystack construction

- Haystacks come from cached corpus text when available, otherwise fallback text.
- A synthetic needle sentence containing a unique code is inserted at controlled positions.
- The model is asked to return only the code.

### Metrics

- Information Profile
- Needle-position sensitivity
- NIAH exact-match accuracy
- Spearman correlation between needle sensitivity and NIAH accuracy
- EIP lost-in-the-middle ratio
- Sink sensitivity

## Current best artifact-backed findings

| Run ID | Model | Main result | Artifact |
| --- | --- | --- | --- |
|  |  |  |  |

## Figures to include

- Position scan: needle sensitivity vs. needle position
- Position scan: NIAH accuracy vs. needle position
- EIP curve over normalized context position
- Cross-model comparison table

## Methods notes

- Only make claims backed by files already present in `artifacts/`.
- Distinguish clearly between theoretical statements from the paper and empirical statements from this repo.
- Note whenever a result depends on small `mc_samples` or limited seeds.
- If a run is inconclusive, record that explicitly instead of smoothing it over.

## Sensitivity section outline

1. MC sample count
2. Random seed
3. Context length
4. Generation length
5. Prompt template
6. Model family / scale

## Limitations

- White-box only: the method needs gradient access.
- Default runs use small MC sample counts for tractability.
- Prompt formatting is intentionally simple and may not be optimal for every instruct model.
- Results may depend on tokenizer and context-budget truncation behavior.

## Next experiments

- [ ] Reproduce the baseline on one small model.
- [ ] Run at least one second model family.
- [ ] Increase MC samples on the most promising baseline.
- [ ] Verify whether the middle of the EIP dips relative to the edges.
- [ ] Add a final abstract only after at least two strong artifact-backed findings exist.

