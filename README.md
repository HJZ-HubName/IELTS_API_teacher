# IELTS Writing Scoring: LLM Trait Scores and Supervised Calibration

This repository contains the code and saved scoring data for a hybrid framework for IELTS Writing Task 2 scoring. An LLM generates scores for four writing criteria, and a supervised model maps these scores to a holistic score. The experiments compare Ridge Regression and LightGBM with arithmetic mean aggregation and holistic LLM scoring.

## Data source and preprocessing

The original data come from the [IELTS Writing Scored Essays Dataset on Kaggle](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset).

This project uses the Writing Task 2 essays. Duplicate records were removed, leaving 711 essays. The original record identifiers were retained rather than replaced with consecutive numbers, so gaps in the identifiers are expected. These identifiers link the essay data to the generated scores and evaluation results.

The prepared input data for Direct and CoT scoring use this deduplicated set. The `Overall` field from the source data is used as the reference score. Reference scores are saved with the generated scores for subsequent evaluation; they are not included in the prompts sent to the LLMs.

## Code folders

| Folder | Purpose |
| --- | --- |
| `API_score_Direct/` | Generates scores through Direct prompting. It makes separate requests for Task Response, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy, followed by an independent request for a holistic score. The prompts request numerical scores without explanations. |
| `API_score_CoT/` | Generates scores using a two-step rationale-then-score procedure. For each criterion, the model first produces an analysis and then receives that analysis in a second request to assign a score. Holistic scoring follows the same two-step procedure. |
| `score_evaluation/` | Reads the saved LLM scores, fits and evaluates the supervised calibration models, and compares them with arithmetic mean aggregation and holistic LLM scoring. It also contains code for score-band analysis and result visualization. |

The folders were renamed when preparing this repository:

| Original folder name | Current folder name |
| --- | --- |
| `prompt_LLM_API_IELTS_score_v1` | `API_score_Direct` |
| `prompt_LLM_API_IELTS_score_CoT` | `API_score_CoT` |
| `ielts_paper1_v1` | `score_evaluation` |

## Saved scoring data

To save space and avoid duplicate copies, the numerical scoring outputs from the Direct and CoT programs are stored under `score_evaluation/score_data_all/`:

| Path | Contents |
| --- | --- |
| `score_evaluation/score_data_all/direct_all/` | Saved scoring CSV files from Direct prompting. |
| `score_evaluation/score_data_all/cot_all/` | Saved scoring CSV files from CoT prompting. |

These files are the outputs of the API scoring stage and the inputs to the calibration and evaluation stage. They are not duplicated in `API_score_Direct/` or `API_score_CoT/`.

Each scoring CSV contains the essay identifier (`essay_id`), the dataset reference score (`human_score`), four LLM-generated criterion scores, and an independently generated holistic score (`ai_overall`). The four criterion scores are used as inputs to the calibration models; `ai_overall` is used for the holistic LLM baseline.

The saved CSV files allow the calibration and evaluation experiments to be run without calling the LLM APIs again. To generate new scores, use the corresponding Direct or CoT program with the prepared essay data and the required API credentials. New API outputs may differ from the saved scores.
