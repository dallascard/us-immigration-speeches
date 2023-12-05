
# Replication Code and Data

This repo collects together the main scripts used for the data preprocessing and analysis in "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration".

Sufficient scripts and processed data are included in the Release to reproduce the figures and findings in the main paper.

Additional scripts are also included to reproduce the processing of the original raw data, which is available from external sources (see below).  

To replicate analysis and plots with processed data included in Release, jump to **Plots** below.
 

### Requirements:

The following python packages are used in this repo

- shap
- tqdm
- numpy
- scipy
- spacy
- torch
- gensim
- pandas
- pystan
- seaborn
- matplotlib
- smart_open
- scikit-learn
- statsmodels
- transformers



### A Note on Usage:

Note that all scripts in this repo should be run from the main directory using the "-m" option, e.g.:

`python -m analysis.count_county_mentions -h`


### Data Sources:

There are three main sources of data for this project, which are all publicly available from external sources.

The primary source for Congressional data is the Stanford copy of the Congressional Record [https://data.stanford.edu/congress_text](https://data.stanford.edu/congress_text). From this, we use the Hein Bound edition for congresses 43 through 111.

For more recent Congresses (104 through 116) we use the scripts in the USCR repo: https://github.com/unitedstates/congressional-record/

For Presidential data, we scrape data from the [American Presidency Project](https://www.presidency.ucsb.edu/) using scripts in the `app` part of this repo: https://github.com/dallascard/scrapers

Additional tone annotations from the Media Frames Corpus are included in this repo.

For population numbers, we use a combination of sources, as described in the paper. A combined file is included in the Release for this repo. 

Processed data which are too large to be included in the source files for this repo, including trained models and model predictions, are available for download in the [latest release](https://github.com/dallascard/us-immigration-speeches/releases).

### Preprocessing:

There are parallel scripts for processing each part of the data. Steps include preprocessing, tokenization, parsing, and recombining into segments

For the Hein Bound data:

- `parsing/tokenize_hein_bound.py`: tokenize hein-bound using spacy  (also drop speeches from one day with corrupted data, and repair false sentence breaks)
- `parsing/rejoin_into_pieces_by_congress.py`: this script has two purposes: split each speech into one json per sentence, or one json per block of text (up to some limit)

For USCR:

- `uscr/download_legislator_data.py` to download the information on all legislators
- `uscr/export_speeches.py`: export the USCR data to .jsonlist files
- `parsing/preprocess_uscr.py`: adjust the text of USCR to more closely match the Gentzkow data (remove apostrophes, hyphens and speaker names)
- `parsing/tokenize_uscr.py`: output tokenized version of USCR (sentences and tokens)
- `parsing/rejoin_into_pieces_by_congress_uscr.py`: rejoin tokenized sentences into longer segments for classification

For Presidential data:

- use `scrapers/app/combine_categories.py` to combine all data into one file (external repo linked above)
- use `presidential/export_presidential_segments.py` to select the subset of paragraphs from presidents
- use `presidential/tokenize_presidential.py` to tokenize documents
- use `presidential/select_segments.py` to select paragraphs with the relevant keywords


### Speech selection for annotation

As a first step, we selected speech segments that could be about immigration using keywords, which we refer to as "keyword segments":

- `speech_selection/export_segments_early_with_overlap.py`: export segments using the early era keywords, with some overlap to the middle era
- `speech_selection/export_segments_mid_with_overlap.py`: export segments using the middle era keywords, with some overlap to the early and modern eras
- `speech_selection/export_segments_modern_with_overlap.py`: export segments using the modern era keywords, with some overlap to the middle era
- `speech_selection/export_segments_uscr.py`: export segments from USCR

We then combined these into batches, and collected annotations:

- `speech_selection/make_batches_early.py` etc: combine segments into batches for annotation
- `speech_selection/make_batches_mid.py` etc: combine segments into batches for annotation
- `speech_selection/make_batches_modern.py` etc: combine segments into batches for annotation

### Annotations

Raw annotations for tone and relevance are provided in online data files

To process the annotations:

- `annotations/tokenize.py`:  Collect all the annotated text segments and tokenize with spacy
- `annotations/export_for_label_aggregation.py`: Collect the annotations and export for aggregating labels (using label-aggregation)
- `annoations/measure_agreement.py` to measure agreement rates using Krippendorff's alpha
- Do label aggregation using label-aggregation repo (`github.com/dallascard/label-aggregation`) using Stan with the --no-vigilance option for both relevance and tone
- `relevance/make_relevance_splits.py`: Collect the tokenizations and estimated label probabilities, and make splits
- `relevance/make_relevance_splits.py` and `tone.make_tone_splits.py`: Divide the annotated data with inferred labels into train, dev, and test files for model training. For the latter, the additional annotations from MFC should be included using the `--extra-data-file` options, pointed to `data/annotations/relevance_and_tone/mfc/mfc_imm_tone.jsonlist`

### Training models

Run Roberta models on congressional annotations

- `classification/run_search_hf.py` to search of seeds (in order to estimate performance)
- `classification/run_final_model.py` to train a final model on all data with one seed
- `classification/make_predictions.py` to predict on keyword segments
- `classification/predict_on_all.py` to predict on all segments from each congress (exported from `parsing.rejoin_into_pieces_by_congress.py`)


### Collecting predictions

- use `relevance/collect_predictions.py` to get the relevant immigration speeches and segments
- use `tone/collect_predictions.py` to get the tones of these speeches and segments
- use `export/export_imm_segments_with_tone_and_metadata.py` to export the text, tone, and metadata
(some of the above depend on intermediate scripts, like `metadata.export_speeech_dates.py`)


### Identifying procedural speeches

- use `filtering/export_training_and_test.py` to export a heuristically labeled dataset of segments (procedural and not)
- use `filtering/export_short_speehces.py` to export short speeches to be classified
- train a model to identify procedural speeches using sklearn or equivalent
- use `filtering/collect_prediction.py` to gather up those speeches identified as procedural


### Additional Preprocessing

The following scripts are required for full replication:

- use `analysis/count_nouns.py` to count the nouns in the Congressional Record (for generating a random subset)
- use `analysis/choose_random_nouns.py` to get a random set of nouns not already used (for metaphor analysis)


### Analysis

Export some additional data based on speeches to simplify plotting:

- use `analysis/count_country_mentions.py` to identify frequently mentioned nationalities and relevance speeches
- use `export/export_imm_speeches_parsed.py` to collect and export the parsed versions of all immigration speeches
- use `analysis/identify_immigrant_mentions.py` to collect and export the mentions of immigrants and groups
- use `analysis/identify_group_mentions.py` to select the subset of mention sentences also mentioning each group
- use `analysis/count_tagged_lemmas.py` to collect counts
- use `analysis/count_speeches_and_tokens.py` to get background counts of non-procedural speeches

Measuring Impact:

- use `export/export_tone_for_lr_models.py` to export data for Logistic Regression classifiers
- train linear models with Frustratingly Easy Domain Adaptation (external repo)

Create contextual embeddings for masked terms and measuring dehumanization:

- use `embeddings/embed_immigrant_terms_masked.py` to get contextual embeddings for each mention
- use `embeddings/convert_embeddings_to_word_probs.py` to compute probabilities for each vector
- use `analysis/run_metaphorical_analysis.py` to compute metaphorical associations

Stan model (Appendix):

- use `stan/run_final_model.py` to run the Bayesian model with session, party, region, and chamber as factors

### Plots

If working with the processed data included in the Release, simply unzip the data.zip file in this directory, then run the following scripts:

- `analysis/count_county_mentions.py`
- `analysis/run_metaphorical_analysis.py`

The following scripts can be used to reproduce the main plots:

- use `plotting/make_tone_plots.py` to make all of the tone plots
- use `plotting/make_pmi_plots.py` to make all of the pmi plots
- use `plotting/make_metaphor_plots.py` to make the separate metaphor plots in the Appendix

To get the terms in table 1:
- use `export/export_imm_segments_for_linear.py` to export classified immigration segments to the appopriate format for the desired range of sessions
- use `linear/get_shap_values.py` to get the data in the right format


### Additional code for validation material in SI

For combining annotations (used for linear and CFM models in SI)
- `relevance/combine_relevance_data.py` (to combine all relevance data into one dataset and create a random test set)
- `tone/combine_tone_data.py` (to combine all relevance data into one dataset and create a random test set)
- `tone/filter_neutral.py` to filter out neutral speehces (for bianry model)

For running all linear models:
- `linear/create_partition.py` to convert dataset to proper format
- `linear/train.py` to train a model
- `linear/predict.py` or `linear/predict_on_all.py` to make predictions on other data
- `linear/export_weight.py` to export model weights

For linear model replication (in SI):
- train and predict using scripts in `linear`
- `relevance/collect_predictions_linear.py`
- `tone/collect_predictions_linear.py`
- use normal plotting scripts, pointing to new directories

For binary model replication (in SI):
- train and predict using scripts in `classification`
- `relevance/collect_predictions_val.py`
- `tone/collect_predictions_binary.py`
- `plotting/make_tone_plots_binary.py`

For CFM model replication (in SI):
- `tone/collect_predictions_cfm.py` to collect predictions and apply corrections
- not that this must be run three times, once with defaults, once with `--party-cfm D` and once with `--party-cfm R`
- use `plotting/make_tone_plots_probs_three.py` to put these all together

For leave-one-out plots and plots by individual speakers
- `plotting/make_tone_plots_loo.py`

For Frame comparison for Europe vs Latin America (in SI):
- `plotting/make_pmi_plots_latin_america.py`

For public opinion and SEI analyses (in SI), refer to `public_opinion_and_sei`

## Citation

To cite this respository or the data contained herein, please use:

Dallas Card, Serina Chang, Chris Becker, Julia Mendelsohn, Rob Voigt, Leah Boustan, Ran Abramitzky, and Dan Jurafsky. Replication code and data for "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration" [dataset] (2022). https://github.com/dallascard/us-immigration-speeches/

```
@article{card.2022.immdata,
  author = {Dallas Card and Serina Chang and Chris Becker and Julia Mendelsohn and Rob Voigt and Leah Boustan and Ran Abramitzky and Dan Jurafsky},
  title = {Replication code and data for ``{C}omputational analysis of 140 years of {US} political speeches reveals more positive but increasingly polarized framing of immigration'' [dataset]},
  year=2022,
  journal={https://github.com/dallascard/us-immigration-speeches/}
}
```
