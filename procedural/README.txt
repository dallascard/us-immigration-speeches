
Scripts in this directory are related to classifying speeches as procedural or not
It also depends on an external directory.

Step 1: export_training_and_test.py to export train and test data (plus very short speech ids)
Step 2: export_short_speeches.py: export all short speeches (for prediction)
Step 3: train a classifier using guac
Step 4: apply classifier to all short speeches
Step 5: collect_predictions.py:  combine the very short speech ids and the predicted procedural speeches
