# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier implemented using scikit-learn. It works by building groups of decision trees and aggregating their predictions to improve generalization and reduce variance.

## Intended Use
The goal of this model is to predict weather an individual's income exceedas $50K per year based on U.S Census data. This can be used for research, education, or exploratory analytics.

## Training Data
The dataset used for model training is the UCI Adult Census Income dataset. I contains categorical and numerical features including: "age", "workclass", "occupation", "education", "race", "marital-status", "relationship", "native-country, and "sex". Categorical features were one-hot encoded for training. The training set cosists of 80% of the full data, randomly sampled.

## Evaluation Data
The ramaining 20% of the data was used as the test set. Both training and test sets were preprocessed using a consistent pipeline. Evaluation metrics were computed `scikit-learn`'s built in metrics functions.

## Metrics
The model was evaluated on the test set using:
- Precision: 0.8008
- Recall: 0.5372
- F1 Score: 0.6430

Model performance was also evaluated by slices based on categorical features. These slices metrics are registered in the `slice_output.txt` file. This shows how the model performs accross different subgroups.

## Ethical Considerations
There are mesurable imbalances in model performance accross feature slices. For example:
- Individuals with lower formal education levels, 7th - 8th grade, had near 0 F1 scores, while those with Doctorate scored 0.82. This may suggest that the model struggles to predict outcomes for individuals with lower education. This result may point to an underlying social or data biases. 
- Females had an F1 Score of 0.5133 while males scored 0.6623. The small gap can indicate that the model works better for males. This can indicate the need for an improvement in fairness.

## Caveats and Recommendations
- Some data slices (e.g., rare native countries) have very small sample sizes and show artificially high performance. These results should not be considered reliable.
- Subgroup performance varies significantly; this model should not yet be used in high-stakes or fairness-sensitive applications.