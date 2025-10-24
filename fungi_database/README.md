README

Preprocessing data step is based on data download from GlobalFungi Database, as published by Větrovský et al (2020; doi.org/10.1038/s41597-020-0567-7). 

##Fungal Community Prediction Model
This model is created to train a pipeline for the prediction of dominant fungi in a community based on soil, plant, and ecosystem data.

##Features
- Preprocessing scripts for cleaning and standardizing fungal abundance and metadata
- Training pipeline for RandomForest classification of dominant fungi
- Column transformers to handle numeric and categorical features
- Evaluation scripts with classification reports

##Installation
-
1. Clone the repository - 
	Download the code to your local machine:

	git clone https://github.com/yourusername/fungi_model_project.git
	cd fungi_model_project
2. Create a Python environment - 
	# Using venv
		python -m venv env
	
Activate the environment
	#Windows
		env\Scripts\activate
	#macOS / Linux
		source env/bin/activate
3. Install required packages - 
	
	pip install -r requirements.txt
	*Python 3.10.xx+ recommended. 
4. Preprocess raw data - (optional)

If you have data that needs preprocessing (abundance values are required to be numeric):
	Run the preprocessing script to clean it-
	
	python preprocessing.py
This produces a clean script to train model on or use for prediction.

5. Train the model - (optional)

If you have data you would like to retrain the model on:

	python train_classifier.py
This will train the RandomForest pipeline, evaluate it on test split, and save the trained pipeline as trained_fungi_model.pkl. This step can be skipped if wanting to use pre-trained model. Training data for pre-trained model can be found in /data.

6. Using the pre-trained model - 

The predict.py script will allow for sample data to predict the dominant species of fungi in an ecosystem without training.

	python predict.py data/new_samples.csv --output predictions.csv
-data/new_samples.csv is your new data
-predictions.csv will contain prediction data for the input dataset

7. Notes - 

The pretrained pipeline contains the preprocessing step.
Ensure your data has the same metadata columns required for efficient predictions. The greater quantity of missing columns, the less accurate the output. When additional columns not present in training data are included, Python warnings will be present, but these are not significant.

##Citation
Větrovský, T., Morais, D., Kohout, P. et al. GlobalFungi, a global database of fungal occurrences from high-throughput-sequencing metabarcoding studies. Sci Data 7, 228 (2020). https://doi.org/10.1038/s41597-020-0567-7

## License

This project is licensed under the [MIT License](LICENSE).

