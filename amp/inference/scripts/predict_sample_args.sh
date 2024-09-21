# amp classifier
python -m amp.inference.scripts.predict_if_amp --model_path models/amp_classifier/ \
--sequence_path data/AMPlify_AMP_test_common_20.fasta --format fasta --output_csv data/out.csv

# mic classifier (binary predictions based on MIC)
python -m amp.inference.scripts.predict_if_amp --model_path models/mic_classifier/ \
--sequence_path data/AMPlify_AMP_test_common_20.fasta --format fasta --output_csv data/out.csv
