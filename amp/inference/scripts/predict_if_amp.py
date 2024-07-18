import argparse
from amp.data_utils.fasta import fasta2csv
from amp.data_utils.data_loader_single_file import AMPDataManagerSingleFile
from amp.utils import basic_model_serializer
from amp.config import MIN_LENGTH, MAX_LENGTH
import pandas as pd


def predict_and_save(model_path, sequence_path, sequence_format, output_csv):
    bms = basic_model_serializer.BasicModelSerializer()
    model = bms.load_model(model_path)()

    if sequence_format.lower() == 'fasta':
        input_csv_path = sequence_path.replace('.fasta', '.csv')
        fasta2csv(sequence_path, input_csv_path)
    elif sequence_format.lower() == 'csv':
        input_csv_path = sequence_path
    else:
        raise ValueError("Unsupported sequence format. Use 'fasta' or 'csv'.")

    data_manager = AMPDataManagerSingleFile(input_csv_path, min_len=MIN_LENGTH, max_len=MAX_LENGTH)
    x_from_file, skipped = data_manager.get_data()

    preds = model.predict(x_from_file)

    pd.DataFrame({'Prediction': preds[:, 0]}).to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sequences and save to CSV.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--sequence_path", type=str, required=True, help="Path to the sequence file.")
    parser.add_argument("--format", type=str, choices=['fasta', 'csv'], required=True,
                        help="Format of the sequence file.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Name of the output CSV file to save predictions.")

    args = parser.parse_args()

    predict_and_save(args.model_path, args.sequence_path, args.format, args.output_csv)
