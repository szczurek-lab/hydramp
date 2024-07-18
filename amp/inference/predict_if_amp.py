from amp.config import MAX_LENGTH, MIN_LENGTH
from amp.data_utils.data_loader_single_file import AMPDataManagerSingleFile
from amp.data_utils.fasta import fasta2csv
from amp.utils import basic_model_serializer

# model loading
bms = basic_model_serializer.BasicModelSerializer()
amp_classfier = bms.load_model('models/amp_classifier')
model = amp_classfier()

input_csv_path = 'data/AMPlify_AMP_test_common_20.csv'
fasta2csv('data/AMPlify_AMP_test_common_20.fasta', input_csv_path)

data_manager_single_file = AMPDataManagerSingleFile(
    input_csv_path,
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)
x_from_file, skipped = data_manager_single_file.get_data()

preds = model.predict(x_from_file)
print(preds, skipped)