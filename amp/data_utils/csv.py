import csv

from sklearn.model_selection import train_test_split

from amp.data_utils import sequence

MIN_LENGTH = 0
INPUT_PATH = 'data/interim/Uniprot_0_200_no_duplicates.csv'


class CSVSplitter:
    """Perform splitting csv file in training, test and validation subsets."""

    def __init__(
            self,
            input_file,
            min_length,
            max_length,
            output_to='data/processed',
            test_size: int=0.1,
            val_size: int=0.1,
    ):
        self.input_file = input_file
        self.name = f'{output_to}/Uniprot_{min_length}_{max_length}'
        self.test_size = test_size + val_size
        self.val_size = val_size / self.test_size

    def get_row_count(self):

        with open(self.input_file) as f:
            return sum(1 for line in f)

    def get_indices(self):
        row_count = self.get_row_count()
        partition = {}

        indices_train, indices_test = train_test_split(
            range(1, row_count + 1),
            test_size=self.test_size,
            random_state=1
        )
        indices_test, indices_val = train_test_split(
            indices_test,
            test_size=self.val_size,
            random_state=1
        )
        for i in indices_train:
            partition[i] = 'train'
        for i in indices_test:
            partition[i] = 'test'
        for i in indices_val:
            partition[i] = 'val'

        # partition['train'] = indices_train
        # partition['test'] = indices_test
        # partition['val'] = indices_val

        return partition

    def csv_partition(self, indices):

        train_file = f'{self.name}_train.csv'
        test_file = f'{self.name}_test.csv'
        val_file = f'{self.name}_val.csv'

        with open(self.input_file) as csv_input, \
                open(train_file, 'w') as csv_train, \
                open(test_file, 'w') as csv_test, \
                open(val_file, 'w') as csv_val:

            reader = csv.reader(csv_input)

            train_writer = csv.writer(csv_train)
            test_writer = csv.writer(csv_test)
            val_writer = csv.writer(csv_val)

            train_writer.writerow(['Name', 'Sequence'])
            test_writer.writerow(['Name', 'Sequence'])
            val_writer.writerow(['Name', 'Sequence'])

            for line_number, row in enumerate(reader):
                if line_number == 0:
                    continue
                if indices[line_number] == 'train':
                    train_writer.writerow(row)
                elif indices[line_number] == 'test':
                    test_writer.writerow(row)
                else:
                    val_writer.writerow(row)

    def split(self):

        partition = self.get_indices()
        self.csv_partition(partition)


def get_various_lengths_csv(
        max_lengths: list,
        min_length:int =MIN_LENGTH,
        input_path= INPUT_PATH,
        output_to='data/interim',
):
    list_of_outputs = [f'{output_to}/Uniprot_{min_length}_{max_len}.csv' for max_len in max_lengths]

    with open(input_path) as input_file:
        reader = csv.reader(input_file)
        writers = []
        output_files = []
        for file_name in list_of_outputs:
            file = open(file_name, 'w')
            output_files.append(file)
            writers.append(csv.writer(file))

        # Write headers in each file
        for writer in writers:
            writer.writerow(['Name', 'Sequence'])

        for line_number, row in enumerate(reader):
            if line_number == 0:
                continue
            seq = row[1]
            for writer, max_len in zip(writers, max_lengths):
                if sequence.check_length(seq, min_length, max_len):
                    writer.writerow(row)

        for file in output_files:
            file.close()