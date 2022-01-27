import csv

from Bio import SeqIO

from amp.data_utils import sequence


def fasta2csv(fasta_input, csv_output):
    with open(fasta_input) as input_file, open(csv_output, 'w') as output_file:
        fieldnames = ['Name', 'Sequence']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        record_iter = SeqIO.parse(input_file, "fasta")
        record = True
        while record:
            try:
                record = next(record_iter)
            except StopIteration:
                record = None
            if record is None:
                # End of file
                break
            seq = str(record.seq)
            if sequence.check_if_std_aa(seq):
                writer.writerow({'Name': record.id, 'Sequence': seq})
