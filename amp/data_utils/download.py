import gzip
import urllib.request
import csv

from Bio import SeqIO

from amp.data_utils import sequence

TREMBL = "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/knowledgebase/uniprot_trembl.fasta.gz"
MIN_LENGTH = 0
MAX_LENGTH = 200

def download_uniprot(
        source=TREMBL,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        output=f'data/raw/Uniprot_{MIN_LENGTH}_{MAX_LENGTH}.csv',
    ):
    with urllib.request.urlopen(source) as f:
        with gzip.open(f, "rt") as input_file, open(output, 'w') as output_file:
            fieldnames = ['Name', 'Sequence']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
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
                if sequence.check_if_std_aa(seq) and sequence.check_length(seq, min_length, max_length):
                    writer.writerow({'Name': record.id, 'Sequence': seq})