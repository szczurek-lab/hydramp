# HYDRAmp - cvae++ approach to generate new amps

### How to generate peptides with HYDRAmp

First, install package via

```console
pip install .
```

interface for generating peptides is enclosed in `HYDRAmpGenerator` class. Latest model can be found under `models/HydrAMP/37`  and latest decomposer can be found under `models/HydrAMP/pca_decomposer.joblib`. Initialize object with path to model and decomposer

```python
from amp.inference import HYDRAmpGenerator

generator = HYDRAmpGenerator(model_path, decomposer_path)
```

The package provides 2 modes for peptide generation:

- **template generation** - allows to change the properties of existing sequences (it may try to make antimicrobial peptides non-amp, non-antimicrobial peptides amp) 
```python
   def template_generation(self, sequences: List[str],
                            constraint: Literal['relative', 'absolute'] = 'absolute',
                            mode: Literal['improve', 'worsen'] = 'improve',
                            n_attempts: int = 100, temp=5, **kwargs) -> Dict[Any, Dict[str, Any]]:
```
>*Parameters:*
>
>`sequences` - list of peptide sequences to process
>
> `constraint` - 'relative' if generated peptides should be strictly better than input sequences (higher
        P(AMP), lower P(MIC) in case of positive generation; lower P(AMP), higher P(MIC) in case on negative generation)
        'absolute' if generated sequences should be good enough but not strictly better
>
> `mode` - "*improve*" or "*worsen*",
>
> `n_attempt` - how many times a single latent vector is decoded 
>
> `seed` - seed for reproducible results
> 
> `**kwargs` - additional boolean arguments for filtering. See [filtering options](#filtering-options)
>
> `temp` - creativity parameter. Controls latent vector sigma scaling
>
> *Returns:*
>
> `res` - dict of dicts, keys corresponds to original sequences, values are dicts with the following fields:
> - **mic** - probability of low mic
> - **amp** - probability of high amp
> - **length** - length of sequence
> - **hydrophobicity**
> - **hydrophobic_moment**
> - **charge**
> - **isoelectric_point**
> - **h-score - [see H-score description](#h-score)**
>
> and 
> - **generated_sequences** - list of generated and filtered sequences, each described with a dict as above
```python
# exemplary method call for Pexiganan and Temporin A (number of generated sequences was trunkated)
>> generator.template_generation(sequences=['GIGKFLKKAKKFGKAFVKILKK' 'FLPLIGRVFSGIL'],
                              mode="improve",
                              constraint='absolute',
                              n_attempts=100)

{'FLPLIGRVLSGIL': {'amp': 0.9999972581863403,
                   'charge': 1.996,
                   'generated_sequences': [{'amp': 0.9998811483383179,
                                            'charge': 1.996,
                                            'hydrophobic_moment': 0.5703602534454862,
                                            'hydrophobicity': 0.6307692307692307,
                                            'isoelectric_point': 10.7421875,
                                            'length': 13,
                                            'mic': 0.8635282516479492,
                                            'sequence': 'FLPLIGRVLPGIL'}],
                   'hydrophobic_moment': 0.5872424608996596,
                   'hydrophobicity': 0.6076923076923078,
                   'isoelectric_point': 10.7421875,
                   'length': 13,
                   'mic': 0.936310887336731},
 'GIGKFLKKAKKFGKAFVKILKK': {'amp': 1.0,
                            'charge': 9.994,
                            'generated_sequences': [{'amp': 1.0,
                                                     'charge': 9.994,
                                                     'hydrophobic_moment': 0.7088747699407729,
                                                     'hydrophobicity': -0.05090909090909091,
                                                     'isoelectric_point': 11.576171875,
                                                     'length': 22,
                                                     'mic': 0.011877596378326416,
                                                     'sequence': 'GIGKFLKKAKKFGKAFVKILKK'},
                                                    {'amp': 1.0,
                                                     'charge': 8.994,
                                                     'hydrophobic_moment': 0.6008331010319634,
                                                     'hydrophobicity': 0.07181818181818182,
                                                     'isoelectric_point': 11.5185546875,
                                                     'length': 22,
                                                     'mic': 0.010871708393096924,
                                                     'sequence': 'GIGKFLKKAKFLGKAFVKIFKK'}],
                            'hydrophobic_moment': 0.7088747699407729,
                            'hydrophobicity': -0.05090909090909091,
                            'isoelectric_point': 11.576171875,
                            'length': 22,
                            'mic': 0.01187763549387455}}

```

- **unconstrained generation** - allows to sample new peptides from encoded latent space
```python    
def unconstrained_generation(self,
                             mode: Literal["amp", "nonamp"] = 'amp',
                             n_target: int = 100,
                             seed: int = None,
                             filter_out: bool = True,
                             properties: bool = True,
                             n_attempts: int = 64,
                             **kwargs) -> Union[List[Dict[str, Any]], List[str]]:
```


> *Parameters:* 
>
>`mode` - str, "*amp*" or "*nonamp*", default: "*amp*"
>
> `n_target` - how many peptides should be succesfully sampled
>
> `seed` - seed for reproducible results
>
> `filter_out` - uses AMP and MIC classifier information to filter sequences that were predicted to not be from desired class
> 
> `properties` - if True, each sequence  is a dictionary with additional properties
>
> `n_attempt` - how many times a single latent vector is decoded 
>
> `**kwargs` - additional boolean arguments for filtering. See [filtering options](#filtering-options)
>*Returns:*
>
> `res` - list of dicts, each dict correspond to single sequence described with same fields as in template generation. 
> 

```python
# exemplary method call, request 2 amp sequences without properties

>> generator.unconstrained_generation(n_target=2, mode="amp", properties=False)

['FRMSLAWKCLLL', 'GLLGGLLKRRRFVR']

# exemplary method call, request 2 amp sequences with properties

>> generator.unconstrained_generation(n_target=2, mode="amp", properties=True)

[{'amp': 0.9997017979621887,
  'charge': 2.928,
  'h_score': 1.4285714285714284,
  'hydrophobic_moment': 0.1713221131786216,
  'hydrophobicity': 0.31500000000000006,
  'isoelectric_point': 10.03125,
  'length': 12,
  'mic': 0.017575599253177643,
  'sequence': 'FRMSLAWKCLLL'},
 {'amp': 0.9999890327453613,
  'charge': 5.996,
  'h_score': 1.1794871794871795,
  'hydrophobic_moment': 0.2868787441416022,
  'hydrophobicity': -0.24000000000000002,
  'isoelectric_point': 12.58447265625,
  'length': 14,
  'mic': 0.06286950409412384,
  'sequence': 'GLLGGLLKRRRFVR'}]

# call with filtering

>> generator.unconstrained_generation(n_target=2, mode="amp", properties=False, filter_positive_clusters=True)


```

### Filtering options

The following are additional filtering:

- **filter_positive_clusters** -  do not allow 3 hydrophobic aminoacids in a row
- **filter_repetitive_clusters** - do not allow the same aminoacid repetead 3 times in a 5-sized window
- **filter_cysteins** - filter peptides with cysteins
- **filter_known_amps** - remove all generated peptides that were found in databases

### H-score

TODO

