import numpy as np
import pandas as pd
from amp.data_utils import sequence

def translate_generated_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el - 1] if el != 0 else "" for el in encoded_peptide[0].argmax(axis=1)])

def translate_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el-1] if el != 0 else "" for el in encoded_peptide])

def generate_unconstrained(
        n: int,
        improve: bool,
        pca: bool,
        selection: bool,
        decoder,
        classifier,
        regressor,
        pca_decomposer,
        latent_dim: int = 64,

):
    generated = []
    amp_input = 1 if improve else 0
    mic_input = 0 if improve else 1
    class_min, class_max = (0.8, 1.0) if improve else (0.0, 0.2)
    mic_min, mic_max = (0, 0.25) if improve else (0.75, 1)

    counter = 0
    while len(generated) < n:
        counter += 1
        z = np.random.normal(size=(1, latent_dim))
        if pca:
            z = pca_decomposer.inverse_transform(z)
        z_cond = np.concatenate([z, np.array([[amp_input]]), np.array([[mic_input]])], axis=1)
        decoded = decoder.predict(z_cond)
        peptide = translate_generated_peptide(decoded)
        peptide = peptide.strip("'")
        if "'" in peptide:
            continue

        class_prediction = classifier.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]
        mic_prediction = regressor.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]

        if selection:
            if not class_min <= class_prediction <= class_max:
                continue
            if not mic_min <= mic_prediction <= mic_max:
                continue

        if (peptide, class_prediction, mic_prediction) not in generated:
            generated.append((peptide, class_prediction, mic_prediction))

    print(f'Generated {counter} peptides, {n} passed')
    return generated


def generate_batch_modify(
        input_sequences,
        classifier_model,
        regressor_model,
        encoder_model,
        decoder_model,
        improve:bool,
):
    amp_input = 1 if improve else 0
    mic_input = 0 if improve else 1
    class_min, class_max = (0.8, 1.0) if improve else (0.0, 0.2)

    changed = []
    originals = []
    original_raw = input_sequences
    original_padded = sequence.pad(sequence.to_one_hot(original_raw))
    original_encoded = encoder_model.predict(original_padded)
    original_probs = classifier_model.predict(original_padded)
    original_mics = regressor_model.predict(original_padded)
    for z, org_peptide, org_prob, org_mic in zip(original_encoded, original_raw, original_probs, original_mics):
        z_cond = np.concatenate([[z], np.array([[amp_input]]), np.array([[mic_input]])], axis=1)
        decoded = decoder_model.predict(z_cond)
        peptide = translate_generated_peptide(decoded)
        peptide = peptide.strip("'")
        if "'" in peptide or peptide == '':
            continue
        if peptide == org_peptide:
            continue

        class_prediction = classifier_model.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]
        mic_prediction = regressor_model.predict(np.array([decoded[0].argmax(axis=1)]))[0][0]

        if class_min <= class_prediction <= class_max:
            if improve:
                if mic_prediction < org_mic[0]:
                    changed.append((peptide, class_prediction, mic_prediction))
                    originals.append((org_peptide, org_prob[0], org_mic[0]))
            else:
                if mic_prediction > org_mic[0]:
                    changed.append((peptide, class_prediction, mic_prediction))
                    originals.append((org_peptide, org_prob[0], org_mic[0]))

    if improve:
        print(f'Improved {len(changed)} peptides, {len(input_sequences)-len(changed)} unchanged')
    else:
        print(f'Worsened {len(changed)} peptides, {len(input_sequences)-len(changed)} unchanged')

    return originals, changed


def generate_template(
        template,
        improve:bool,
        classifier_model,
        regressor_model,
        encoder_model,
        decoder_model,
        n_attempts,

):
    amp_input = 1 if improve else 0
    mic_input = 0 if improve else 1
    class_min, class_max = (0.8, 1.0) if improve else (0.0, 0.2)

    sequence = []
    amp = []
    mic = []

    org_amp = classifier_model.predict(template.reshape(1, 25))
    org_mic = regressor_model.predict(template.reshape(1, 25))
    org_z = encoder_model.predict(template.reshape(1, 25))

    sequence.append('ORIGINAL_PEPTIDE:' + translate_peptide(template))
    amp.append(org_amp[0][0])
    mic.append(org_mic[0][0])

    for i in range(n_attempts):
        for mean in [0, 0.01, 0.1]:
            for std in [0, 0.01, 0.1]:
                z = org_z + np.random.normal(mean, std, org_z.shape[0])
                z_cond = np.concatenate([z, np.array([[amp_input]]), np.array([[mic_input]])], axis=1)
                candidate = decoder_model.predict(z_cond)
                class_prediction = classifier_model.predict(np.array([candidate[0].argmax(axis=1)]))
                mic_prediction = regressor_model.predict(np.array([candidate[0].argmax(axis=1)]))

                if translate_generated_peptide(candidate) in sequence:
                    continue
                if translate_generated_peptide(candidate) == translate_peptide(template):
                    continue
                if not class_min <= class_prediction[0][0] <= class_max:
                    continue
                else:
                    if improve:
                        if mic_prediction < org_mic[0]:
                            sequence.append(translate_generated_peptide(candidate))
                            amp.append(class_prediction[0][0])
                            mic.append(mic_prediction[0][0])
                    else:
                        if mic_prediction > org_mic[0]:
                            sequence.append(translate_generated_peptide(candidate))
                            amp.append(class_prediction[0][0])
                            mic.append(mic_prediction[0][0])


    df = pd.DataFrame.from_dict(
        {
            'sequence': sequence,
            'amp_prob': amp,
            'mic_pred': mic,
        }
    )

    print(f'Generated {len(sequence)} peptides based on template,'
          f' {int(n_attempts)*3*3} attempts')

    return df