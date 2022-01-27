from amp.models.discriminators import veltri_amp_classifier
from amp.models.discriminators import amp_classifier_noCONV

from amp.models.decoders import amp_expanded_decoder
from amp.models.encoders import amp_expanded_encoder
from amp.models.master import master

MODEL_GAREDN = {
    'VeltriAMPClassifier': veltri_amp_classifier.VeltriAMPClassifier,
    'NoConvAMPClassifier': amp_classifier_noCONV.NoConvAMPClassifier,
    'AMPExpandedDecoder': amp_expanded_decoder.AMPDecoder,
    'AMPExpandedEncoder': amp_expanded_encoder.AMPEncoder,
    'MasterAMPTrainer': master.MasterAMPTrainer,
}
