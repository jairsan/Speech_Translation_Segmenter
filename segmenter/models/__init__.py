from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_text_model import RNNFFTextModel
from segmenter.models.rnn_ff_audio_text_model import RNNFFAudioTextModel
from segmenter.models.rnn_ff_audio_text_feas_copy_model import RNNFFAudioTextFeasCopyModel

STR_TO_CLASS = {
    SimpleRNNTextModel.name: SimpleRNNTextModel,
    RNNFFTextModel.name: RNNFFTextModel,
    RNNFFAudioTextModel.name: RNNFFAudioTextModel,
    RNNFFAudioTextFeasCopyModel.name: RNNFFAudioTextFeasCopyModel
}
