def add_audio_train_arguments(parser):
    parser.add_argument("--frozen_text_model_path", type=str, help="Load this text model to extract features", required=True)
    parser.add_argument("--train_audio_features_corpus", type=str, help="Training audio features file", required=True)
    parser.add_argument("--dev_audio_features_corpus", type=str, help="Dev audio features file", required=True)


def add_ff_arguments(parser):
    parser.add_argument("--feedforward_layers", type=int, default=2, help="Number of feedforward layers (Only for ff architecture)")
    parser.add_argument("--feedforward_size", type=int, default=128, help="Size of feedforward layers (Only for ff architecture)")


def add_rnn_arguments(parser):
    parser.add_argument("--rnn_layer_size", type=int, default=256, help="Hidden size of the RNN layers")
    parser.add_argument("--embedding_size", type=int, default=256, help="Size of the embedding")

def add_common_arguments(parser):
    parser.add_argument("--dropout", type=float, default=0.3)
