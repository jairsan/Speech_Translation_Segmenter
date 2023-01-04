from segmenter.models import STR_TO_CLASS
from segmenter.models.rnn_ff_text_model import RNNFFTextModel


def add_train_arguments(parser):
    # Model parameters
    parser.add_argument("--train_corpus", type=str, help="Training text file", required=True)
    parser.add_argument("--dev_corpus", type=str, help="Dev text file", required=True)
    parser.add_argument("--output_folder", type=str, help="Save model training here", required=True)
    parser.add_argument("--vocabulary", type=str, help="Vocabulary to be used by the network", required=True)
    parser.add_argument("--vocabulary_max_size", type=int, help="Use up to this number of vocabulary words", default=None)
    parser.add_argument("--epochs", type=int, help="Train the model this number of epochs", default=40)
    parser.add_argument("--batch_size", type=int, help="Train batch size", default=256)
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Perform gradiend update every [gradient_accumulation] batches")
    parser.add_argument("--sampling_temperature", type=float, default=1, help="Sampling temperature used for multiclass problems")
    parser.add_argument("--samples_per_random_epoch", type=int, help="If using upsampling, how many samples we wish to draw per epoch. If -1, draw all", default=-1)
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n updates")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model every [checkpoint_interval] epochs")
    parser.add_argument("--checkpoint_every_n_updates", type=int, help="Save and eval model every [checkpoint_every_n_updates] updates have been processed")
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_schedule", choices=['fixed', 'reduce_on_plateau'], default="reduce_on_plateau")
    parser.add_argument("--lr_reduce_patience", type=int, default=10)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.9999)
    parser.add_argument("--adam_eps", type=float, default=1e-08)
    parser.add_argument("--seed", type=int, help="Random seed", default=1)


def add_general_arguments(parser):
    # Model parameters
    parser.add_argument("--model_architecture", type=str, choices=list(STR_TO_CLASS.keys()), default=RNNFFTextModel.name)
    parser.add_argument("--transformer_model_name", type=str, default=None)
    parser.add_argument("--sample_max_len", type=int, help="Total number of tokens on each sample.")
    parser.add_argument("--sample_window_size", type=int, help="Number of tokens in the future window of each sample. "
                                                               "A sample with max_len=l and window_size=s "
                                                               "has (l - s - 1) previous history tokens. ")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classification targets")


def add_infer_arguments(parser):
    parser.add_argument("--model_path", type=str, help="Load this model", required=True)
    parser.add_argument("--beam", type=int, help="Use beam search, with beam of size b", default=1 )
    parser.add_argument("--input_format",  choices=['sample_file', 'list_of_text_files'], help="Input format for the decoding process. ", required=True)
    parser.add_argument("--file", type=str, help="Infer from this sample file")
    parser.add_argument("--input_file_list", type=str, help="Infer from list of files.")
    parser.add_argument("--input_audio_file_list", type=str, help="Infer from list of files containing audio features.")
    parser.add_argument("--debug", action='store_true', help="Activate debug mode")
    parser.add_argument("--segment_max_size", type=int, default=99999, help="Force split for segments longer than [segment_max_size]")

