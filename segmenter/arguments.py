def add_train_arguments(parser):
    # Model parameters
    parser.add_argument("--train_corpus", type=str, help="Training text file", required=True)
    parser.add_argument("--dev_corpus", type=str, help="Dev text file", required=True)
    parser.add_argument("--output_folder", type=str, help="Save model training here", required=True)
    parser.add_argument("--vocabulary", type=str, help="Vocabulary to be used by the network", required=True)
    parser.add_argument("--classes_vocabulary", type=str, help="Dictionary of <class> <count> used for multiclass problems")
    parser.add_argument("--epochs", type=int, help="Train the model this number of epochs",default=40)
    parser.add_argument("--batch_size", type=int, help="Train batch size", default=256)
    parser.add_argument("--split_weight", type=float, help="Weight given to split samples "
                                                           "(Class 1). No split (Class 0) weight is 1)", default=1)
    parser.add_argument("--min_split_samples_batch_ratio", type=float, help="Upsample split samples (class 1) so that each"
                                                                          "batch contains, on average, this ratio of split vs non-split", default=0.3)
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Perform gradiend update every [gradient_accumulation] batches")
    parser.add_argument("--sampling_temperature", type=float, default=1, help="Sampling temperature used for multiclass problems")
    parser.add_argument("--samples_per_random_epoch", type=int, help="If using upsampling, how many samples we wish to draw per epoch. If -1, draw all", default=-1)
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n updates")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model every [checkpoint_interval] epochs")
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default="adam")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_schedule", choices=['fixed', 'reduce_on_plateau'], default="reduce_on_plateau")
    parser.add_argument("--lr_reduce_patience", type=int, default=10)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.9999)
    parser.add_argument("--adam_eps", type=float, default=1e-08)
    parser.add_argument("--unk_noise_prob", type=float, default=0.0, help = "Every word in a sample will have probability p to be replaced by \"<unk>\""
                                                                            "(Noise is applied every time a sample is drawn)" )
    parser.add_argument("--seed", type=int, help="Random seed", default=1)
    parser.add_argument("--train_audio_feas_corpus", type=str, help="Training containing audio feas")
    parser.add_argument("--dev_audio_feas_corpus", type=str, help="Dev containing audio feas")

def add_audio_train_arguments(parser):
    parser.add_argument("--model_path", type=str, help="Load this model", required=True)
    parser.add_argument("--train_audio_features_corpus", type=str, help="Training audio features file", required=True)
    parser.add_argument("--dev_audio_features_corpus", type=str, help="Dev audio features file", required=True)

def add_model_arguments(parser):
    # Model parameters
    parser.add_argument("--rnn_layer_size", type=int, default=256, help="Hidden size of the RNN layers")
    parser.add_argument("--audio_rnn_layer_size", type=int, default=8, help="Hidden size of the RNN layers for the audio encoder")
    parser.add_argument("--embedding_size", type=int, default=256, help="Size of the embedding")
    parser.add_argument("--feedforward_layers", type=int, default=2, help="Number of feedforward layers (Only for ff architecture)")
    parser.add_argument("--feedforward_size", type=int, default=128, help="Size of feedforward layers (Only for ff architecture)")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classification targets")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_architecture", type=str, choices = ["simple-text", "ff-text", "ff-audio-text",
                                                                     "ff-audio-text-copy-feas"], default = "ff-text",
                        )
    parser.add_argument("--transformer_architecture", type=str, default=None)
    parser.add_argument("--sample_max_len", type=int, help="Total number of tokens on each sample.")
    parser.add_argument("--sample_window_size", type=int, help="Number of tokens in the future window of each sample. "
                                                               "A sample with max_len=l and window_size=s "
                                                               "has (l - s - 1) previous history tokens. ")

def add_multiclass_infer_arguments(parser):
    parser.add_argument("--return_type", choices=['casing', 'words'], default="words")

def add_infer_arguments(parser):
    parser.add_argument("--model_path", type=str, help="Load this model", required=True)
    parser.add_argument("--beam", type=int, help="Use beam search, with beam of size b", default=1 )
    parser.add_argument("--input_format",  choices=['sample_file', 'list_of_text_files'], help="Input format for the decoding process. ", required=True)
    parser.add_argument("--file", type=str, help="Infer from this sample file")
    parser.add_argument("--input_file_list", type=str, help="Infer from list of files.")
    parser.add_argument("--input_audio_file_list", type=str, help="Infer from list of files containing audio features.")
    parser.add_argument("--debug", action='store_true', help="Activate debug mode")
    parser.add_argument("--segment_max_size", type=int, default=99999, help="Force split for segments longer than [segment_max_size]")

