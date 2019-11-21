def add_train_arguments(parser):
    # Model parameters
    parser.add_argument("--train_corpus", type=str, help="Training text file", required=True)
    parser.add_argument("--dev_corpus", type=str, help="Dev text file", required=True)
    parser.add_argument("--output_folder", type=str, help="Save model training here", required=True)
    parser.add_argument("--vocabulary", type=str, help="Vocabulary to be used by the network", required=True)
    parser.add_argument("--epochs", type=int, help="Train the model this number of epochs",default=50)
    parser.add_argument("--batch_size", type=int, help="Train batch size", default=64)
    parser.add_argument("--split_weight", type=float, help="Weight given to split samples "
                                                           "(Class 1). No split (Class 0) weight is 1)", default=1)
    parser.add_argument("--upsample_split", type=int, help="Upsample split samples by this factor", default=1)
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n updates")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model every n epochs")
    parser.add_argument("--optimizer", choices=['sgd', 'adam'], default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_schedule", choices=['fixed', 'reduce_on_plateau'], default="fixed")
    parser.add_argument("--lr_reduce_patience", type=int, default=5)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)





def add_model_arguments(parser):
    # Model parameters
    parser.add_argument("--rnn_layer_size", type=int, help="Hidden size of the RNN layers")
    # TODO: Remove this argument, does nothing for now
    parser.add_argument("--classifier_hidden_size", type=int, help="Size of the classifier layers")
    parser.add_argument("--embedding_size", type=int, help="Size of the embedding")
    parser.add_argument("--n_classes", type=int, help="Number of classification targets")


# TODO: Fix stream so that it only infers from stdin.
def add_infer_arguments(parser):
    parser.add_argument("--vocabulary", type=str, help="Vocabulary to be used by the network", required=True)
    parser.add_argument("--model_path", type=str, help="Load this model", required=True)
    parser.add_argument("--input_format",  choices=['sample_file', 'stream', 'list_of_text_files'], help="Input format for the decoding process. ", required=True)
    parser.add_argument("--file", type=str, help="Infer from this sample file")
    parser.add_argument("--stream", type=str, help="Infer from this stream. By default, sys.stdin", default="sys.stdin")
    parser.add_argument("--input_file_list", type=str, help="Infer from list of files.")


