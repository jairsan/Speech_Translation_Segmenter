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
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n updates")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model every n epochs")

def add_model_arguments(parser):
    # Model parameters
    parser.add_argument("--rnn_layer_size", type=int, help="Hidden size of the RNN layers")
    parser.add_argument("--classifier_hidden_size", type=int, help="Size of the classifier layers")
    parser.add_argument("--embedding_size", type=int, help="Size of the embedding")
    parser.add_argument("--n_classes", type=int, help="Number of classification targets")


def add_infer_arguments(parser):
    parser.add_argument("--vocabulary", type=str, help="Vocabulary to be used by the network", required=True)
    parser.add_argument("--model_path", type=str, help="Load this model", required=True)
    parser.add_argument("--file", type=str, help="Infer from this sample file instead of a stream")

