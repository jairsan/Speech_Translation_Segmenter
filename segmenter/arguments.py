def add_train_arguments(parser):
    # Model parameters
    parser.add_argument("--train_corpus", type=str, help="Folder containing paired training images")
    parser.add_argument("--vocabulary", type=str, help="Folder containing paired training images")
    parser.add_argument("--epochs", type=int, help="Train the model this number of epochs")
    parser.add_argument("--batch_size", type=int, help="Train batch size")


def add_model_arguments(parser):
    # Model parameters
    parser.add_argument("--rnn_layer_size", type=int, help="Hidden size of the RNN layers")
    parser.add_argument("--classifier_hidden_size", type=int, help="Size of the classifier layers")
    parser.add_argument("--embedding_size", type=int, help="Size of the embedding")
    parser.add_argument("--n_classes", type=int, help="Number of classification targets")

