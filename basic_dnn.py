import os
import itertools
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.utils import gen_batches, check_random_state

from keras.utils import to_categorical, plot_model


def parse_cmdline_args():
    parser = ArgumentParser(description="Bunga bunga")
    parser.add_argument("--data_dir", type=str,
                        default="/home/elvis/datasets/pac_2018",
                        help="directory containing PAC 2018 data")
    parser.add_argument("--hidden_dims", type=list, nargs="+", default=[8],
                        help="hidden layer sizes")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs to run for training")
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="fraction of training data to use for training")
    return parser.parse_args()


def data_gen(indices, batch_size=100, random_state=None):
    """
    Smart generator of data batches
    """
    rng = check_random_state(random_state)
    n_indices = len(indices)
    for batch in gen_batches(n_indices, batch_size):
        batch_indices = indices[batch]
        rng.shuffle(batch_indices)
        x_batch = X[batch_indices]
        df_batch = df_.iloc[batch_indices]
        y_batch = {}
        for target_name in targets_setup:
            y_batch[target_name] = np.asarray(df_batch[target_name].tolist())
            if target_name in ["Label", "Gender"]:
                y_batch[target_name] = to_categorical(y_batch[target_name] - 1,
                                                      num_classes=2)

        yield x_batch, y_batch


def output_of_expand_dims(input_shape):
        return tuple(list(input_shape) + [1])


def build_deep_model(input_dim, hidden_dims=[128, 32], meta_data={}):
    from keras.backend import tensorflow_backend as K
    from keras.layers import Dense, Input, concatenate, Embedding, Lambda
    from keras.models import Model

    brain_tensor = Input(shape=(input_dim,), name="voxels")
    features = [brain_tensor]
    meta_embeddings = []
    for meta_name, setup in meta_data.items():
        meta_tensor = Input(shape=(1,), name=meta_name)
        if setup.get("categorical", False):
            vocab_size = setup["vocab_size"]
            embedding_dim = setup["embedding_dim"]
            meta_embedding = Embedding(vocab_size, embedding_dim)(meta_tensor)
        else:
            meta_embedding = Lambda(K.expand_dims,
                                    output_of_expand_dims)(meta_tensor)
        features.append(meta_tensor)
        meta_embeddings.append(meta_embedding)

    # concatenate all features and embeddings into one big vector
    embeddings = [brain_tensor] + meta_embeddings
    if len(embeddings) > 1:
        embedding = concatenate(embeddings, name="BigVector")
    else:
        embedding = embeddings[0]
    for h, hidden_dim in enumerate(hidden_dims):
        embedding = Dense(hidden_dim, activation="relu",
                          name="hidden_%d" % h)(embedding)

    # output layers
    predictions = []
    loss_funcs = {}
    for target_name, setup in targets_setup.items():
        if setup["binary"]:
            activation = "softmax"
            output_dim = 2
            loss_funcs[target_name] = "binary_crossentropy"
        else:
            activation = "linear"
            output_dim = 1
            loss_funcs[target_name] = "mean_squared_error"
        y_pred = Dense(output_dim, activation=activation,
                       name=target_name)(embedding)
        predictions.append(y_pred)

    # build and compile model
    model = Model(inputs=brain_tensor, outputs=predictions)
    model.compile(loss=loss_funcs, optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    # set up some paths
    args = parse_cmdline_args()
    data_dir = args.data_dir
    train_size = args.train_size
    hidden_dims = args.hidden_dims
    num_epochs = args.num_epochs

    t1_dir = os.path.join(data_dir, "pac2018.zip.001_FILES")
    masked_t1_all_npy = os.path.join(data_dir, "arr.npy")
    targets_file = os.path.join(data_dir, "PAC2018_Covariates_Upload.xlsx")

    # load into pandas dataframe
    df = pd.read_excel(targets_file)
    df["t1_path"] = df["PAC_ID"].apply(lambda pac_id: os.path.join(
        t1_dir, "%s.nii" % pac_id))
    print(df)

    X = np.load(masked_t1_all_npy, mmap_mode="r")
    df_ = df.iloc[:len(X)]

    # split into train / val
    indices = np.arange(len(X))
    train_indices, val_indices = train_test_split(indices, train_size=train_size)

    # setup config for features, meta-data, and prediction targets
    meta_data = {
        # "Gender": {"categorical": True, "embedding_dim": 2,
        #            "vocab_size": 2},
        # "Age": {"binary": False},
    }
    targets_setup = {"Label": {"binary": True}}

    # setup train and validation data generators
    train_data = data_gen(train_indices)
    train_data = itertools.cycle(train_data)
    val_data = data_gen(val_indices)
    val_data = list(val_data)

    # build model
    x_batch, _ = next(train_data)
    input_dim = len(x_batch[0])
    del x_batch
    model = build_deep_model(input_dim, hidden_dims=hidden_dims,
                             meta_data=meta_data)

    # plot model (can help debugging quirks)
    plot_model(model, to_file="model.png", show_shapes=True,
               show_layer_names=True)
    os.system("gnome-open model.png")

    # fit model
    model.fit_generator(train_data, epochs=num_epochs, steps_per_epoch=20,
                        # validation_data=val_data,
                        validation_data=itertools.cycle(val_data),
                        validation_steps=20,
    )
