from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, output_raw: bool, use_cuda: bool):
        """
        Initializes the MoleculeModel.

        :param raw_score: Whether the model should apply activation to output.
        :param cuda: Whether or not to use cuda.
        """
        super(MoleculeModel, self).__init__()

        self.activation = nn.Identity()
        if output_raw:
            self.activation = nn.Sigmoid()
        self.use_cuda = use_cuda

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        self.ops = args.ops

        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size*2  # To account for 2 molecules
            if args.use_input_features:
                first_linear_dim += args.features_dim

        if args.drug_only or args.cmpd_only or self.ops != 'concat':
            first_linear_dim = int(first_linear_dim/2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, 1)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, 1),
            ])

        # Create FFN model
        self.single = nn.Sequential(*ffn)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(args.hidden_size, 1)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(args.hidden_size, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, 1),
            ])
        self.combo = nn.Sequential(*ffn)


    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input  # TODO: in future, move drug/cmpd feats out of MPN

        learned_drug = self.encoder([x[0] for x in smiles], [x[0] for x in feats])
        drugOut = self.single(learned_drug)
        learned_cmpd = self.encoder([x[1] for x in smiles], [x[1] for x in feats])
        cmpdOut = self.single(learned_cmpd)

        if self.ops == 'plus':
            newInput = learned_drug + learned_cmpd
        elif self.ops == 'minus':
            newInput = learned_drug - learned_cmpd
        else:
            features_batch = torch.from_numpy(np.stack([x[2] for x in feats])).float()
            if self.use_cuda:
                features_batch = features_batch.cuda()
            newInput = torch.cat((newInput, features_batch), dim=1)

        comboOut = self.combo(newInput)
        output = torch.cat((comboOut, drugOut, cmpdOut), dim=1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if not self.training:
            output = self.activation(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        raise NotImplementedError

    model = MoleculeModel(output_raw=args.output_raw, use_cuda=args.cuda)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
