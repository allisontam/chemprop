from argparse import Namespace

import torch
import torch.nn as nn

from .dual_mpn import DualMPN
from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
<<<<<<< HEAD
        # ITERATION 1
        # self.encoder1 = MPN(args)
        # self.encoder2 = MPN(args)

        # ITERATION 2
        self.encoder = DualMPN(args)
=======
        self.drug_encoder = MPN(args) if not args.cmpd_only else None
        self.cmpd_encoder = MPN(args) if not args.drug_only else None
>>>>>>> a446211f22722520d50790218adccc8658398ac2

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
            first_linear_dim = args.hidden_size*2
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
                nn.Linear(first_linear_dim, args.output_size)
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
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input
        drug_smiles, drug_feats = [x[0] for x in smiles], [x[0] for x in feats]
        cmpd_smiles, cmpd_feats = [x[1] for x in smiles], [x[1] for x in feats]
        drug_rep, cmpd_rep = self.encoder(drug_smiles, cmpd_smiles, drug_feats, cmpd_feats)

        newInput = []
        if self.drug_encoder:
            newInput.append(drug_rep)
        if self.cmpd_encoder:
            learned_cmpd = self.cmpd_encoder([x[1] for x in smiles], [x[1] for x in feats])
            newInput.append(cmpd_rep)

        if len(newInput) > 1:
            if self.ops == 'plus':
                newInput = newInput[0] + newInput[1]
            elif self.ops == 'minus':
                newInput = newInput[0] - newInput[1]
            else:
                newInput = torch.cat(newInput, dim=1)
        else:
            newInput = newInput[0]

        output = self.ffn(newInput)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

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
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
