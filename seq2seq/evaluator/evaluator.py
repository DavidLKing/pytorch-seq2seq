from __future__ import print_function, division

# TODO temp hack DLK
import pdb
import numpy as np
# from seq2seq.trainer.supervised_trainer import build_vec_batch, clean_word
# / temp hack

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    # TODO temp hack
    # make this an importable class:
    def clean_word(self, word_array):
        no_grammar = []
        for elem in word_array:
            if '=' not in elem:
                no_grammar.append(elem)
        return no_grammar

    def build_vec_batch(self, vocab, input_var, vectors):
        vec_size = vectors.vector_size
        # holder for vectors
        batch_vecs = []
        # get the strings
        # input_text = []
        for ex in input_var:
            text = [vocab.itos[x] for x in ex]
            text = self.clean_word(text)
            text = ''.join(text)
            # input_text.append(text)
            if text in vectors:
                batch_vecs.append(vectors[text])
            else:
                batch_vecs.append(np.random.normal(0.0, 0.1, vec_size))
        return batch_vecs

    def evaluate(self, model, data, vectors, input_vocab):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:

                input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                if vectors:
                    vecs = self.build_vec_batch(input_vocab, input_variables, vectors)
                else:
                    vecs = None

                # decoder_outputs, decoder_hidden, other = model(vecs, input_variables, input_lengths.tolist(), target_variables)
                decoder_outputs, decoder_hidden, ret_dict, encoder_outputs, encoder_hidden = model(vecs, input_variables, input_lengths.tolist(), target_variables)

                # Evaluation
                seqlist = ret_dict['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    # TODO temp hack DLK
                    if torch.cuda.is_available():
                        target = target.cuda()
                        non_padding = non_padding.cuda()
                        # pdb.set_trace()
                    if len(seqlist[step]) > 1:
                        for item in seqlist[step]:
                            correct = item.view(-1).eq(target).masked_select(non_padding).sum().item()
                            match += correct
                    else:
                        correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                        match += correct
                    # / temp hack
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
