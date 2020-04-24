# TODO temp hack DLK
import pdb
import numpy as np
# / temp hack

import torch
from torch.autograd import Variable


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab, vectors):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.vectors = vectors

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

    def get_decoder_features(self, src_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            # TODO temp hack DLK
            # pdb.set_trace()
            if torch.cuda.is_available():
                src_id_seq = src_id_seq.cuda()
            # / temp hack
            # pdb.set_trace()
            if self.vectors:
                vecs = self.build_vec_batch(self.src_vocab, src_id_seq, self.vectors)
            else:
                vecs = None
            # softmax_list, _, other = self.model(vecs, src_id_seq, [len(src_seq)])
            # test = self.model(vecs, src_id_seq, [len(src_seq)])
            decoder_outputs, decoder_hidden, ret_dict, encoder_outputs, encoder_hidden = self.model(vecs, src_id_seq, [len(src_seq)])

        # pdb.set_trace()

        return ret_dict, encoder_outputs

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other, encoder_outputs = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, encoder_outputs

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        topk_tgt_seq = []
        for k in range(other['topk_sequence'][0].shape[1]):
            tgt_id_seq = [other['sequence'][di][k].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            topk_tgt_seq.append(tgt_seq)

        scores = other['score'].data.tolist()[0]

        return topk_tgt_seq, scores

