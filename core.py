from typing import List, Tuple, Optional

from functools import partial

import numpy as np

def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s

def encode_string(
    input_string: str,
    vocab: str,
) -> List[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
    ----
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
    -------
        A list encoding the input_string
    """
    try:
        return list(map(vocab.index, input_string))
    except ValueError:
        raise ValueError(
            f"some characters cannot be found in 'vocab'. \
                         Please check the input string {input_string} and the vocabulary {vocab}"
        )


def encode_sequences(
    sequences: List[str],
    vocab: str,
    target_size: Optional[int] = None,
    eos: int = -1,
    sos: Optional[int] = None,
    pad: Optional[int] = None,
    dynamic_seq_length: bool = False,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
    ----
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD
        dynamic_seq_length: if `target_size` is specified, uses it as upper bound and enables dynamic sequence size

    Returns:
    -------
        the padded encoded data as a tensor
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    if not isinstance(target_size, int) or dynamic_seq_length:
        # Maximum string length + EOS
        max_length = max(len(w) for w in sequences) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)

    # Pad all sequences
    if isinstance(pad, int):  # pad with padding symbol
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        # In that case, add EOS at the end of the word before padding
        default_symbol = pad
    else:  # pad with eos symbol
        default_symbol = eos
    encoded_data: np.ndarray = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)

    # Encode the strings
    for idx, seq in enumerate(map(partial(encode_string, vocab=vocab), sequences)):
        if isinstance(pad, int):  # add eos at the end of the sequence
            seq.append(eos)
        encoded_data[idx, : min(len(seq), target_size)] = seq[: min(len(seq), target_size)]

    if isinstance(sos, int):  # place sos symbol at the beginning of each sequence
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos

    return encoded_data


class NestedObject:
    """Base class for all nested objects in doctr"""

    _children_names: List[str]

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-object, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        if hasattr(self, "_children_names"):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + "\n"
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append("(" + key + "): " + child_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def build_target(
        self,
        gts: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
        ----
            gts: list of ground-truth labels

        Returns:
        -------
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        self.vocab = vocab
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"
