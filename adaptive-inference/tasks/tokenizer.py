from typing import List, Dict

class CharTokenizer:
    """
    A simple character-level tokenizer for synthetic tasks.
    Supports special tokens for padding, beginning of sequence,
    end of sequence, unknown characters, and input-target separator.
    """
    def __init__(self):
        # Define special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.sep_token = "<sep>"

        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.sep_token
        ]

        # ID mapping for special tokens
        self.id_to_token: Dict[int, str] = {i: token for i, token in enumerate(self.special_tokens)}
        self.token_to_id: Dict[str, int] = {token: i for i, token in enumerate(self.special_tokens)}

        # Add printable ASCII characters to vocabulary
        # Printable ASCII is from range(32, 127), which is space to tilde (~)
        next_id = len(self.special_tokens)
        for char_code in range(32, 127):
            char = chr(char_code)
            self.id_to_token[next_id] = char
            self.token_to_id[char] = next_id
            next_id += 1

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.sep_id = self.token_to_id[self.sep_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        ids = []
        if add_bos:
            ids.append(self.bos_id)
            
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                ids.append(self.unk_id)
                
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        chars = []
        for token_id in ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token in self.special_tokens:
                    if not skip_special_tokens:
                        chars.append(token)
                else:
                    chars.append(token)
            else:
                if not skip_special_tokens:
                    chars.append(self.unk_token)
        return "".join(chars)
