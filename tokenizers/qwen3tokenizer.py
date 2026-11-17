'''
    Qwen3 Tokenizer
    - Uses the HuggingFace Tokenizer library to tokenize text for the Qwen3 model.
'''
import re
from tokenizers import Tokenizer
from pathlib import Path

class QwenTokenizer:

    # initializing the special Tokens
    _SPECIAL = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"
    ]
    # split only on the known special tokens (anything else shaped like <|...|> is encoded as normal text)
    _SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _SPECIAL) + ")")

    def __init__(self,
                 tokenizer_file_path = "tokenizer-base.json",
                 apply_chat_template = False,
                 add_generation_prompt = False,
                 add_thinking = False):
        
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        # loading the tokenizer
        tok_path = Path(tokenizer_file_path)
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file_path}")

        self._tok = Tokenizer.from_file(tokenizer_file_path)
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIAL}
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id[self.pad_token]

        f = tok_path.name.lower()

        # setting the eos token based on the model type i.e base model or reasoning model.
        if "base" in f and "reasoning" not in f:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"

        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        """
            Encoding the prompt based on the configuuration
        """
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        # a prompt that is exactly one special token maps directly to its id
        if prompt in self._special_to_id:
            return [self._special_to_id[prompt]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)

        return ids


    def _wrap_chat(self, user_msg):
        # we are wrapping the user message with the chat template.
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if not self.add_thinking:
                # thinking disabled -> empty think block, as in the Qwen3 template
                s += "<think>\n\n</think>\n\n"
        return s
     


    def decode(self, token_ids):
        # Decoding the token ids back to the token string.
        return self._tok.decode(token_ids, skip_special_tokens=False)
        


# sanity check for the tokenizer
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = QwenTokenizer(tokenizer_file_path="tokenizer-base.json", apply_chat_template=True)

    # Sample prompt
    prompt = "Hello, how are you?"

    # Encode the prompt
    encoded_ids = tokenizer.encode(prompt)
    print("Encoded IDs:", encoded_ids)

    # Decode the token IDs back to string
    decoded_string = tokenizer.decode(encoded_ids)
    print("Decoded String:", decoded_string)