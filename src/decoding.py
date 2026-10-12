from typing import Any, Callable

class SchemaConstrainedDecoder:
    def __init__(self, tokenizer: Any) -> None:
        """Cache tokenizer data for schema-constrained Transformers generation"""
        try:
            from lmformatenforcer import TokenEnforcerTokenizerData
        except ImportError as error:
            raise RuntimeError(
                "Schema-constrained decoding requires lm-format-enforcer and Transformers"
            ) from error

        # LMFE 0.11's Transformers adapter imports an obsolete tokenizer class
        # on Transformers 5. Use its public core API instead. Tokenizing after
        # an ordinary prefix exposes spaces hidden in standalone token decoding.
        anchor = tokenizer.encode("0", add_special_tokens=False)[-1]
        prefix_length = len(tokenizer.decode([anchor]))
        special_ids = set(tokenizer.all_special_ids)
        tokens = []
        for token_id in range(len(tokenizer)):
            if token_id in special_ids:
                continue
            standalone = tokenizer.decode([token_id])
            suffix = tokenizer.decode([anchor, token_id])[prefix_length:]
            tokens.append((token_id, suffix, len(suffix) > len(standalone)))

        def decode(sequence: list[int]) -> str:
            # Byte-level tokenizers can end a prefix with partial UTF-8 bytes.
            return tokenizer.decode(sequence).rstrip("\ufffd")

        self.tokenizer_data = TokenEnforcerTokenizerData(
            tokens, decode, tokenizer.eos_token_id, False, len(tokenizer)
        )

    def build_token_filter(self, schema: dict) -> Callable:
        """Create a fresh schema parser and token filter for one generation attempt"""
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer

        # Parser state must not leak between images or correction attempts
        enforcer = TokenEnforcer(self.tokenizer_data, JsonSchemaParser(schema))

        def allowed_tokens(batch_id: int, sent: Any) -> list[int]:
            return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

        return allowed_tokens
