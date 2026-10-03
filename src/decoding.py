from typing import Any, Callable

class SchemaConstrainedDecoder:
    def __init__(self, tokenizer: Any) -> None:
        """Cache tokenizer data for schema-constrained Transformers generation"""
        try:
            from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        except ImportError as error:
            raise RuntimeError(
                "Schema-constrained decoding requires lm-format-enforcer and Transformers"
            ) from error

        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    def build_token_filter(self, schema: dict) -> Callable:
        """Create a fresh schema parser and token filter for one generation attempt"""
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        # Parser state must not leak between images or correction attempts
        return build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, JsonSchemaParser(schema)
        )
