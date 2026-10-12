"""Exercise schema filtering with real tokenizer backends, without weights."""

import json
import string
import unittest

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.decoding import SchemaConstrainedDecoder
from src.schemas import load_perception_schema


class DecodingTests(unittest.TestCase):
    @staticmethod
    def tokenizer() -> PreTrainedTokenizerFast:
        symbols = ["[UNK]", "[EOS]", "0", "{", "}", '"', ":", " ", "x", "y", "1", "2", "true", "false"]
        backend = Tokenizer(models.WordLevel({symbol: i for i, symbol in enumerate(symbols)}, unk_token="[UNK]"))
        backend.decoder = decoders.Fuse()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")

    def test_boolean_grammar_excludes_number_and_early_eos(self) -> None:
        tokenizer = self.tokenizer()
        allowed = SchemaConstrainedDecoder(tokenizer).build_token_filter({"type": "boolean"})
        prompt = [tokenizer.eos_token_id]
        first = allowed(0, torch.tensor(prompt))
        self.assertIn(tokenizer.convert_tokens_to_ids("true"), first)
        self.assertIn(tokenizer.convert_tokens_to_ids("false"), first)
        self.assertNotIn(tokenizer.convert_tokens_to_ids("0"), first)
        self.assertNotIn(tokenizer.eos_token_id, first)
        prompt.append(tokenizer.convert_tokens_to_ids("true"))
        self.assertIn(tokenizer.eos_token_id, allowed(0, torch.tensor(prompt)))

    def test_required_json_field_and_enum_are_enforced(self) -> None:
        tokenizer = self.tokenizer()
        allowed = SchemaConstrainedDecoder(tokenizer).build_token_filter({
            "type": "object", "properties": {"x": {"type": "string", "enum": ["y"]}},
            "required": ["x"], "additionalProperties": False,
        })
        sent = [tokenizer.eos_token_id]
        for index, symbol in enumerate(('{', '"', 'x', '"', ':', '"', 'y', '"', '}')):
            choices = allowed(0, torch.tensor(sent))
            self.assertIn(tokenizer.convert_tokens_to_ids(symbol), choices)
            if index == 2:
                self.assertNotIn(tokenizer.convert_tokens_to_ids('y'), choices)
            if index == 6:
                self.assertNotIn(tokenizer.convert_tokens_to_ids('x'), choices)
            sent.append(tokenizer.convert_tokens_to_ids(symbol))
        self.assertIn(tokenizer.eos_token_id, allowed(0, torch.tensor(sent)))

    def test_new_filter_does_not_inherit_previous_generation_state(self) -> None:
        tokenizer = self.tokenizer()
        decoder = SchemaConstrainedDecoder(tokenizer)
        first = decoder.build_token_filter({"type": "string", "enum": ["x"]})
        prompt = [tokenizer.eos_token_id]
        sent = list(prompt)
        for symbol in ('"', 'x', '"'):
            first(0, torch.tensor(sent))
            sent.append(tokenizer.convert_tokens_to_ids(symbol))
        self.assertIn(tokenizer.eos_token_id, first(0, torch.tensor(sent)))
        second = decoder.build_token_filter({"type": "string", "enum": ["y"]})
        choices = second(0, torch.tensor(prompt))
        self.assertNotIn(tokenizer.eos_token_id, choices)
        choices = second(0, torch.tensor(prompt + [tokenizer.convert_tokens_to_ids('"')]))
        self.assertIn(tokenizer.convert_tokens_to_ids("y"), choices)
        self.assertNotIn(tokenizer.convert_tokens_to_ids("x"), choices)

    def test_space_prefixed_tokens_are_kept_for_word_tokenizers(self) -> None:
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "0": 2, "true": 3, "false": 4}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
        decoder = SchemaConstrainedDecoder(tokenizer)
        tokens = {token_id: (text, starts_word) for token_id, text, starts_word in decoder.tokenizer_data.regular_tokens}
        self.assertEqual(tokens[3], (" true", True))
        allowed = decoder.build_token_filter({"type": "boolean"})
        self.assertIn(3, allowed(0, torch.tensor([tokenizer.eos_token_id])))

    def test_actual_perception_schema_accepts_complete_graph_and_blocks_early_eos(self) -> None:
        symbols = ["[UNK]", "[EOS]"] + list(string.printable)
        backend = Tokenizer(models.WordLevel({symbol: i for i, symbol in enumerate(symbols)}, unk_token="[UNK]"))
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
        allowed = SchemaConstrainedDecoder(tokenizer).build_token_filter(load_perception_schema())
        graph = {"image": {"width": 80, "height": 60}, "objects": [{
            "id": "red_cube_1", "kind": "movable", "label": "red cube", "color": "red",
            "bbox_2d": [10, 10, 30, 30], "pick_point_2d": [12, 12], "confidence": 0.8,
            "relations": {"inside": None, "on_top_of": None,
                          "held_by_human": False, "partially_occluded": False},
        }]}
        sent = [tokenizer.eos_token_id]
        for symbol in json.dumps(graph):
            choices = allowed(0, torch.tensor(sent))
            self.assertNotIn(tokenizer.eos_token_id, choices)
            self.assertIn(tokenizer.convert_tokens_to_ids(symbol), choices, tokenizer.decode(sent))
            sent.append(tokenizer.convert_tokens_to_ids(symbol))
        self.assertIn(tokenizer.eos_token_id, allowed(0, torch.tensor(sent)))


if __name__ == "__main__":
    unittest.main()
