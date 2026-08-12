import unittest
import sys
import os
import re

# Add the parent directory to sys.path so we can import modules from tasks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator import SyntheticTaskGenerator
from tokenizer import CharTokenizer

class TestSyntheticTaskGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticTaskGenerator(seed=42)
        self.tokenizer = CharTokenizer()

    def test_dataset_distribution(self):
        """
        Verify that generate_dataset yields exactly:
        - 40% Arithmetic
        - 35% Logic
        - 25% Language
        """
        num_samples = 1000
        dataset = self.generator.generate_dataset(num_samples)
        
        self.assertEqual(len(dataset), num_samples)
        
        arithmetic_count = sum(1 for item in dataset if item['domain'] == 'arithmetic')
        logic_count = sum(1 for item in dataset if item['domain'] == 'logic')
        language_count = sum(1 for item in dataset if item['domain'] == 'language')
        
        # Check exact counts
        self.assertEqual(arithmetic_count, 400)
        self.assertEqual(logic_count, 350)
        self.assertEqual(language_count, 250)

        # Check oracle distribution properties
        for item in dataset:
            if item['domain'] == 'arithmetic':
                self.assertEqual(item['oracle_dist'], [1.0, 0.0, 0.0])
            elif item['domain'] == 'logic':
                self.assertEqual(item['oracle_dist'], [0.0, 1.0, 0.0])
            elif item['domain'] == 'language':
                self.assertEqual(item['oracle_dist'], [0.0, 0.0, 1.0])

    def test_arithmetic_correctness(self):
        """
        Verify that generated arithmetic expressions evaluate to the target string.
        """
        for _ in range(50):
            input_prompt, target_str = self.generator.generate_arithmetic_task()
            
            # Input format: "calc: <expression>"
            self.assertTrue(input_prompt.startswith("calc: "))
            expr = input_prompt[len("calc: "):]
            
            # Evaluate using standard python arithmetic
            expected_val = eval(expr)
            self.assertEqual(target_str, str(expected_val))

    def test_logic_correctness(self):
        """
        Verify that generated logic tasks evaluate to the target string ("True" or "False").
        """
        for _ in range(50):
            input_prompt, target_str = self.generator.generate_logic_task()
            self.assertTrue(input_prompt.startswith("logic: "))
            
            self.assertIn(target_str, ["True", "False"])
            
            if "->" in input_prompt:
                # Variable-based deduction: "logic: A=True; B=False -> A or B?"
                match = re.match(r"logic: (.*) -> (.*)\?", input_prompt)
                self.assertIsNotNone(match)
                assignments_str, query_str = match.groups()
                
                # Parse assignments
                local_env = {}
                for part in assignments_str.split("; "):
                    var, val = part.split("=")
                    local_env[var] = (val == "True")
                
                # Evaluate query
                expected_val = eval(query_str, {}, local_env)
                self.assertEqual(target_str, str(expected_val))
            else:
                # Simple expression: "logic: True and not False"
                expr = input_prompt[len("logic: "):]
                expected_val = eval(expr)
                self.assertEqual(target_str, str(expected_val))

    def test_language_correctness(self):
        """
        Verify that generated language tasks operate correctly.
        """
        for _ in range(50):
            input_prompt, target_str = self.generator.generate_language_task()
            self.assertTrue(input_prompt.startswith("lang: "))
            
            if "reverse" in input_prompt:
                match = re.match(r"lang: reverse\((.*)\)", input_prompt)
                word = match.group(1)
                self.assertEqual(target_str, word[::-1])
            elif "count_vowels" in input_prompt:
                match = re.match(r"lang: count_vowels\((.*)\)", input_prompt)
                word = match.group(1)
                expected_count = sum(1 for c in word if c in "aeiouAEIOU")
                self.assertEqual(target_str, str(expected_count))
            elif "repeat" in input_prompt:
                match = re.match(r"lang: repeat\((.*), (\d+)\)", input_prompt)
                word, count = match.groups()
                self.assertEqual(target_str, word * int(count))
            elif "shift" in input_prompt:
                match = re.match(r"lang: shift\((.*), (\d+)\)", input_prompt)
                word, shift_amt = match.groups()
                shift_amt = int(shift_amt)
                expected_shifted = "".join([chr((ord(c) - 97 + shift_amt) % 26 + 97) for c in word])
                self.assertEqual(target_str, expected_shifted)

    def test_tokenizer_roundtrip(self):
        """
        Verify that tokenizer encodes and decodes printables correctly.
        """
        test_strings = [
            "hello world",
            "calc: 45 + 23 - 10",
            "logic: True and not False",
            "lang: reverse(elephant)",
            "Special characters: !@#$%^&*()_+=-`~[]\\{}|;':\",./<>?"
        ]
        
        for s in test_strings:
            encoded = self.tokenizer.encode(s, add_bos=True, add_eos=True)
            self.assertEqual(encoded[0], self.tokenizer.bos_id)
            self.assertEqual(encoded[-1], self.tokenizer.eos_id)
            
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
            self.assertEqual(decoded, s)

if __name__ == "__main__":
    unittest.main()
