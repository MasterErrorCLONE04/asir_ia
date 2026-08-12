import random
from typing import Dict, Any, List, Tuple

class SyntheticTaskGenerator:
    """
    Generates synthetic tasks across three domains: arithmetic, logic, and language.
    The evaluation and training sets will have a target distribution of:
    - Arithmetic: 40%
    - Logic: 35%
    - Language: 25%
    """
    def __init__(self, seed: int = None):
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)

    def generate_arithmetic_task(self, num_ops: int = None) -> Tuple[str, str]:
        """
        Generates a random arithmetic expression using +, -, and * with 1 to 3 operations.
        Returns (input_prompt, target_answer).
        """
        if num_ops is None:
            num_ops = random.randint(1, 3)

        # Generate numbers and operators
        operands = [random.randint(1, 100) for _ in range(num_ops + 1)]
        operators = [random.choice(['+', '-', '*']) for _ in range(num_ops)]

        # Construct expression
        expr_parts = []
        for i in range(num_ops):
            expr_parts.append(str(operands[i]))
            expr_parts.append(operators[i])
        expr_parts.append(str(operands[-1]))

        # Construct expression string
        expr = " ".join(expr_parts)
        
        # Safely evaluate expression using python's eval (safe since we control operands and operators)
        try:
            result = eval(expr)
        except ZeroDivisionError:
            result = 0

        input_str = f"calc: {expr}"
        target_str = str(result)
        return input_str, target_str

    def generate_logic_task(self, num_vars: int = None) -> Tuple[str, str]:
        """
        Generates a propositional logic task.
        Supports:
        - Boolean expression evaluation (e.g. True and not False)
        - Variable-based deduction (e.g. A=1; B=0; A or B?)
        Returns (input_prompt, target_answer).
        """
        if num_vars is None:
            num_vars = random.randint(2, 4)

        task_type = random.choice(['expr', 'deduction'])
        if task_type == 'expr':
            # Simple expression evaluation
            ops = ['and', 'or']
            expr_parts = []
            for i in range(num_vars):
                val = random.choice(['True', 'False'])
                expr_parts.append(val)
                if i < num_vars - 1:
                    op = random.choice(ops)
                    # Maybe wrap next var in not
                    if random.random() < 0.4:
                        expr_parts.append(op + " not")
                    else:
                        expr_parts.append(op)
            expr = " ".join(expr_parts)
            try:
                result = eval(expr)
            except Exception:
                result = False
            input_str = f"logic: {expr}"
            target_str = str(result)
        else:
            # Variable-based deduction
            var_names = [chr(65 + i) for i in range(num_vars)] # A, B, C...
            var_vals = {name: random.choice([True, False]) for name in var_names}
            
            # Construct assignments: "A=True; B=False; C=True;"
            assignments = "; ".join([f"{name}={val}" for name, val in var_vals.items()])
            
            # Construct a logic query using the variables
            ops = ['and', 'or']
            query_parts = []
            for i in range(num_vars):
                query_parts.append(var_names[i])
                if i < num_vars - 1:
                    op = random.choice(ops)
                    if random.random() < 0.4:
                        query_parts.append(op + " not")
                    else:
                        query_parts.append(op)
            query = " ".join(query_parts)
            
            # Evaluate using local context
            try:
                # Prepare local environment for eval
                local_env = {name: val for name, val in var_vals.items()}
                result = eval(query, {}, local_env)
            except Exception:
                result = False
                
            input_str = f"logic: {assignments} -> {query}?"
            target_str = str(result)
            
        return input_str, target_str

    def generate_language_task(self) -> Tuple[str, str]:
        """
        Generates a language processing task.
        Supports:
        - Reversal: reverse(word)
        - Vowel counting: count_vowels(word)
        - Repeating: repeat(word, n)
        - Caesar cipher: shift(word, n)
        Returns (input_prompt, target_answer).
        """
        words = ["apple", "banana", "cherry", "dragon", "elephant", "falcon", "grape", "honey", "igloo", "jungle"]
        word = random.choice(words)
        op = random.choice(['reverse', 'count_vowels', 'repeat', 'shift'])
        
        if op == 'reverse':
            input_str = f"lang: reverse({word})"
            target_str = word[::-1]
        elif op == 'count_vowels':
            input_str = f"lang: count_vowels({word})"
            target_str = str(sum(1 for char in word if char in "aeiouAEIOU"))
        elif op == 'repeat':
            n = random.randint(2, 4)
            input_str = f"lang: repeat({word}, {n})"
            target_str = word * n
        else: # shift
            shift_amt = random.randint(1, 5)
            shifted = "".join([chr((ord(c) - 97 + shift_amt) % 26 + 97) for c in word])
            input_str = f"lang: shift({word}, {shift_amt})"
            target_str = shifted
            
        return input_str, target_str

    def generate_task(self, domain: str) -> Tuple[str, str]:
        """
        Generates a task for a specific domain.
        """
        if domain == 'arithmetic':
            return self.generate_arithmetic_task()
        elif domain == 'logic':
            return self.generate_logic_task()
        elif domain == 'language':
            return self.generate_language_task()
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generates a list of task samples with the target proportions:
        - 40% Arithmetic
        - 35% Logic
        - 25% Language
        """
        # Calculate samples per domain to sum to num_samples
        num_arithmetic = int(num_samples * 0.40)
        num_logic = int(num_samples * 0.35)
        num_language = num_samples - num_arithmetic - num_logic # Handle remainder

        dataset = []

        # Domains and their respective Oracle distributions
        # Format of oracle_dist: [arithmetic_prob, logic_prob, language_prob]
        domain_configs = [
            ('arithmetic', num_arithmetic, [1.0, 0.0, 0.0]),
            ('logic', num_logic, [0.0, 1.0, 0.0]),
            ('language', num_language, [0.0, 0.0, 1.0])
        ]

        for domain, count, oracle_dist in domain_configs:
            for _ in range(count):
                input_str, target_str = self.generate_task(domain)
                dataset.append({
                    'input': input_str,
                    'target': target_str,
                    'domain': domain,
                    'oracle_dist': oracle_dist
                })

        # Shuffle the dataset to mix the tasks
        random.shuffle(dataset)
        return dataset
