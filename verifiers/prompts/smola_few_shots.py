"""
Few-shot examples for SmolaAgents tools.
"""

from verifiers.parsers.smola_parser import SmolaParser

smola_parser = SmolaParser(fields=["reasoning", ("tool_call", "answer")])
result_parser = SmolaParser(fields=["result"])

SMOLA_CALCULATOR_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?'
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="Let's solve this step by step:\n1. First, calculate how many marbles Janet has (3 times Tom's)",
                tool_call='{"name": "python_interpreter", "args": {"code": "3 * 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="36")
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="2. Now we know:\n   - Tom has 12 marbles\n   - Janet has 36 marbles\n   Let's add them together",
                tool_call='{"name": "python_interpreter", "args": {"code": "36 + 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="48")
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="Together they have 48 marbles:\n- Tom: 12 marbles\n- Janet: 36 marbles (3 times Tom's)\n- Total: 48 marbles",
                answer="48 marbles"
            )
        }
    ],
    [
        {
            'role': 'user',
            'content': 'Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?'
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="To find how many complete batches Samantha can make, I need to divide the total amount of flour by the amount needed per batch.",
                tool_call='{"name": "python_interpreter", "args": {"code": "10 / 2.5"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="4.0")
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="Samantha has 10 cups of flour and each batch requires 2.5 cups of flour.\n10 ÷ 2.5 = 4\nSo Samantha can make 4 complete batches of cookies with her 10 cups of flour.",
                answer="4 batches"
            )
        }
    ]
]

SMOLA_SEARCH_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Question: Which of the following is an example of mechanical weathering?\n\nChoices:\nA. A rock turns red when iron minerals are exposed to oxygen.\nB. A rock breaks apart after water freezes and expands in a crack.\nC. A rock dissolves as acidic rainwater flows over its surface.\nD. A rock changes composition when exposed to extreme heat.'
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="I need to understand what mechanical weathering is and which of these options matches that phenomenon. Let me search for information about mechanical weathering.",
                tool_call='{"name": "web_search", "args": {"query": "what is mechanical weathering examples"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result="## Search Results\n\n• Mechanical Weathering - Definition, Processes and Types\n  Mechanical weathering, also known as physical weathering, is the process by which rocks are broken down into smaller pieces by external forces. Examples include frost wedging (when water freezes and expands in cracks), thermal expansion/contraction, and pressure release.\n\n• Mechanical and Chemical Weathering - Earth Science\n  Mechanical weathering breaks rocks into smaller pieces without changing their composition. Ice wedging occurs when water gets into cracks, freezes, and expands, breaking the rock apart. Other forms include plant root growth, animal activity, and abrasion.\n\n• Mechanical Weathering: Definition, Types, Examples - Geology Science\n  Types of mechanical weathering include frost wedging, salt crystallization, thermal expansion, plant activity, and animal activity. The most common mechanical weathering process in cold climates is frost wedging, where water enters rock cracks, freezes, and expands."
            )
        },
        {
            'role': 'assistant',
            'content': smola_parser.format(
                reasoning="Based on the search results, mechanical weathering is the process by which rocks are broken down into smaller pieces by external forces, without changing their chemical composition. Some examples include:\n\n1. Frost wedging - when water gets into cracks, freezes, and expands, breaking the rock apart\n2. Thermal expansion/contraction\n3. Plant root growth\n4. Animal activity\n5. Abrasion\n\nLooking at the answer choices:\nA. A rock turns red when iron minerals are exposed to oxygen - This is chemical weathering, as it involves a chemical change (oxidation).\nB. A rock breaks apart after water freezes and expands in a crack - This is frost wedging, a form of mechanical weathering.\nC. A rock dissolves as acidic rainwater flows over its surface - This is chemical weathering (dissolution).\nD. A rock changes composition when exposed to extreme heat - This is a chemical/thermal change, not mechanical weathering.\n\nThe answer is B, as it's the only option that describes mechanical weathering.",
                answer="B"
            )
        }
    ]
]