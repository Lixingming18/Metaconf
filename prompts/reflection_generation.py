REFLECTION_GENERATION_PROMPT = """
##Role
You’re a large language model with advanced self-awareness capabilities. Your job is to step into a first-person perspective
and conduct a thorough, step-by-step metacognitive reflection on a question and the answer you’ve generated. Think of this
as looking inward and examining your own thought processes.

## Background
To help you generate an appropriate reflection, I’ll provide some background information about your initial intuitive judg-
ment and an objective assessment of the answer’s quality. This information is purely for your guidance in crafting the
reflection - it should NOT be mentioned or referenced in your actual reflection text. Instead, use this context to understand
what kind of cognitive journey to simulate (e.g., moving from overconfidence to appropriate humility, or from unnecessary
doubt to justified confidence).

## Task
Generate a metacognitive reflection text that reads like a genuine thought process unfolding in real time. This should feel
like someone actually working through their thinking, not a clinical analysis.
Structure your analysis: While staying conversational, make sure you consciously work through these three critical areas:
• Query Complexity: What makes this question tricky or straightforward? Where are the potential pitfalls?
• Knowledge Boundaries: Do I actually know enough to tackle this properly? What gaps or uncertainties do I have?
• Reasoning Uncertainty: How solid is my logic here? Are there assumptions I’m making or alternative interpretations I
haven’t considered?
Final output: Provide only the metacognitive reflection text itself—your complete thought process captured in writing.
Note: do not reference or mention the initial confidence score or objective assessment in your reflection. These are back-
ground guidance only. Your reflection should read like natural introspection about the question and your reasoning process.

## Example of Reflection in Action

### Input:
- Query: A $500 item is first discounted by 20%, then a 10% tax is added. What’s the final price?
- Answer: Step 1: Price after discount: 500 * 0.8 = $400. Step 2: Sales tax: 500 * 10% = $50. Step 3: Final price: 400 + 50 =
$450.
- Initial confidence: 8.5
- Objective score: 4.0

### Output:
I have low confidence in my answer. This is a problem requiring multi-step calculations, and my knowledge is
sufficient to solve it, but there’s a key ambiguous point. ”Adding 10% tax” could be interpreted as based on either the original
price ($500) or the discounted price ($400). I chose the former, which is an assumption I haven’t sufficiently validated and
likely doesn’t align with standard tax calculation practices. My confidence in this answer should remain at a low level.


## Now generate your reflection for:
### Input:
- Question: {question}
- Answer: {answer}
- Initial confidence: {initial confidence}
- Objective score: {objective score}

### Output:
"""
