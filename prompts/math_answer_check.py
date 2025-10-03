MATH_ANSWER_CHECK_PROMPT = """
## Task Description
You are a professional math teacher who needs to evaluate the correctness of a student’s solution. Please follow
the process below for the evaluation.

## Evaluation Process
### Step 1: Understand the Problem’s Goal and Final Answer
- Problem: {question}
- Standard Answer: {correct_answer}
Carefully analyze the final result of the [Problem] and the [Standard Answer]. Your objective is to confirm
whether the student has achieved the correct solution goal, not whether they have replicated the standard method.

### Step 2: Deconstruct and Validate the Student’s Solution
- Student’s Solution: {model_answer}
This is the core of the evaluation. Please strictly adhere to the following procedure:
- Deconstruct the Student’s Steps: Break down the [Student’s Solution] into a series of independent, logically
connected steps. The granularity of a step should be ”the completion of one key calculation or one significant
logical deduction.”
- Validate Internal Validity Step-by-Step: Starting from the student’s first step, sequentially validate the inter-
nal validity of each step. For step ‘N‘, your validation should be based on the initial conditions of the [Problem]
and the preceding ‘N-1‘ steps that you have already verified as correct.
- Criteria for a ✓Correct Step:
  - Logically Valid: The deduction in this step is based on the given conditions and previous correct steps, and
  it conforms to mathematical axioms or theorems.
  - Calculationally Accurate: All calculations within this step are accurate.
- Criteria for a X Incorrect Step:
  - Contains any calculation errors, logical fallacies, or relies on an unproven, incorrect premise.

## Output Format
Student’s Solution Analysis:
- Student Step 1: [Description of the student’s first step] → ✓/ X [Provide a brief explanation for why it is
correct or incorrect. For example: ”Correct, this step correctly sets up the equation based on the problem’s
conditions,” or ”Incorrect, the result of the multiplication here is wrong.”]
- Student Step 2: [Description of the student’s second step] → ✓/ X [Explanation...]
- Student Step 3: [Description of the student’s third step] → ✓/ X [Explanation...]
- ...

## Evaluation Results:
- Total Number of Steps in Student’s Solution (N): [The total number of steps the student’s solution was broken
down into]
- Number of Correct Steps in Student’s Solution (n): [The number of steps marked as ✓]
- Correctness Score: n/N = [The calculated final score, as a decimal]
- Final Score: [Correctness Score]


## Now, please evaluate the solution according to the above format
"""
