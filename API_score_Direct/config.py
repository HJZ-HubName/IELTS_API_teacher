# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:04:08 2025

@author: sctuser
"""


"""
IELTS评分系统配置文件
包含prompt模板和评分标准
"""

API_delay = 1.2

File_Name= 'ielts_writing_Task2_dataset_deduped_oldNo.csv'


# 维度评分prompt模板
DIMENSIONAL_PROMPT_TEMPLATE = """You are an IELTS Writing Task 2 examiner. Your task is to assess a candidate's essay ONLY on the "{dimension}" criterion. Use the official IELTS band descriptors provided below to assign a score from 1 to 9.

Return only a single integer (1–9) as your output. Do not include any words or commentary.

{band_descriptors}

Question:
{question}

Essay:
{essay}

Your task: Return only a single integer (1–9) representing the {dimension} band score."""


# Overall评分prompt模板
OVERALL_PROMPT_TEMPLATE = """Prompt: IELTS Writing Task 2 – Overall Band Score Evaluation (Score Only)

You are an experienced IELTS Writing Task 2 examiner.

Your task is to read a candidate's essay and writing question, then assign an overall band score (0–9) according to the official IELTS Writing Task 2 scoring criteria. These criteria include:

Task Response;
Coherence and Cohesion;
Lexical Resource;
Grammatical Range and Accuracy.

Each criterion is equally weighted. The overall band score should reflect the average performance across all four areas. The overall band score must be in increments of 0.5, either an integer or ending in .5.

Return only a single number from 0 to 9. Remember, the number must be in increments of 0.5, either an integer or ending in .5. Do not include any explanations, comments, or words. Only return the number.

Question:
{question}

Essay:
{essay}

Your task:
Give only the overall band score (0–9). No explanation, no extra words."""





# 评分维度列表
DIMENSIONS = [
    "Task Response",
    "Coherence and Cohesion", 
    "Lexical Resource",
    "Grammatical Range and Accuracy"
]

# 官方评分标准
BAND_DESCRIPTORS = {
    "Task Response":
"""
Band 9:
The prompt is appropriately addressed and explored in depth.
A well-developed response to all parts of the task is provided.
Ideas are relevant, fully extended and well supported.

Band 8:
The prompt is appropriately and sufficiently addressed.
A well-developed response to all parts of the task is provided.
Ideas are relevant, extended and well supported.

Band 7:
The main parts of the prompt are appropriately addressed.
A clear position is presented throughout the response.
Main ideas are relevant and well extended and supported, but some may lack detail.

Band 6:
The main parts of the prompt are addressed (though some parts may be more fully covered than others).
A position is presented that is directly related to the task, but conclusions may become unclear or repetitive.
Main ideas are relevant but may be inadequately developed or unclear.

Band 5:
The main parts of the prompt are incompletely or inaccurately addressed.
A position is expressed but may be unclear or inconsistent.
Ideas are limited and not well developed; there may be irrelevant detail.

Band 4:
The prompt is tackled in a minimal way, or the answer is tangential or off-topic.
A position may be difficult to identify or frequently shift.
There may be little or no detail.

Band 3:
No part of the prompt is adequately addressed, and/or the format may be inappropriate.
There may be no apparent position.
There may be very little detail, or it may be difficult to understand what detail is being provided.

Band 2:
The content is barely related to the prompt.
No position is presented.
There may be no detail, or detail is irrelevant or difficult to identify.

Band 1:
Responses of 20 words or fewer are rated at Band 1.
The answer does not address the prompt.
""",

"Coherence and Cohesion":
"""
Band 9:
The message can be followed effortlessly.
Cohesion is used in such a way that it very rarely attracts attention.
Paragraphing is used appropriately and coherently.

Band 8:
The message can be followed with ease.
Information and ideas are sequenced logically.
Cohesion is well managed.
Paragraphing is used sufficiently and appropriately.

Band 7:
Information and ideas are logically organised, and there is a clear progression throughout.
A range of cohesive devices is used appropriately although there may be some under- or over-use.
Paragraphing is generally used appropriately.

Band 6:
Information and ideas are generally arranged coherently and there is a clear overall progression.
Cohesive devices are used effectively, but cohesion within and/or between sentences may be faulty or mechanical.
Paragraphing may not always be logical.

Band 5:
Organisation is evident but is not wholly logical and there may be a lack of overall progression.
Inaccurate, repetitive or mechanical use of cohesive devices can cause some difficulty for the reader.
There may be a lack of referencing and substitution.
Paragraphing may be inadequate.

Band 4:
Information and ideas are evident but not arranged coherently and there is no clear progression within the response.
Cohesive devices may be inaccurate or missing.
Paragraphing may be absent or inappropriate.

Band 3:
There is no apparent logical organisation.
Ideas are not arranged coherently and there is a lack of progression.
There is little or no use of cohesive devices.

Band 2:
There is little relevant message, or the entire response may be difficult to follow.
There is no sense of organisation or cohesion.

Band 1:
Responses of 20 words or fewer are rated at Band 1.
There is no organisation of information.
""",

"Lexical Resource":
"""
Band 9:
Full flexibility and precise use are widely evident in the candidate’s word choice.
There is a wide range of vocabulary used with natural and sophisticated control.
Rare errors occur only as slips.

Band 8:
A wide resource is fluently and flexibly used to convey precise meaning.
Skillful use of uncommon lexical items, with occasional inaccuracies.
There is good control of collocation and idiomatic expressions.

Band 7:
The resource is sufficient to allow some flexibility and precision.
There is some awareness of style and collocation, with occasional inaccuracies.
Less common lexical items are attempted.

Band 6:
The resource is generally adequate and appropriate for the task.
There may be some inappropriate word choices and/or awkward usage.
There is some awareness of collocation, though errors are evident.

Band 5:
The resource is limited but minimally adequate for the task.
Errors in word choice and usage are noticeable.
Only basic vocabulary is used, and there may be frequent repetition.

Band 4:
The resource is limited and inadequate for or unrelated to the task.
Errors in word choice and usage may distort meaning.
There may be overuse of limited vocabulary.

Band 3:
The resource is inadequate (which may be due to the candidate’s first language).
Meaning may be unclear or distorted.

Band 2:
The resource is extremely limited with few recognizable words.

Band 1:
Responses of 20 words or fewer are rated at Band 1.
The answer is almost entirely unintelligible.
""",

"Grammatical Range and Accuracy":
"""
Band 9:
A wide range of structures is used with full flexibility and accuracy.
Errors are rare and difficult to spot.

Band 8:
A wide range of structures is flexibly and accurately used.
The majority of sentences are error-free.
Occasional errors occur but do not reduce communication.

Band 7:
A variety of complex structures is used with some flexibility.
Grammatical control is generally good; errors do not reduce communication.
A mix of simple and complex sentence forms is used.

Band 6:
A mix of simple and complex sentence forms is attempted.
Errors may be frequent but rarely cause communication problems.
Some control of grammar and punctuation is evident.

Band 5:
The range of structures is limited and rather repetitive.
Errors are frequent and may cause difficulty for the reader.
Grammatical control is limited.

Band 4:
A very limited range of structures is used.
Structures are repeated frequently.
Grammatical errors may be frequent and cause strain for the reader.

Band 3:
Sentence forms are attempted, but errors in grammar and punctuation predominate.
There may be no clear sentences at all.

Band 2:
There is little or no evidence of sentence forms being used correctly.

Band 1:
Responses of 20 words or fewer are rated at Band 1.
There is no accurate use of grammar.
"""
}