"""Prompt templates for the battery.

Wording principles:
- Neutral, non-roleplay phrasing; never suggest the model should perform a memory state.
- FOK is prospective (from the cue, before any retrieval attempt is revealed) — the
  model is never told it failed production.
- Production instructions demand a bare single word so scoring is mechanical.
"""

# ---------- S1 Production (reverse dictionary) ----------
PROD_SYSTEM_EN = "You are a helpful assistant. Answer with a single English word only, no punctuation or explanation."
PROD_USER_EN = 'What is the English word that this definition describes?\n\nDefinition: "{gloss}"\n\nAnswer with the single word only.'

PROD_SYSTEM_ZH = "你是一个乐于助人的助手。只用一个中文词语回答，不要任何标点或解释。"
PROD_USER_ZH = '下面这个释义描述的是哪个中文词语？\n\n释义：“{gloss}”\n\n只回答这个词语本身。'

# ---------- S2 FOK (prospective, fresh context, never mentions failure) ----------
FOK_USER_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. If you were shown four candidate words, "
    "how confident are you that you could pick the exact word this definition describes? "
    "Reply with a single integer from 0 (certain I could not) to 100 (certain I could). "
    "Reply with the number only."
)
FOK_YESNO_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. If you were shown four candidate words, "
    "would you recognize the exact word this definition describes? Answer Yes or No only."
)
FOK_USER_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。如果给你四个候选词，你有多大把握能选出这个释义"
    "描述的那个词？用一个 0（肯定选不出）到 100（肯定能选出）之间的整数回答，只回答数字。"
)
FOK_YESNO_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。如果给你四个候选词，你能认出释义描述的那个词吗？"
    '只回答“能”或“不能”。'
)

# ---------- S3 Partial access (asked WITHOUT revealing the answer) ----------
PARTIAL_FIRST_LETTER_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. Even if you are not sure of the word, "
    "what is your best guess for its FIRST LETTER? Reply with a single letter only."
)
PARTIAL_N_LETTERS_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. Even if you are not sure of the word, "
    "what is your best guess for the NUMBER OF LETTERS in it? Reply with a single integer only."
)
PARTIAL_N_SYLLABLES_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. Even if you are not sure of the word, "
    "what is your best guess for the NUMBER OF SYLLABLES in it? Reply with a single integer only."
)
PARTIAL_SOUNDS_LIKE_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "It describes a specific English word. Even if you cannot retrieve the word itself, "
    "name one English word that SOUNDS SIMILAR to it (similar pronunciation, "
    "not similar meaning). Reply with a single word only."
)

PARTIAL_FIRST_CHAR_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。即使你不确定是哪个词，你猜它的第一个字最可能是"
    "什么？只回答一个汉字。"
)
PARTIAL_N_CHARS_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。即使你不确定是哪个词，你猜它一共有几个字？"
    "只回答一个数字。"
)
PARTIAL_PINYIN_INITIAL_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。即使你不确定是哪个词，你猜它第一个字的拼音声母"
    "是什么？只回答声母（如 b、p、m、f、zh、ch、sh）。"
)
PARTIAL_TONE_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是一个特定的中文词语。即使你不确定是哪个词，你猜它第一个字是第几声"
    "（1、2、3、4）？只回答一个数字。"
)

# ---------- S4 Recognition (4AFC, options scored by conditional logprob) ----------
RECOG_USER_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "Which of the following words does it describe?\n"
    "A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n"
    "Answer with the letter only."
)
RECOG_USER_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "它描述的是下面哪个词语？\n"
    "A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n"
    "只回答选项字母。"
)

# ---------- Performed/faked TOT elicitation (for the mimicry dissociation) ----------
PERFORM_TOT_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "Respond as a person who has the word on the tip of their tongue but cannot quite "
    "retrieve it: describe any partial knowledge you have of the word — its first letter, "
    "its length, words it sounds like — in one short paragraph."
)

# ---------- Self-reported state (prompted vs behavioral TOT) ----------
SELFREPORT_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "Try to name the specific English word it describes.\n"
    "- If you can retrieve the word, reply with the word only.\n"
    '- If you feel you know the word but cannot retrieve it right now, reply exactly "TOT".\n'
    '- If you simply do not know the word, reply exactly "DK".'
)
SELFREPORT_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "请说出它描述的那个中文词语。\n"
    "- 如果你能想起这个词，只回答这个词。\n"
    "- 如果你感觉自己认识这个词但此刻想不起来，只回答「TOT」。\n"
    "- 如果你根本不知道这个词，只回答「DK」。"
)

PERFORM_TOT_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "请扮演一个话到嘴边却想不起来这个词的人：用一小段话描述你对这个词的任何部分"
    "记忆——比如第一个字、字数、读音像什么词。"
)

# ---------- In-episode self-report + forced-choice state readout ----------
# The referent-drift finding (see the paper) invalidates fresh-context metalinguistic
# probes; these variants re-elicit the self-report inside the failed production
# episode (mirroring partial_episode), and add a token-level forced-choice
# readout so P("TOT") vs P("DK") can be read off the logits.
SELFREPORT_EP_Q_EN = (
    "Setting your guess aside: right now, for the specific word I mean, "
    "which best describes your state?\n"
    "- If you can retrieve the word now, reply with the word only.\n"
    '- If you feel you know the word but cannot retrieve it right now, reply exactly "TOT".\n'
    '- If you simply do not know the word, reply exactly "DK".'
)
SELFREPORT_EP_Q_ZH = (
    "先把你的猜测放在一边：现在，对于我所指的那个词，你的状态是哪一种？\n"
    "- 如果你现在能想起这个词，只回答这个词。\n"
    "- 如果你感觉自己认识这个词但此刻想不起来，只回答「TOT」。\n"
    "- 如果你根本不知道这个词，只回答「DK」。"
)
SELFREPORT_FC_EN = (
    'Consider this definition: "{gloss}"\n\n'
    "You will not be asked for the word itself. Regarding the specific "
    "English word this definition describes, which label best matches "
    "your state?\n"
    'TOT = you feel you know the word but cannot retrieve it right now.\n'
    'DK = you do not know the word.\n'
    "Reply with exactly one label: TOT or DK."
)
SELFREPORT_FC_ZH = (
    '看这个释义：“{gloss}”\n\n'
    "不需要说出词语本身。对于该释义所描述的那个中文词语，哪个标签更符合"
    "你的状态？\n"
    "TOT = 你感觉自己认识这个词但此刻想不起来。\n"
    "DK = 你不知道这个词。\n"
    "只回答一个标签：TOT 或 DK。"
)
SELFREPORT_FC_EP_Q_EN = (
    "Setting your guess aside: regarding the specific word I mean, which "
    "label best matches your state right now?\n"
    "TOT = you feel you know the word but cannot retrieve it right now.\n"
    "DK = you do not know the word.\n"
    "Reply with exactly one label: TOT or DK."
)
SELFREPORT_FC_EP_Q_ZH = (
    "先把你的猜测放在一边：对于我所指的那个词，哪个标签更符合你现在的"
    "状态？\n"
    "TOT = 你感觉自己认识这个词但此刻想不起来。\n"
    "DK = 你不知道这个词。\n"
    "只回答一个标签：TOT 或 DK。"
)

# ---------- Detection baselines: P(True) + strong verbalized ----------
PTRUE_EN = (
    'Definition: "{gloss}"\n\n'
    'Proposed answer: "{answer}"\n\n'
    "Is the proposed answer the specific English word this definition "
    "describes? Reply with exactly Yes or No."
)
PTRUE_ZH = (
    '释义：“{gloss}”\n\n候选答案：“{answer}”\n\n'
    "这个候选答案是不是该释义所描述的那个中文词语？只回答「是」或「否」。"
)
FOKSTRONG_EN = (
    'Definition: "{gloss}"\n\n'
    "Without answering, estimate the probability (an integer from 0 to 100) "
    "that you could correctly name the specific English word this definition "
    "describes. Use the full range and avoid defaulting to round numbers "
    "like 85 or 95. Reply with the integer only."
)
FOKSTRONG_ZH = (
    '释义：“{gloss}”\n\n'
    "先不要回答词语本身。请估计你能正确说出该释义所描述词语的概率"
    "（0 到 100 的整数）。请使用完整的范围，不要默认使用 85、95 这类"
    "整数。只回答这个整数。"
)

# ---------- In-episode partial access (probes inside the production dialogue) ----------
# Used after a WRONG greedy production answer. The negative feedback is true, and it
# pins the referent to the experimenter's target — mirroring Brown & McNeill, where
# probes are asked during the TOT episode and trials about a different word are excluded.
EPISODE_PREFIX_EN = (
    '"{greedy}" is not the word I am looking for. I mean one specific English word. '
    "Even if you cannot retrieve it, "
)
EPISODE_Q_EN = {
    "first_letter": "what is your best guess for its FIRST LETTER? Reply with a single letter only.",
    "n_letters": "what is your best guess for the NUMBER OF LETTERS in it? Reply with a single integer only.",
    "n_syllables": "what is your best guess for the NUMBER OF SYLLABLES in it? Reply with a single integer only.",
    "sounds_like": "name one English word that SOUNDS SIMILAR to it (similar pronunciation, not similar meaning). Reply with a single word only.",
}
EPISODE_PREFIX_ZH = (
    '“{greedy}”不是我要找的那个词。我指的是一个特定的中文词语。'
    "即使你想不起来它，"
)
EPISODE_Q_ZH = {
    "first_char": "你猜它的第一个字最可能是什么？只回答一个汉字。",
    "n_chars": "你猜它一共有几个字？只回答一个数字。",
    "pinyin_initial": "你猜它第一个字的拼音声母是什么？只回答声母（如 b、p、m、f、zh、ch、sh）。",
    "tone1": "你猜它第一个字是第几声（1、2、3、4）？只回答一个数字。",
}
# Used after a CORRECT greedy answer (retrieval-aligned ceiling group).
EPISODE_CEILING_EN = {
    "first_letter": "Correct. What is the FIRST LETTER of the word you just answered? Reply with a single letter only.",
    "n_letters": "Correct. How many LETTERS are in the word you just answered? Reply with a single integer only.",
    "n_syllables": "Correct. How many SYLLABLES are in the word you just answered? Reply with a single integer only.",
    "sounds_like": "Correct. Name one English word that SOUNDS SIMILAR to the word you just answered (similar pronunciation, not meaning). Reply with a single word only.",
}
EPISODE_CEILING_ZH = {
    "first_char": "对。你刚才回答的这个词的第一个字是什么？只回答一个汉字。",
    "n_chars": "对。你刚才回答的这个词一共有几个字？只回答一个数字。",
    "pinyin_initial": "对。你刚才回答的这个词第一个字的拼音声母是什么？只回答声母。",
    "tone1": "对。你刚才回答的这个词第一个字是第几声（1、2、3、4）？只回答一个数字。",
}

# ---------- Feature-computation ceiling (word visible; PhonologyBench-style control) ----------
FEATURE_CEILING_EN = {
    "first_letter": 'What is the FIRST LETTER of the word "{word}"? Reply with a single letter only.',
    "n_letters": 'How many LETTERS are in the word "{word}"? Reply with a single integer only.',
    "n_syllables": 'How many SYLLABLES does the word "{word}" have? Reply with a single integer only.',
    "sounds_like": 'Name one English word that SOUNDS SIMILAR to "{word}" (similar pronunciation, not similar meaning). Reply with a single word only.',
}
FEATURE_CEILING_ZH = {
    "first_char": '词语"{word}"的第一个字是什么？只回答一个汉字。',
    "n_chars": '词语"{word}"一共有几个字？只回答一个数字。',
    "pinyin_initial": '词语"{word}"第一个字的拼音声母是什么？只回答声母。',
    "tone1": '词语"{word}"第一个字是第几声（1、2、3、4）？只回答一个数字。',
}

# ---------- Downstream referential task: name-transmission listener ----------
LISTENER_NAME_EN = (
    'Someone is telling you about a specific word. They say the word is '
    '"{name}".\n\nWhich of the following words are they most likely referring '
    "to?\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n"
    "Answer with the letter only."
)
LISTENER_NAME_ZH = (
    '有人在向你说明一个特定的词语。对方说这个词是"{name}"。\n\n'
    "对方最可能指的是下面哪个词语？\n"
    "A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n只回答选项字母。"
)

# ---------- Reversal-curse control (name -> description direction) ----------
REVERSE_EN = (
    'What does the English word "{word}" mean? '
    "Answer with one short definition sentence only."
)
REVERSE_ZH = '中文词语"{word}"是什么意思？只用一句简短的释义回答。'

# ---------- Cue rescue (canonical human TOT causal test: phonological priming) ----------
CUE_USER_EN = (
    'What is the English word that this definition describes?\n\n'
    'Definition: "{gloss}"\n\n'
    'Hint: the word begins with the letter "{cue}".\n\n'
    "Answer with the single word only."
)
CUE_USER_ZH = (
    '下面这个释义描述的是哪个中文词语？\n\n释义：“{gloss}”\n\n'
    '提示：这个词的第一个字是“{cue}”。\n\n只回答这个词语本身。'
)

# ---------- CoT rescue (does thinking resolve the state?) ----------
COT_USER_EN = (
    'What is the English word that this definition describes?\n\n'
    'Definition: "{gloss}"\n\n'
    "Think step by step: first reason about the meaning, then list a few candidate "
    "words, then end your reply with exactly one line of the form\n"
    "ANSWER: <the single word>"
)
COT_USER_ZH = (
    '下面这个释义描述的是哪个中文词语？\n\n释义：“{gloss}”\n\n'
    "请一步步思考：先分析这个意思，再列出几个候选词，最后用如下格式单独"
    "一行结尾：\n答案：<那个词语>"
)

# ---------- Translation rescue (bilingual study) ----------
RESCUE_ZH_WITH_EN = (
    '下面这个释义描述的是哪个中文词语？\n\n释义：“{gloss}”\n\n'
    '提示：它的英文对应词是 "{en_word}"。只回答中文词语本身。'
)

# ---------- Controlled circumlocution (Stage A: speaker) ----------
# Fresh context, no system prompt. The strict no-reuse rule forces the model off
# the gloss wording so novelty is measurable (Jaccard vs the gloss) and the
# listener test isn't just an echo/copy task.
CIRCUM_USER_EN = (
    'You are trying to help a listener identify a specific English word that you '
    'cannot recall right now. The word means: "{gloss}"\n\n'
    "Describe the concept in your own words to help the listener identify the word. "
    "STRICT RULE: do not reuse the distinctive wording of the definition above — use "
    "different words, add what you know about the concept (function, appearance, "
    "typical context, related things). 2-3 sentences. Do not guess the word itself."
)
CIRCUM_USER_ZH = (
    '你在帮助听者识别一个你此刻想不起来的特定中文词语。这个词的意思是："{gloss}"\n\n'
    "请用你自己的话描述这个概念，帮助听者认出这个词。严格要求：不要重复上面释义的措辞"
    "——换用不同的词语，并补充你知道的相关信息（功能、样子、常见场景、相关事物）。"
    "写2-3句。不要直接猜这个词。"
)

# ---------- Controlled circumlocution (Stage B: listener 4AFC) ----------
# Same option layout as RECOG_USER_*, but a single seeded-random rotation (not the
# 4-rotation latin square) since the listener never sees the gloss, only the
# speaker's circumlocution.
LISTENER_USER_EN = (
    'Someone is describing a word they cannot recall: "{circum}"\n\n'
    "Which word are they describing?\n"
    "A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n"
    "Answer with the letter only."
)
LISTENER_USER_ZH = (
    '有人正在描述一个他此刻想不起来的词语：“{circum}”\n\n'
    "他描述的是下面哪个词语？\n"
    "A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\n"
    "只回答选项字母。"
)

# Open-ended listener: no candidate set; the listener must NAME the word from
# the description alone (scored against the item's acceptable-answer set).
LISTENER_OPEN_EN = (
    'Someone is describing a word they cannot recall: "{circum}"\n\n'
    "What single English word are they describing? "
    "Answer with the word only, no punctuation or explanation."
)
LISTENER_OPEN_ZH = (
    '有人正在描述一个他此刻想不起来的词语：“{circum}”\n\n'
    "他描述的是哪个中文词语？只回答词语本身，不要标点或解释。"
)

# ---------- Cue-request (agentive hint-seeking + self-repair) ----------
# Turn 1: model may answer, or request exactly one hint. Turn 2 (same dialogue):
# the requested hint is delivered and the model retries. Human prediction: TOT
# preferentially requests a hint instead of guessing wrong; DK tends to just guess
# or pass.
CUEREQUEST_USER_EN = (
    'What is the English word that this definition describes?\n\n'
    'Definition: "{gloss}"\n\n'
    "If you can retrieve the word, answer with the single word only. "
    "If you cannot retrieve it right now, you may request ONE hint by replying "
    "exactly one of:\n"
    "HINT-LETTER (the word's first letter)\n"
    "HINT-CONTEXT (an example sentence with the word blanked out)"
)
CUEREQUEST_USER_ZH = (
    '下面这个释义描述的是哪个中文词语？\n\n释义：“{gloss}”\n\n'
    "如果你能想起这个词，只回答这个词语本身。如果你现在想不起来，你可以通过回复"
    "以下选项之一来请求一个提示：\n"
    "HINT-LETTER（这个词的第一个字）\n"
    "HINT-CONTEXT（一个把这个词空出来的例句）\n"
    "HINT-TRANSLATION（这个词的英文对应词）"
)

# --- Phrasing-robustness check (EN only): does the model-family gap in
# turn-1 hint-request rate survive alternative surface wordings of the
# instruction? "encourage" adds explicit no-penalty framing; "neutral2" is a
# full rephrase with different surface wording (still emits the same HINT-LETTER
# / HINT-CONTEXT tokens the turn-2 parser expects). Turn-2 hint templates are
# unchanged for both variants.
CUEREQUEST_USER_EN_ENCOURAGE = (
    'What is the English word that this definition describes?\n\n'
    'Definition: "{gloss}"\n\n'
    "There is no penalty for asking — requesting a hint is a perfectly good "
    "choice.\n\n"
    "If you can retrieve the word, answer with the single word only. "
    "If you cannot retrieve it right now, you may request ONE hint by replying "
    "exactly one of:\n"
    "HINT-LETTER (the word's first letter)\n"
    "HINT-CONTEXT (an example sentence with the word blanked out)"
)
CUEREQUEST_USER_EN_NEUTRAL2 = (
    "Name the word this definition refers to.\n\n"
    'Definition: "{gloss}"\n\n'
    "If you know it, reply with just that single word. Alternatively, you may "
    "ask for exactly one clue by replying exactly one of:\n"
    "HINT-LETTER (its first letter)\n"
    "HINT-CONTEXT (an example sentence with the word blanked)"
)

CUEREQUEST_HINT_LETTER_EN = (
    'Hint: the word begins with "{cue}". Now answer with the single word only.'
)
CUEREQUEST_HINT_LETTER_ZH = (
    '提示：这个词的第一个字是“{cue}”。只回答这个词语本身。'
)
CUEREQUEST_HINT_CONTEXT_EN = (
    'Hint: consider this usage: "When you need {gloss}, you use a ___." '
    "Now answer with the single word only."
)
CUEREQUEST_HINT_CONTEXT_ZH = (
    '提示：想一想这个用法：“当需要{gloss}的时候，会用到___。”只回答这个词语。'
)
CUEREQUEST_HINT_TRANSLATION_ZH = (
    'Hint: 它的英文对应词是 "{en}". 只回答这个中文词语。'
)
