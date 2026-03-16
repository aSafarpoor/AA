class Prompt_General:

    @staticmethod
    def Prompt_Post(topic, percent, intensity_level):
        return f"""
You are a social media user.

Topic: {topic}
Opinion strength: {intensity_level}

Write 1–3 sentences expressing your view.
Match tone to strength:
- Low/moderate → casual or uncertain
- High → decisive or strong
Do not argue both sides.
Do not mention numbers or being an AI.

Post:
""".strip()


    @staticmethod
    def Prompt_Influence(topic, percent, stubbornness, post, is_friend_flag, responsiveness=0.5):
        return f"""
You are scrolling social media.

Topic: {topic}
Previous opinion: {percent} (0=against, 100=favor)
Stubbornness: {stubbornness} (higher = less change)

You see:

\"\"\"{post}\"\"\" 

Higher stubbornness limits movement.
Remaining unchanged is common.

Return your new opinion (0–100).
Answer with ONE number only.

Answer:
""".strip()

