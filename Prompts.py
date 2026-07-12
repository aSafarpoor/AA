"""
Prompt templates used by the simulation.

These match the prompt templates reported in Appendix A of the paper:
  (i)  Prompt_Post      -> post-generation prompt (social-media-style post)
  (ii) Prompt_Influence -> response-generation prompt (opinion update)

Opinions are represented to the model as an integer percentage in [0, 100],
where 0 = fully against the topic and 100 = fully in favor. This numeric
representation was found to give the most stable behaviour across simulations
(see Appendix A).
"""


class Prompt_General:

    @staticmethod
    def Prompt_Post(topic, percent, intensity_level):
        """Post-generation prompt.

        Conditions a short social-media post on the discussion topic and the
        author's opinion strength. `percent` is the opinion mapped to [0, 100];
        `intensity_level` is a human-readable bucket (e.g. "strongly in favor").
        """
        return f"""
You are a social network user.

Topic: Supporting {topic}
Your opinion strength: {intensity_level}, which is equal to {percent} percent (0=against, 100=favor).

Write 1-3 sentences expressing your views.
Match tone to strength:
Low/moderate: casual or uncertain
High: decisive or strong

Do not argue for both sides.
Do not mention numbers or being an AI.
Post:
""".strip()

    @staticmethod
    def Prompt_Influence(topic, percent, stubbornness, post):
        """Response-generation prompt.

        Produces an updated opinion after the agent is exposed to `post`.
        The response is conditioned on the agent's prior opinion (`percent`,
        mapped to [0, 100]) and its `stubbornness` (higher => less change).
        """
        return f"""
You are scrolling a social network.
Topic: {topic}
Your initial opinion strength: {percent} percent (0=against, 100=favor).
Your stubbornness: {stubbornness} (higher = less change)
You see:
\"\"\" Supporting {post} \"\"\"

Higher levels of stubbornness limit movement.
This is a common occurrence.
Return your new opinion (0-100).
Answer with ONE number only.
Answer:
""".strip()
