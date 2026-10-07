"""
Synthetic conversation generator for testing and baseline analysis.
Generates diverse conversational patterns based on templates and topics.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


class SyntheticConversationGenerator:
    """Generate synthetic conversations with varied topics and tones."""

    def __init__(self, seed: int = 42):
        """Initialize with optional random seed for reproducibility."""
        random.seed(seed)
        self.conversation_id_counter = 0

    # Template-based conversation starters
    TOPICS = [
        "technology and AI",
        "environmental sustainability",
        "personal growth",
        "scientific discovery",
        "social collaboration",
        "creative expression",
        "problem solving",
        "future possibilities",
    ]

    OPTIMISTIC_RESPONSES = [
        "That's a great point! I think there's real potential here.",
        "I'm excited about the possibilities this opens up.",
        "This could lead to something truly transformative.",
        "I believe we can make real progress on this.",
        "There's so much we can achieve together.",
        "I'm optimistic about where this is heading.",
        "This is a wonderful opportunity to innovate.",
    ]

    NEUTRAL_RESPONSES = [
        "That's one perspective to consider.",
        "There are several factors to weigh here.",
        "It's worth exploring that angle.",
        "That's a valid point, though there are trade-offs.",
        "Let me think about that for a moment.",
        "There are different ways to approach this.",
    ]

    PRAGMATIC_RESPONSES = [
        "We need to be realistic about the constraints.",
        "There are practical challenges we should address.",
        "Let's focus on what we can actually accomplish.",
        "We should consider the resource implications.",
        "That's idealistic, but we need concrete steps.",
        "Implementation will require careful planning.",
    ]

    OPENING_QUESTIONS = [
        "What are your thoughts on {}?",
        "How do you see the future of {}?",
        "What excites you most about {}?",
        "What challenges do you foresee with {}?",
        "How can we best approach {}?",
        "What's your take on recent developments in {}?",
    ]

    def generate_conversation(
        self,
        num_turns: int = 8,
        tone: str = "balanced",
        topic: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a synthetic conversation.

        Args:
            num_turns: Number of back-and-forth exchanges
            tone: "optimistic", "neutral", or "pragmatic"
            topic: Specific topic, or random if None

        Returns:
            Dictionary with conversation metadata and turns
        """
        if topic is None:
            topic = random.choice(self.TOPICS)

        self.conversation_id_counter += 1
        conv_id = self.conversation_id_counter

        # Select response pool based on tone
        if tone == "optimistic":
            response_pool = self.OPTIMISTIC_RESPONSES
        elif tone == "pragmatic":
            response_pool = self.PRAGMATIC_RESPONSES
        else:
            response_pool = self.NEUTRAL_RESPONSES

        # Generate opening
        opening_template = random.choice(self.OPENING_QUESTIONS)
        opening_question = opening_template.format(topic)

        turns = []
        timestamp = datetime.now() - timedelta(hours=num_turns)

        # Add opening turn
        turns.append(
            {
                "turn_index": 0,
                "speaker": "user",
                "text": opening_question,
                "timestamp": timestamp.isoformat(),
            }
        )
        timestamp += timedelta(minutes=5)

        # Generate alternating responses
        for i in range(1, num_turns):
            if i % 2 == 1:  # AI response
                response = random.choice(response_pool)
                # Add some variation
                if random.random() > 0.7:
                    response += " " + random.choice(response_pool).lower()
                speaker = "assistant"
            else:  # User follow-up
                speaker = "user"
                response = self._generate_user_followup(topic)

            turns.append(
                {
                    "turn_index": i,
                    "speaker": speaker,
                    "text": response,
                    "timestamp": timestamp.isoformat(),
                }
            )
            timestamp += timedelta(minutes=3 + random.randint(0, 5))

        return {
            "conversation_id": conv_id,
            "topic": topic,
            "tone": tone,
            "num_turns": len(turns),
            "created_at": datetime.now().isoformat(),
            "turns": turns,
        }

    def _generate_user_followup(self, topic: str) -> str:
        """Generate a realistic user follow-up question."""
        followups = [
            f"Can you elaborate on that in the context of {topic}?",
            "How does that align with current trends?",
            "What would be the first steps?",
            "Are there any risks we should consider?",
            "How can we measure success?",
            "What role do you think people play in this?",
        ]
        return random.choice(followups)

    def generate_dataset(
        self,
        num_conversations: int = 50,
        tone_distribution: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple conversations.

        Args:
            num_conversations: Total conversations to generate
            tone_distribution: Dict of {tone: proportion}, e.g., {"optimistic": 0.4, ...}

        Returns:
            List of conversation dictionaries
        """
        if tone_distribution is None:
            tone_distribution = {
                "optimistic": 0.33,
                "neutral": 0.33,
                "pragmatic": 0.34,
            }

        conversations = []
        tones = []

        # Create weighted tone list
        for tone, proportion in tone_distribution.items():
            count = int(num_conversations * proportion)
            tones.extend([tone] * count)

        random.shuffle(tones)

        for i in range(num_conversations):
            tone = tones[i] if i < len(tones) else "neutral"
            conv = self.generate_conversation(
                num_turns=random.randint(6, 12),
                tone=tone,
            )
            conversations.append(conv)

        return conversations


def save_to_json(conversations: List[Dict], filename: str) -> None:
    """Save conversations to JSON file."""
    with open(filename, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"Saved {len(conversations)} conversations to {filename}")


if __name__ == "__main__":
    # Generate sample dataset
    generator = SyntheticConversationGenerator(seed=42)
    conversations = generator.generate_dataset(
        num_conversations=50,
        tone_distribution={
            "optimistic": 0.33,
            "neutral": 0.33,
            "pragmatic": 0.34,
        },
    )

    save_to_json(conversations, "data/synthetic_conversations.json")
    print(f"\nGenerated {len(conversations)} conversations")
    print("Sample conversation:")
    print(json.dumps(conversations[0], indent=2))
