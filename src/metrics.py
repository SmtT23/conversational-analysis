"""
Compute linguistic and semantic metrics for conversational analysis.
Includes sentiment, complexity, coherence, and diversity measures.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ConversationMetrics:
    """Compute linguistic metrics for conversations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize metrics computer.

        Args:
            model_name: HuggingFace sentence transformer model
        """
        self.embedding_model = SentenceTransformer(model_name)

    def sentiment_score(self, text: str) -> float:
        """
        Compute sentiment polarity (-1 to 1).

        Args:
            text: Input text

        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def optimism_score(self, text: str) -> float:
        """
        Compute optimism index based on positive language cues.

        Args:
            text: Input text

        Returns:
            Optimism score between 0 and 1
        """
        positive_words = {
            "great": 1.0,
            "excellent": 1.0,
            "wonderful": 1.0,
            "exciting": 0.9,
            "potential": 0.8,
            "opportunity": 0.8,
            "progress": 0.7,
            "improve": 0.7,
            "achieve": 0.7,
            "possible": 0.6,
            "hope": 0.7,
            "optimistic": 1.0,
            "innovative": 0.8,
            "transform": 0.8,
            "success": 0.8,
        }

        negative_words = {
            "fail": -0.8,
            "problem": -0.5,
            "challenge": -0.3,
            "difficult": -0.4,
            "impossible": -1.0,
            "risk": -0.4,
            "constraint": -0.3,
        }

        text_lower = text.lower()
        score = 0.0
        count = 0

        for word, weight in positive_words.items():
            if word in text_lower:
                score += weight
                count += 1

        for word, weight in negative_words.items():
            if word in text_lower:
                score += weight
                count += 1

        if count == 0:
            return 0.5  # Neutral if no key words found

        # Normalize to 0-1 range
        normalized = (score / count + 1) / 2
        return max(0.0, min(1.0, normalized))

    def lexical_diversity(self, text: str) -> float:
        """
        Compute type-token ratio (lexical diversity).

        Args:
            text: Input text

        Returns:
            Ratio of unique words to total words (0-1)
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) == 0:
            return 0.0
        unique_words = len(set(words))
        return unique_words / len(words)

    def vocabulary_richness(self, text: str) -> int:
        """
        Count unique vocabulary items.

        Args:
            text: Input text

        Returns:
            Number of unique words
        """
        words = re.findall(r"\b\w+\b", text.lower())
        return len(set(words))

    def sentence_count(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r"[.!?]+", text)
        return len([s for s in sentences if s.strip()])

    def average_sentence_length(self, text: str) -> float:
        """Compute average words per sentence."""
        words = re.findall(r"\b\w+\b", text.lower())
        sentences = self.sentence_count(text)
        if sentences == 0:
            return 0.0
        return len(words) / sentences

    def semantic_coherence(self, texts: List[str]) -> float:
        """
        Compute average semantic similarity between consecutive utterances.

        Args:
            texts: List of text segments (in order)

        Returns:
            Average cosine similarity between consecutive embeddings (0-1)
        """
        if len(texts) < 2:
            return 0.5

        embeddings = self.embedding_model.encode(texts)
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0.5

    def semantic_complexity(self, text: str) -> float:
        """
        Estimate complexity based on vocabulary and sentence structure.

        Args:
            text: Input text

        Returns:
            Complexity score (0-1, higher = more complex)
        """
        words = re.findall(r"\b\w+\b", text.lower())
        sentences = self.sentence_count(text)

        if len(words) == 0:
            return 0.0

        # Factor 1: Unique word ratio (lexical diversity)
        diversity = self.lexical_diversity(text)

        # Factor 2: Average sentence length (more words per sentence = more complex)
        avg_sent_len = self.average_sentence_length(text)
        normalized_sent_len = min(avg_sent_len / 20.0, 1.0)  # Cap at 20 words

        # Factor 3: Word length (longer words often more complex)
        avg_word_length = np.mean([len(w) for w in words])
        normalized_word_len = min(avg_word_length / 10.0, 1.0)  # Cap at 10 chars

        complexity = (diversity + normalized_sent_len + normalized_word_len) / 3
        return min(complexity, 1.0)

    def analyze_conversation(self, turns: List[Dict]) -> Dict:
        """
        Compute all metrics for a conversation.

        Args:
            turns: List of turn dictionaries with 'text' and 'speaker' keys

        Returns:
            Dictionary of computed metrics
        """
        texts = [turn["text"] for turn in turns]
        assistant_texts = [turn["text"] for turn in turns if turn.get("speaker") == "assistant"]

        return {
            "num_turns": len(turns),
            "num_speaker_turns": len([t for t in turns if t.get("speaker") == "user"]),
            "num_assistant_turns": len(assistant_texts),
            # Sentiment metrics
            "avg_sentiment": np.mean([self.sentiment_score(t) for t in texts]),
            "avg_optimism": np.mean([self.optimism_score(t) for t in texts]),
            "assistant_avg_optimism": np.mean(
                [self.optimism_score(t) for t in assistant_texts]
            ) if assistant_texts else 0.0,
            # Diversity metrics
            "avg_lexical_diversity": np.mean(
                [self.lexical_diversity(t) for t in texts]
            ),
            "total_unique_words": len(set(w for t in texts for w in re.findall(r"\b\w+\b", t.lower()))),
            # Complexity metrics
            "avg_complexity": np.mean([self.semantic_complexity(t) for t in texts]),
            "avg_sentence_length": np.mean(
                [self.average_sentence_length(t) for t in texts]
            ),
            # Coherence metrics
            "semantic_coherence": self.semantic_coherence(texts),
            "assistant_semantic_coherence": self.semantic_coherence(assistant_texts)
            if len(assistant_texts) > 1
            else 0.5,
        }
