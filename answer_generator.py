import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        pass
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer from context chunks."""
        if not context_chunks:
            return "No relevant information found in the document."
        
        # Combine context
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        # Generate answer based on question type
        return self._extract_answer(question, context)
    
    def _extract_answer(self, question: str, context: str) -> str:
        """Extract answer using pattern matching."""
        question_lower = question.lower()
        
        # Grace period questions
        if "grace period" in question_lower:
            return self._extract_grace_period(context)
        
        # Waiting period questions
        elif "waiting period" in question_lower:
            return self._extract_waiting_period(context)
        
        # Coverage questions
        elif any(word in question_lower for word in ["covered", "coverage", "benefit"]):
            return self._extract_coverage_info(question, context)
        
        # Maternity questions
        elif "maternity" in question_lower:
            return self._extract_maternity_info(context)
        
        # Room rent questions
        elif "room rent" in question_lower or "icu" in question_lower:
            return self._extract_room_rent_info(context)
        
        # General extraction
        else:
            return self._extract_general_info(question, context)
    
    def _extract_grace_period(self, text: str) -> str:
        """Extract grace period information."""
        patterns = [
            r"grace period of (\d+) days?",
            r"(\d+) days? grace period",
            r"grace.*?(\d+) days?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                days = match.group(1)
                return f"A grace period of {days} days is provided for premium payment."
        
        # Look for sentences mentioning grace period
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if "grace" in sentence.lower() and "period" in sentence.lower():
                return sentence.strip()
        
        return "Grace period information not found in the document."
    
    def _extract_waiting_period(self, text: str) -> str:
        """Extract waiting period information."""
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if "waiting" in sentence.lower() and ("months" in sentence.lower() or "years" in sentence.lower()):
                # Extract time period
                time_match = re.search(r'(\d+)\s*(months?|years?)', sentence, re.IGNORECASE)
                if time_match:
                    period = time_match.group(1)
                    unit = time_match.group(2)
                    return f"There is a waiting period of {period} {unit}. {sentence.strip()}"
                return sentence.strip()
        
        return "Waiting period information not found in the document."
    
    def _extract_coverage_info(self, question: str, text: str) -> str:
        """Extract coverage information."""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', text)
        
        best_match = ""
        max_overlap = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = sentence.strip()
        
        return best_match if best_match else "Coverage information not found in the document."
    
    def _extract_maternity_info(self, text: str) -> str:
        """Extract maternity coverage information."""
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if "maternity" in sentence.lower():
                return sentence.strip()
        
        return "Maternity coverage information not found in the document."
    
    def _extract_room_rent_info(self, text: str) -> str:
        """Extract room rent information."""
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if ("room rent" in sentence.lower() or "icu" in sentence.lower()) and "%" in sentence:
                return sentence.strip()
        
        return "Room rent information not found in the document."
    
    def _extract_general_info(self, question: str, text: str) -> str:
        """Extract general information based on keyword matching."""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', text)
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue
            
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap >= 2:
                scored_sentences.append((sentence.strip(), overlap))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return scored_sentences[0][0]
        
        # Return first meaningful sentence if no good match
        for sentence in sentences:
            if len(sentence.strip()) > 50:
                return sentence.strip()
        
        return "Relevant information not found in the document."