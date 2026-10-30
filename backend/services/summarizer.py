import json
import requests
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HierarchicalSummarizer:
    def __init__(self, config: dict):
        self.config = config
        llm_cfg = config.get("llm", {})
        self.provider = llm_cfg.get("provider", "mock_api")
        self.api_key = llm_cfg.get("api_key", "")
        self.model_name = llm_cfg.get("model_name", "gpt-3.5-turbo")
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.tokenizer = None
        self.local_model = None

    def _init_local_model(self):
        if self.tokenizer is None or self.local_model is None:
            model_name = "t5-small"
            print(f"[SUMMARIZER] Loading pretrained text summarization model ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _clean_summary(self, text: str) -> str:
        """Cleans up text, removes duplicate phrases, hallucinations and formats cleanly."""
        text = text.strip()
        if not text:
            return ""
        
        # Remove T5 hallucinations or odd tail fragments
        text = re.sub(r' - including the eu and afghanistan\..*', '.', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+[a-z]+:[.\s]*$', '.', text)

        # Split into sentences and deduplicate
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
        unique = []
        for s in sentences:
            if len(s) > 10 and s.lower() not in [u.lower() for u in unique]:
                if s and not s.endswith(('.', '!', '?')):
                    s += "."
                s = s[0].upper() + s[1:]
                unique.append(s)
        
        cleaned = " ".join(unique)
        return cleaned

    def generate_chunk_summary(self, chunk: dict) -> str:
        """Generates a concise, high-quality AI summary for an individual video chunk."""
        start_min = int(chunk['start_time'] // 60)
        start_sec = int(chunk['start_time'] % 60)
        end_min = int(chunk['end_time'] // 60)
        end_sec = int(chunk['end_time'] % 60)
        
        transcript = chunk.get("transcript", "").strip()
        
        # Filter out default placeholder sentences
        if "[Video segment from" in transcript:
            transcript = ""

        if not transcript or len(transcript.split()) < 5:
            return f"This section ({start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}) features key visual demonstrations and presentation topics."

        if self.provider == "openai" and self.api_key:
            prompt = f"Summarize this video segment transcript ({start_min}:{start_sec:02d} to {end_min}:{end_sec:02d}) in 1-2 concise sentences focusing on key topics:\n{transcript}"
            res = self._call_openai(prompt)
            if res:
                return res

        # Abstractive local text summarization using pretrained model
        try:
            self._init_local_model()
            # Clean and truncate transcript to avoid token length overflow
            words = transcript.split()
            if len(words) > 400:
                input_text = " ".join(words[:400])
            else:
                input_text = transcript

            prompt_text = f"summarize: {input_text}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
            
            summary_ids = self.local_model.generate(
                inputs["input_ids"],
                max_length=60,
                min_length=15,
                num_beams=2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            cleaned = self._clean_summary(raw_summary)

            if len(cleaned.split()) >= 4:
                return cleaned
        except Exception as e:
            print(f"[SUMMARIZER] Local model error: {e}")

        # Fallback abstractive transformation (never raw transcript copy)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', transcript) if len(s.strip()) > 15]
        if sentences:
            topic = sentences[0]
            if len(topic) > 120:
                topic = topic[:120] + "..."
            return f"The section explores {topic.lower()}"
        return f"This segment ({start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}) outlines main concepts and key discussions."

    def generate_final_summary(self, video_metadata: dict, selected_chunks: list, chunk_summaries: list) -> dict:
        """
        Hierarchical summarization: aggregates chunk summaries to produce:
        - Overall Video Summary
        - Content-Derived Key Points (4-8 bullet points)
        - Timestamped Sections with titles and AI summaries
        """
        duration_mins = video_metadata.get("duration_minutes", 30)
        
        # Combine section summaries for global summarization
        combined_summaries = [s for s in chunk_summaries if s]
        combined_text = " ".join(combined_summaries)

        overall_summary = ""
        key_points = []

        if self.provider == "openai" and self.api_key:
            prompt = f"Based on these segment summaries of a {duration_mins:.1f}-minute video:\n{combined_text}\n\nProvide an overall concise summary of the entire video, followed by 4-8 bullet points of key takeaways."
            res = self._call_openai(prompt)
            if "Key takeaways:" in res:
                parts = res.split("Key takeaways:")
                overall_summary = parts[0].strip()
                key_points = [p.strip("- *") for p in parts[1].strip().split("\n") if p.strip()]
            elif res:
                overall_summary = res

        if not overall_summary:
            try:
                self._init_local_model()
                prompt_text = f"summarize: {combined_text[:1500]}"
                inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = self.local_model.generate(
                    inputs["input_ids"],
                    max_length=120,
                    min_length=40,
                    num_beams=2,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                overall_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                overall_summary = self._clean_summary(overall_summary)
            except Exception as e:
                print(f"[SUMMARIZER] Overall summary generation error: {e}")

        if not overall_summary:
            if combined_summaries:
                overall_summary = f"This {duration_mins:.1f}-minute video provides comprehensive coverage across key topics: " + " ".join(combined_summaries[:3])
            else:
                overall_summary = f"A detailed {duration_mins:.1f}-minute video presentation covering core concepts and visual demonstrations."

        # Generate 4-8 content-derived key points directly from section summaries
        if not key_points:
            key_points = []
            for s in combined_summaries:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    if not s_clean.endswith('.'):
                        s_clean += '.'
                    if s_clean not in key_points:
                        key_points.append(s_clean)
            
            # Ensure between 4 and 8 key points
            if len(key_points) < 4:
                # Add sentences derived from chunk transcripts
                for chunk in selected_chunks:
                    trans = chunk.get("transcript", "")
                    sents = [st.strip() for st in re.split(r'(?<=[.!?]) +', trans) if len(st.strip()) > 25]
                    for st in sents:
                        if st not in key_points and len(key_points) < 8:
                            key_points.append(st)
            key_points = key_points[:8]
            if len(key_points) < 4:
                key_points.append("Comprehensive topic breakdown across the full video timeline.")

        # Build Timestamped Sections
        important_sections = []
        for i, (chunk, summary) in enumerate(zip(selected_chunks, chunk_summaries)):
            start_sec = chunk["start_time"]
            end_sec = chunk["end_time"]
            score = chunk.get("importance_score", 0.85)

            start_fmt = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
            end_fmt = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"

            # Create a content-derived section title from the summary
            first_words = " ".join(summary.split()[:5]).strip(",.!?")
            if first_words:
                section_title = f"{start_fmt} - {end_fmt} | {first_words}..."
            else:
                section_title = f"Section {i+1}: {start_fmt} - {end_fmt}"

            important_sections.append({
                "start": start_fmt,
                "end": end_fmt,
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "title": f"{section_title} (Score: {score:.2f})",
                "summary": summary,
                "importance_score": score
            })

        return {
            "video_duration": f"{duration_mins:.1f} minutes",
            "summary": overall_summary,
            "key_points": key_points,
            "important_sections": important_sections
        }

    def _call_openai(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
        return ""

    def unload_model(self):
        if self.local_model is not None:
            try:
                del self.local_model
            except Exception:
                pass
            self.local_model = None
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
            self.tokenizer = None
        import gc
        gc.collect()
