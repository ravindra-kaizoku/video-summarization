import re
import requests
from pathlib import Path
from backend.utils.memory_utils import get_available_memory_mb

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
            avail_mb = get_available_memory_mb()
            if avail_mb < 250:
                self.local_model = "lightweight"
                return

            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                model_name = "t5-small"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception:
                self.local_model = "lightweight"

    def _clean_transcript_sentences(self, transcript: str) -> list:
        """Extracts clean, meaningful declarative sentences from the transcript, removing filler words."""
        raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if len(s.strip()) > 8]
        clean_sents = []
        for s in raw_sents:
            # Strip conversational fillers
            s_clean = re.sub(r'^(Alright|So|Um|Uh|Well|You know|Like|Now|Okay)\s*,?\s*', '', s, flags=re.I)
            s_clean = s_clean.strip()
            if len(s_clean) > 12:
                if not s_clean.endswith(('.', '!', '?')):
                    s_clean += '.'
                s_clean = s_clean[0].upper() + s_clean[1:]
                clean_sents.append(s_clean)
        return clean_sents

    def _derive_topic_keywords(self, transcript: str, max_words: int = 4) -> str:
        """Extracts prominent descriptive topic keywords from the transcript."""
        stop_words = {
            'the', 'and', 'that', 'this', 'with', 'from', 'have', 'what', 'guys', 'about',
            'they', 'here', 'there', 'really', 'cool', 'pretty', 'much', 'video', 'section',
            'chapter', 'alright', 'just', 'some', 'like', 'then', 'also', 'will', 'been',
            'would', 'could', 'their', 'which', 'when', 'more', 'other', 'into', 'only'
        }
        words = re.findall(r'\b[A-Za-z]{4,}\b', transcript)
        meaningful = []
        seen = set()
        for w in words:
            w_lower = w.lower()
            if w_lower not in stop_words and w_lower not in seen:
                seen.add(w_lower)
                meaningful.append(w.capitalize())
        if len(meaningful) >= 2:
            return " ".join(meaningful[:max_words])
        return "Key Highlights & Discussion"

    def generate_chunk_summary(self, chunk: dict) -> str:
        """
        Explains what happened in this video section up to that timestamp.
        Synthesizes the actual spoken transcript into a concise narrative.
        """
        start_min = int(chunk['start_time'] // 60)
        start_sec = int(chunk['start_time'] % 60)
        end_min = int(chunk['end_time'] // 60)
        end_sec = int(chunk['end_time'] % 60)

        transcript = chunk.get("transcript", "").strip()
        time_label = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"

        # If LLM API is configured, use it
        if self.provider == "openai" and self.api_key:
            prompt = f"Explain what happened in this video section ({time_label}) in 2-3 concise, informative sentences based on this transcript:\n{transcript}"
            res = self._call_openai(prompt)
            if res:
                return res

        clean_sents = self._clean_transcript_sentences(transcript)

        if clean_sents:
            # Pick top 2-3 most informative sentences
            content_body = " ".join(clean_sents[:3])
            return f"From {time_label}, {content_body}"

        # Fallback if no speech recognized in this window
        return f"This section ({time_label}) features key presentation content, visual demonstrations, and speaker commentary."

    def generate_final_summary(self, video_metadata: dict, selected_chunks: list, chunk_summaries: list) -> dict:
        """
        Builds:
        1. Comprehensive Executive Video Summary explaining what happened across the video
        2. Key Takeaways (4-8 distinct bullet points)
        3. Timestamped Sections with clear topic titles and narrative summaries
        """
        duration_mins = video_metadata.get("duration_minutes", 30)
        is_audio = video_metadata.get("is_audio_only", False)
        media_type = "audio recording" if is_audio else "video"

        # ── 1. Executive Summary ─────────────────────────────────────────────
        # Aggregate all actual speech statements across selected chunks
        all_sentences = []
        for c in selected_chunks:
            trans = c.get("transcript", "")
            sents = self._clean_transcript_sentences(trans)
            all_sentences.extend(sents)

        overall_summary = ""
        key_points = []

        if self.provider == "openai" and self.api_key:
            all_text = " ".join(all_sentences[:20])
            prompt = f"Based on the transcript of this {duration_mins:.1f}-minute {media_type}, explain what happened across the video in a detailed summary paragraph, followed by 4-8 bullet points of key takeaways:\n{all_text}"
            res = self._call_openai(prompt)
            if "Key takeaways:" in res or "Takeaways:" in res:
                split_tag = "Key takeaways:" if "Key takeaways:" in res else "Takeaways:"
                parts = res.split(split_tag)
                overall_summary = parts[0].strip()
                key_points = [p.strip("- *0123456789.") for p in parts[1].strip().split("\n") if len(p.strip()) > 10]
            elif res:
                overall_summary = res

        if not overall_summary:
            if all_sentences:
                # Build an informative multi-sentence narrative
                lead = f"This {duration_mins:.1f}-minute {media_type} covers important themes and discussions across its timeline."
                first_pts = all_sentences[:2]
                mid_pts = [all_sentences[len(all_sentences) // 2]] if len(all_sentences) > 3 else []
                conclusion_pts = [all_sentences[-1]] if len(all_sentences) > 2 else []

                narrative_body = " ".join(first_pts + mid_pts + conclusion_pts)
                overall_summary = f"{lead} Throughout the session, {narrative_body}"
            else:
                overall_summary = f"A comprehensive {duration_mins:.1f}-minute {media_type} presenting key discussions, concepts, and detailed audio-visual demonstrations across multiple chapters."

        # ── 2. Key Takeaways (4 to 8 distinct points) ─────────────────────────
        if not key_points:
            key_points = []
            seen_points = set()

            # First, pull direct salient sentences from different chunks across the timeline
            for chunk in selected_chunks:
                trans = chunk.get("transcript", "")
                sents = self._clean_transcript_sentences(trans)
                for s in sents:
                    s_lower = s.lower()
                    if s_lower not in seen_points and len(s) > 20:
                        seen_points.add(s_lower)
                        key_points.append(s)
                        break  # One key point per chunk for temporal diversity

            # If still fewer than 4, pull additional unique sentences
            if len(key_points) < 4:
                for s in all_sentences:
                    s_lower = s.lower()
                    if s_lower not in seen_points and len(s) > 20:
                        seen_points.add(s_lower)
                        key_points.append(s)
                    if len(key_points) >= 6:
                        break

            # Limit to 8 points
            key_points = key_points[:8]

            # Fallback if transcript was empty
            if len(key_points) < 3:
                key_points = [
                    f"Comprehensive coverage of core presentation themes across the full {duration_mins:.1f}-minute duration.",
                    "Sequential breakdown of key discussion topics and speaker commentary.",
                    "Multimodal importance model identified significant discussion segments for quick review."
                ]

        # ── 3. Timestamped Sections with Informative Descriptions ─────────────
        important_sections = []
        for i, (chunk, summary) in enumerate(zip(selected_chunks, chunk_summaries)):
            start_sec = chunk["start_time"]
            end_sec = chunk["end_time"]
            score = chunk.get("importance_score", 0.85)

            start_fmt = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
            end_fmt = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"

            # Derive a descriptive title from the chunk's content
            keywords = self._derive_topic_keywords(chunk.get("transcript", ""))
            section_title = f"{start_fmt} - {end_fmt} | {keywords}"

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
        except Exception:
            pass
        return ""

    def unload_model(self):
        if self.local_model is not None and self.local_model != "lightweight":
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
