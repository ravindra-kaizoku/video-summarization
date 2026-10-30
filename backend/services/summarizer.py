import os
import re
import requests
from pathlib import Path
from backend.utils.memory_utils import get_available_memory_mb

class HierarchicalSummarizer:
    def __init__(self, config: dict):
        self.config = config
        llm_cfg = config.get("llm", {})
        self.provider = os.environ.get("LLM_PROVIDER") or llm_cfg.get("provider", "mock_api")
        self.api_key = (
            os.environ.get("GROQ_API_KEY") or
            os.environ.get("GEMINI_API_KEY") or
            os.environ.get("OPENAI_API_KEY") or
            llm_cfg.get("api_key", "")
        )
        self.tokenizer = None
        self.local_model = None

    def _call_llm(self, prompt: str) -> str:
        """Calls Groq, Gemini, or OpenAI API if any key is provided in the environment."""
        # 1. Groq API (free, ultra-fast Llama-3.3-70B)
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
                r = requests.post(url, headers=headers, json=payload, timeout=20)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[LLM] Groq error: {e}")

        # 2. Gemini API
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
                }
                r = requests.post(url, headers=headers, json=payload, timeout=20)
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                print(f"[LLM] Gemini error: {e}")

        # 3. OpenAI API
        openai_key = os.environ.get("OPENAI_API_KEY") or (self.api_key if self.provider == "openai" else "")
        if openai_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
                r = requests.post(url, headers=headers, json=payload, timeout=20)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[LLM] OpenAI error: {e}")

        return ""

    def _clean_transcript_sentences(self, transcript: str) -> list:
        """Extracts clean, meaningful declarative sentences from the transcript, removing conversational fillers."""
        raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if len(s.strip()) > 8]
        clean_sents = []
        # Filter greetings and channel boilerplate
        boilerplate = {'welcome back', 'subscribe to', 'hit the bell', 'thanks for watching', 'leave a comment', 'see you in the next video'}

        for s in raw_sents:
            s_lower = s.lower()
            if any(b in s_lower for b in boilerplate):
                continue
            # Strip conversational fillers
            s_clean = re.sub(r'^(Alright|So|Um|Uh|Well|You know|Like|Now|Okay)\s*,?\s*', '', s, flags=re.I).strip()
            if len(s_clean) > 15:
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
            'would', 'could', 'their', 'which', 'when', 'more', 'other', 'into', 'only',
            'going', 'said', 'were', 'does', 'don\'t', 'doesn\'t'
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
        """Explains what happened in this video section up to that timestamp."""
        start_min = int(chunk['start_time'] // 60)
        start_sec = int(chunk['start_time'] % 60)
        end_min = int(chunk['end_time'] // 60)
        end_sec = int(chunk['end_time'] % 60)

        transcript = chunk.get("transcript", "").strip()
        time_label = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"

        # If LLM API is available, ask LLM
        prompt = (
            f"Explain clearly what happens in this video section ({time_label}) in 2-3 concise, "
            f"informative sentences based on this spoken transcript:\n{transcript}"
        )
        llm_res = self._call_llm(prompt)
        if llm_res:
            return llm_res

        clean_sents = self._clean_transcript_sentences(transcript)
        if clean_sents:
            content_body = " ".join(clean_sents[:3])
            return f"From {time_label}, {content_body}"

        return f"This section ({time_label}) covers key presentation content, visual demonstrations, and speaker commentary."

    def generate_final_summary(self, video_metadata: dict, selected_chunks: list, chunk_summaries: list) -> dict:
        """
        Builds:
        1. Executive Video Summary: explains what happened across the video
        2. Key Takeaways: distinct key points from different parts of the video
        3. Timestamped Chapters: what happened up to each timestamp
        """
        duration_mins = video_metadata.get("duration_minutes", 30)
        is_audio = video_metadata.get("is_audio_only", False)
        media_type = "audio recording" if is_audio else "video"

        # Combine transcripts
        all_sentences = []
        for c in selected_chunks:
            trans = c.get("transcript", "")
            sents = self._clean_transcript_sentences(trans)
            all_sentences.extend(sents)

        overall_summary = ""
        key_points = []

        # Try LLM first if API key configured
        full_text = " ".join(all_sentences[:30])
        if full_text:
            prompt = (
                f"You are an expert video summarizer. Based on the following spoken transcript of a "
                f"{duration_mins:.1f}-minute {media_type}, provide:\n"
                f"1. A comprehensive summary paragraph explaining who is speaking, what the video discusses, "
                f"and how the topic develops across the video.\n"
                f"2. A numbered list of 4 to 8 distinct Key Takeaways covering different facts and advice.\n\n"
                f"Format strictly as:\n"
                f"Summary:\n[Summary paragraph]\n\n"
                f"Key Takeaways:\n1. [Point 1]\n2. [Point 2]...\n\n"
                f"Transcript:\n{full_text}"
            )
            llm_res = self._call_llm(prompt)
            if llm_res:
                if "Key Takeaways:" in llm_res or "Takeaways:" in llm_res:
                    tag = "Key Takeaways:" if "Key Takeaways:" in llm_res else "Takeaways:"
                    parts = llm_res.split(tag)
                    sum_part = parts[0].replace("Summary:", "").strip()
                    if len(sum_part) > 30:
                        overall_summary = sum_part
                    pts = [re.sub(r'^\d+\.\s*', '', p.strip('- *')).strip() for p in parts[1].strip().split("\n") if len(p.strip()) > 15]
                    if len(pts) >= 3:
                        key_points = pts[:8]

        # Built-in Extractive Synthesis if LLM not active
        if not overall_summary:
            if all_sentences:
                lead = f"This {duration_mins:.1f}-minute {media_type} covers key insights, arguments, and practical guidance across its timeline."
                first_pts = all_sentences[:2]
                mid_pts = [all_sentences[len(all_sentences) // 2]] if len(all_sentences) > 3 else []
                conclusion_pts = [all_sentences[-1]] if len(all_sentences) > 2 else []
                narrative_body = " ".join(first_pts + mid_pts + conclusion_pts)
                overall_summary = f"{lead} The discussion highlights: {narrative_body}"
            else:
                overall_summary = f"A comprehensive {duration_mins:.1f}-minute {media_type} presenting core concepts, practical guidance, and detailed demonstrations across multiple chapters."

        # Key Takeaways: ensure distinct points from across the timeline
        if not key_points:
            key_points = []
            seen_points = set()

            for chunk in selected_chunks:
                trans = chunk.get("transcript", "")
                sents = self._clean_transcript_sentences(trans)
                for s in sents:
                    s_lower = s.lower()
                    if s_lower not in seen_points and len(s) > 25:
                        seen_points.add(s_lower)
                        key_points.append(s)
                        break  # One distinct point per chunk

            if len(key_points) < 4:
                for s in all_sentences:
                    s_lower = s.lower()
                    if s_lower not in seen_points and len(s) > 25:
                        seen_points.add(s_lower)
                        key_points.append(s)
                    if len(key_points) >= 6:
                        break

            key_points = key_points[:8]
            if len(key_points) < 3:
                key_points = [
                    f"Comprehensive coverage of core presentation themes across the full {duration_mins:.1f}-minute duration.",
                    "Sequential breakdown of key discussion topics and speaker commentary.",
                    "Multimodal importance model identified significant discussion segments for quick review."
                ]

        # Timestamped Sections (Chapters)
        important_sections = []
        for i, (chunk, summary) in enumerate(zip(selected_chunks, chunk_summaries)):
            start_sec = chunk["start_time"]
            end_sec = chunk["end_time"]
            score = chunk.get("importance_score", 0.85)

            start_fmt = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
            end_fmt = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"

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

    def unload_model(self):
        pass
