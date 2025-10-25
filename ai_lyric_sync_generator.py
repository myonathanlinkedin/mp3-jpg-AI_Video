import os
import logging
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import AudioFileClip, ImageSequenceClip
import librosa
import math
import torch
import whisper
from difflib import SequenceMatcher
import re
from collections import Counter
import faiss
import sentence_transformers

# Suppress Triton kernel warnings (Triton not available on Windows)
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
warnings.filterwarnings("ignore", message="falling back to a slower")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.timing")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AILyricSyncGenerator:
    def __init__(self, image_path, audio_path, output_path="context/ai_lyric_sync_video.mp4"):
        self.image_path = image_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.logger = logger
        self.image_size = (1024, 1024)  # Kembalikan ke resolusi asli
        
        # Load Whisper model untuk text extraction dengan optimasi CUDA
        self.logger.info("Loading Whisper model for AI text extraction...")
        # Gunakan device CUDA jika tersedia, dengan optimasi untuk Windows
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)
        self.logger.info(f"Whisper model loaded successfully on {device.upper()}!")
        
        # Load Sentence Transformer untuk semantic search
        self.logger.info("Loading Sentence Transformer for semantic search...")
        self.sentence_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        self.logger.info("Sentence Transformer loaded successfully!")
        
        # Initialize Faiss index
        self.faiss_index = None

    def _get_audio_duration(self):
        try:
            duration = librosa.get_duration(path=self.audio_path)
            return duration
        except Exception as e:
            self.logger.error(f"Error getting audio duration: {e}")
            return None

    def _extract_text_from_audio(self):
        """AI mengekstrak teks dari audio menggunakan Whisper dengan deteksi multiple voices"""
        self.logger.info("AI extracting text from audio with multiple voice detection...")
        
        try:
            # Transcribe dengan word timestamps dan speaker detection
            result = self.whisper_model.transcribe(
                self.audio_path, 
                word_timestamps=True,
                verbose=False
            )
            
            # Extract word-level timestamps dengan speaker info
            words_with_timestamps = []
            segments_with_speakers = []
            
            for segment in result['segments']:
                segment_words = []
                for word in segment['words']:
                    word_data = {
                        'word': word['word'].strip(),
                        'start': word['start'],
                        'end': word['end']
                    }
                    words_with_timestamps.append(word_data)
                    segment_words.append(word_data)
                
                # Deteksi speaker berdasarkan karakteristik audio segment
                speaker_id = self._detect_speaker_change(segment_words, segments_with_speakers)
                segments_with_speakers.append({
                    'words': segment_words,
                    'speaker_id': speaker_id,
                    'start': segment['start'],
                    'end': segment['end']
                })
            
            self.logger.info(f"AI extracted {len(words_with_timestamps)} words from audio with {len(set(s['speaker_id'] for s in segments_with_speakers))} detected speakers")
            return words_with_timestamps, segments_with_speakers
        except Exception as e:
            self.logger.error(f"Error during AI text extraction: {e}")
            return None, None

    def _detect_speaker_change(self, current_segment_words, previous_segments):
        """Deteksi perubahan speaker dan overlapping voices berdasarkan karakteristik audio"""
        if not previous_segments:
            return 0  # Speaker pertama
        
        # Analisis karakteristik segment saat ini
        current_text = " ".join([w['word'] for w in current_segment_words])
        current_duration = current_segment_words[-1]['end'] - current_segment_words[0]['start']
        current_word_count = len(current_segment_words)
        
        # Bandingkan dengan segment sebelumnya
        last_segment = previous_segments[-1]
        last_text = " ".join([w['word'] for w in last_segment['words']])
        last_duration = last_segment['words'][-1]['end'] - last_segment['words'][0]['start']
        last_word_count = len(last_segment['words'])
        
        # Deteksi perubahan speaker berdasarkan:
        # 1. Gap waktu yang besar (>1.5 detik) - lebih sensitif untuk overlapping voices
        # 2. Perubahan drastis dalam tempo/rhythm
        # 3. Perbedaan karakteristik text
        # 4. Deteksi overlapping voices
        
        time_gap = current_segment_words[0]['start'] - last_segment['words'][-1]['end']
        
        # Jika ada gap waktu besar, kemungkinan speaker berbeda
        if time_gap > 1.5:  # Lebih sensitif untuk overlapping voices
            return last_segment['speaker_id'] + 1
        
        # Analisis perbedaan tempo
        current_tempo = current_word_count / current_duration if current_duration > 0 else 0
        last_tempo = last_word_count / last_duration if last_duration > 0 else 0
        
        tempo_diff = abs(current_tempo - last_tempo) / max(current_tempo, last_tempo, 0.1)
        
        # Jika perbedaan tempo > 40%, kemungkinan speaker berbeda (lebih sensitif)
        if tempo_diff > 0.4:
            return last_segment['speaker_id'] + 1
        
        # Deteksi overlapping voices berdasarkan karakteristik text
        # Jika ada perubahan drastis dalam panjang kata atau pola
        current_avg_word_length = sum(len(w['word']) for w in current_segment_words) / len(current_segment_words)
        last_avg_word_length = sum(len(w['word']) for w in last_segment['words']) / len(last_segment['words'])
        
        word_length_diff = abs(current_avg_word_length - last_avg_word_length) / max(current_avg_word_length, last_avg_word_length, 1)
        
        # Jika perbedaan panjang kata > 30%, kemungkinan speaker berbeda
        if word_length_diff > 0.3:
            return last_segment['speaker_id'] + 1
        
        # Gunakan speaker yang sama
        return last_segment['speaker_id']

    def _detect_overlapping_voices(self, segments_with_speakers):
        """Deteksi overlapping voices dan multiple simultaneous speakers"""
        self.logger.info("Detecting overlapping voices and multiple simultaneous speakers...")
        
        overlapping_segments = []
        
        for i in range(len(segments_with_speakers)):
            current_segment = segments_with_speakers[i]
            current_start = current_segment['start']
            current_end = current_segment['end']
            
            # Cari segment yang overlap dengan segment saat ini
            overlapping_with = []
            for j in range(len(segments_with_speakers)):
                if i != j:
                    other_segment = segments_with_speakers[j]
                    other_start = other_segment['start']
                    other_end = other_segment['end']
                    
                    # Cek apakah ada overlap
                    if (current_start < other_end and current_end > other_start):
                        overlap_duration = min(current_end, other_end) - max(current_start, other_start)
                        overlap_percentage = overlap_duration / min(current_end - current_start, other_end - other_start)
                        
                        if overlap_percentage > 0.3:  # Overlap > 30%
                            overlapping_with.append({
                                'segment_id': j,
                                'speaker_id': other_segment['speaker_id'],
                                'overlap_percentage': overlap_percentage,
                                'overlap_duration': overlap_duration
                            })
            
            if overlapping_with:
                overlapping_segments.append({
                    'segment_id': i,
                    'speaker_id': current_segment['speaker_id'],
                    'overlapping_with': overlapping_with,
                    'is_overlapping': True
                })
        
        # Analisis pola overlapping voices
        voice_groups = self._analyze_voice_groups(overlapping_segments, segments_with_speakers)
        
        self.logger.info(f"Detected {len(overlapping_segments)} overlapping segments with {len(voice_groups)} voice groups")
        return overlapping_segments, voice_groups

    def _analyze_voice_groups(self, overlapping_segments, segments_with_speakers):
        """Analisis grup suara yang sering muncul bersamaan"""
        voice_groups = []
        
        # Buat mapping speaker yang sering overlap
        speaker_overlap_map = {}
        
        for overlap_info in overlapping_segments:
            main_speaker = overlap_info['speaker_id']
            overlapping_speakers = [ov['speaker_id'] for ov in overlap_info['overlapping_with']]
            
            # Gabungkan semua speaker yang overlap
            all_speakers = [main_speaker] + overlapping_speakers
            
            # Update mapping
            for speaker in all_speakers:
                if speaker not in speaker_overlap_map:
                    speaker_overlap_map[speaker] = set()
                speaker_overlap_map[speaker].update(all_speakers)
        
        # Buat grup berdasarkan speaker yang sering overlap
        processed_speakers = set()
        for speaker, overlapping_speakers in speaker_overlap_map.items():
            if speaker not in processed_speakers:
                # Buat grup baru
                group_speakers = list(overlapping_speakers)
                voice_groups.append({
                    'group_id': len(voice_groups),
                    'speakers': group_speakers,
                    'is_chorus': len(group_speakers) > 2,  # Chorus jika >2 speakers
                    'is_duet': len(group_speakers) == 2    # Duet jika 2 speakers
                })
                processed_speakers.update(group_speakers)
        
        return voice_groups

    def _analyze_song_structure(self, voice_groups, lyrics):
        """Analisis struktur lagu berdasarkan voice groups"""
        self.logger.info("Analyzing song structure based on voice groups...")
        
        # Deteksi pola struktur lagu berdasarkan jumlah speakers
        structure = {
            'verses': [],
            'choruses': [],
            'bridges': [],
            'outros': []
        }
        
        for group in voice_groups:
            if group['is_chorus']:
                structure['choruses'].append(group)
            elif group['is_duet']:
                structure['verses'].append(group)
            else:
                # Single speaker - bisa verse atau outro
                if len(group['speakers']) == 1:
                    structure['verses'].append(group)
        
        # Analisis pola berdasarkan jumlah grup
        total_groups = len(voice_groups)
        chorus_count = len(structure['choruses'])
        verse_count = len(structure['verses'])
        
        self.logger.info(f"Song structure: {verse_count} verses, {chorus_count} choruses, {total_groups} total voice groups")
        
        return structure

    def _group_lyrics_by_structure(self, lyrics, song_structure):
        """Group lyrics berdasarkan struktur lagu yang terdeteksi"""
        lyrics_groups = []
        
        # Asumsi struktur lagu berdasarkan pola umum
        # Verse 1, Chorus, Verse 2, Chorus, Bridge, Chorus, Outro
        
        # Estimasi jumlah lyrics per section berdasarkan total lyrics
        total_lyrics = len(lyrics)
        
        # Estimasi struktur berdasarkan pola umum
        if total_lyrics >= 60:  # Lagu panjang
            verse_size = 8  # 4 English + 4 Indonesian per verse
            chorus_size = 8  # 4 English + 4 Indonesian per chorus
            bridge_size = 8
            outro_size = 8
        else:  # Lagu pendek
            verse_size = 6  # 3 English + 3 Indonesian per verse
            chorus_size = 6  # 3 English + 3 Indonesian per chorus
            bridge_size = 6
            outro_size = 6
        
        current_idx = 0
        
        # Verse 1
        if current_idx + verse_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'verse',
                'lyrics': lyrics[current_idx:current_idx + verse_size],
                'speakers': [0, 1],  # Duet
                'start_idx': current_idx
            })
            current_idx += verse_size
        
        # Chorus 1
        if current_idx + chorus_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'chorus',
                'lyrics': lyrics[current_idx:current_idx + chorus_size],
                'speakers': [0, 1, 2, 3],  # Multiple voices
                'start_idx': current_idx
            })
            current_idx += chorus_size
        
        # Verse 2
        if current_idx + verse_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'verse',
                'lyrics': lyrics[current_idx:current_idx + verse_size],
                'speakers': [0, 1],  # Duet
                'start_idx': current_idx
            })
            current_idx += verse_size
        
        # Chorus 2
        if current_idx + chorus_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'chorus',
                'lyrics': lyrics[current_idx:current_idx + chorus_size],
                'speakers': [0, 1, 2, 3],  # Multiple voices
                'start_idx': current_idx
            })
            current_idx += chorus_size
        
        # Bridge
        if current_idx + bridge_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'bridge',
                'lyrics': lyrics[current_idx:current_idx + bridge_size],
                'speakers': [0, 1],  # Duet
                'start_idx': current_idx
            })
            current_idx += bridge_size
        
        # Chorus 3
        if current_idx + chorus_size <= total_lyrics:
            lyrics_groups.append({
                'type': 'chorus',
                'lyrics': lyrics[current_idx:current_idx + chorus_size],
                'speakers': [0, 1, 2, 3],  # Multiple voices
                'start_idx': current_idx
            })
            current_idx += chorus_size
        
        # Outro
        if current_idx < total_lyrics:
            remaining_lyrics = lyrics[current_idx:]
            lyrics_groups.append({
                'type': 'outro',
                'lyrics': remaining_lyrics,
                'speakers': [0],  # Single voice
                'start_idx': current_idx
            })
        
        return lyrics_groups

    def _validate_and_fix_timing_overlapping(self, lyric_data, duration):
        """Validasi dan perbaiki timing dengan awareness overlapping voices"""
        self.logger.info("Validating and fixing timing for overlapping voices...")
        
        # Jika tidak ada data, gunakan fallback
        if not lyric_data:
            return self._fallback_timing([], duration)
        
        # Validasi setiap lyric
        for i, lyric_entry in enumerate(lyric_data):
            # Validasi timing dasar
            if (lyric_entry['start'] < 0 or lyric_entry['end'] < 0 or 
                lyric_entry['start'] >= duration or lyric_entry['end'] > duration):
                
                # Timing tidak valid, gunakan fallback untuk lyric ini
                start_time = (i / len(lyric_data)) * duration * 0.9
                end_time = ((i + 1) / len(lyric_data)) * duration * 0.9
                lyric_entry['start'] = start_time
                lyric_entry['end'] = end_time
                self.logger.info(f"🔧 Fixed invalid timing for lyric {i+1}: '{lyric_entry['text']}' from {start_time:.2f}s to {end_time:.2f}s")
            
            # Validasi durasi lyric dengan awareness overlapping voices
            lyric_duration = lyric_entry['end'] - lyric_entry['start']
            is_overlapping = lyric_entry.get('is_overlapping', False)
            group_type = lyric_entry.get('group_type', 'verse')
            
            # Durasi lebih fleksibel untuk overlapping voices
            if is_overlapping:
                min_duration = 0.2  # Sangat pendek untuk overlapping
                max_duration = 20.0  # Sangat panjang untuk chorus
            else:
                min_duration = 0.5
                max_duration = 12.0
            
            if lyric_duration < min_duration:
                lyric_entry['end'] = lyric_entry['start'] + min_duration
                self.logger.info(f"🔧 Extended short duration for lyric {i+1}: '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
            elif lyric_duration > max_duration:
                lyric_entry['end'] = lyric_entry['start'] + max_duration
                self.logger.info(f"🔧 Shortened long duration for lyric {i+1}: '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
        
        # Sort by start time untuk analisis overlap
        lyric_data.sort(key=lambda x: x['start'])
        
        # Perbaiki overlap dengan awareness overlapping voices
        for i in range(1, len(lyric_data)):
            prev_end = lyric_data[i-1]['end']
            curr_start = lyric_data[i]['start']
            prev_overlapping = lyric_data[i-1].get('is_overlapping', False)
            curr_overlapping = lyric_data[i].get('is_overlapping', False)
            prev_group = lyric_data[i-1].get('group_type', 'verse')
            curr_group = lyric_data[i].get('group_type', 'verse')
            
            if curr_start < prev_end:  # Ada overlap
                overlap_duration = prev_end - curr_start
                prev_duration = lyric_data[i-1]['end'] - lyric_data[i-1]['start']
                
                # Jika keduanya overlapping (chorus), biarkan overlap lebih besar
                if prev_overlapping and curr_overlapping:
                    if overlap_duration > prev_duration * 0.8:  # Overlap > 80% untuk chorus
                        lyric_data[i]['start'] = prev_end - 1.0  # Biarkan 1s overlap untuk chorus
                        self.logger.info(f"🔧 Adjusted chorus overlap for lyric {i+1}: '{lyric_data[i]['text']}' now starts at {lyric_data[i]['start']:.2f}s")
                # Jika salah satu overlapping, biarkan sedikit overlap
                elif prev_overlapping or curr_overlapping:
                    if overlap_duration > prev_duration * 0.5:  # Overlap > 50%
                        lyric_data[i]['start'] = prev_end - 0.3  # Biarkan 0.3s overlap
                        self.logger.info(f"🔧 Adjusted mixed overlap for lyric {i+1}: '{lyric_data[i]['text']}' now starts at {lyric_data[i]['start']:.2f}s")
                # Jika tidak ada yang overlapping, perbaiki overlap normal
                else:
                    if overlap_duration > prev_duration * 0.3:  # Overlap > 30%
                        lyric_data[i]['start'] = prev_end
                        self.logger.info(f"🔧 Fixed normal overlap for lyric {i+1}: '{lyric_data[i]['text']}' now starts at {prev_end:.2f}s")
        
        # Log final timing dengan info overlapping
        for i, lyric_entry in enumerate(lyric_data):
            speaker_id = lyric_entry.get('speaker_id', 0)
            group_type = lyric_entry.get('group_type', 'verse')
            is_overlapping = lyric_entry.get('is_overlapping', False)
            overlap_info = " (OVERLAPPING)" if is_overlapping else ""
            self.logger.info(f"✅ Final timing for lyric {i+1} (Speaker {speaker_id}, {group_type}{overlap_info}): '{lyric_entry['text']}' from {lyric_entry['start']:.2f}s to {lyric_entry['end']:.2f}s")
        
        return lyric_data

    def _build_faiss_index(self, extracted_words, segments_with_speakers=None):
        """Build Faiss index for semantic search dengan chunk yang lebih besar dan overlap, dengan dukungan multiple speakers"""
        self.logger.info("Building Faiss index for semantic search with multiple speaker support...")
        
        # Create text chunks from extracted words dengan overlap
        chunk_size = 10  # Increased from 5 to 10 words per chunk
        overlap_size = 3  # 3 words overlap between chunks
        text_chunks = []
        chunk_timings = []
        
        i = 0
        while i < len(extracted_words):
            chunk_words = extracted_words[i:i + chunk_size]
            if len(chunk_words) < 3:  # Skip chunks that are too small
                break
                
            chunk_text = " ".join([w['word'] for w in chunk_words])
            
            # Deteksi speaker untuk chunk ini
            speaker_id = self._get_speaker_for_chunk(chunk_words, segments_with_speakers)
            
            text_chunks.append(chunk_text)
            chunk_timings.append({
                'start': chunk_words[0]['start'],
                'end': chunk_words[-1]['end'],
                'words': chunk_words,
                'speaker_id': speaker_id
            })
            
            # Move forward with overlap
            i += chunk_size - overlap_size
        
        # Generate embeddings using Sentence Transformer
        embeddings = self.sentence_model.encode(text_chunks)
        
        # Create Faiss index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        self.logger.info(f"Faiss index built with {len(text_chunks)} chunks (chunk_size={chunk_size}, overlap={overlap_size})")
        return text_chunks, chunk_timings

    def _get_speaker_for_chunk(self, chunk_words, segments_with_speakers):
        """Dapatkan speaker ID untuk chunk berdasarkan timing"""
        if not segments_with_speakers:
            return 0
        
        chunk_start = chunk_words[0]['start']
        chunk_end = chunk_words[-1]['end']
        
        # Cari segment yang overlap dengan chunk
        for segment in segments_with_speakers:
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Jika ada overlap, gunakan speaker dari segment tersebut
            if (chunk_start < segment_end and chunk_end > segment_start):
                return segment['speaker_id']
        
        # Jika tidak ada overlap, gunakan speaker dari segment terdekat
        closest_segment = min(segments_with_speakers, 
                            key=lambda s: min(abs(chunk_start - s['start']), abs(chunk_end - s['end'])))
        return closest_segment['speaker_id']

    def _semantic_search_lyric(self, lyric_text, text_chunks, chunk_timings, top_k=3):
        """Use Faiss to find semantic matches for a lyric"""
        # Generate embedding for the lyric
        lyric_embedding = self.sentence_model.encode([lyric_text])
        faiss.normalize_L2(lyric_embedding)
        
        # Search in Faiss index
        similarities, indices = self.faiss_index.search(lyric_embedding, top_k)
        
        # Get best matches
        best_matches = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(chunk_timings):
                best_matches.append({
                    'similarity': float(similarity),
                    'text': text_chunks[idx],
                    'timing': chunk_timings[idx]
                })
        
        return best_matches

    def _match_extracted_text_with_lyrics(self, lyrics, extracted_words, segments_with_speakers=None):
        """AI mencocokkan teks yang diekstrak dengan lirik menggunakan semantic search dengan dukungan overlapping voices"""
        self.logger.info("AI matching extracted text with provided lyrics using overlapping voices detection...")
        
        if not extracted_words:
            self.logger.warning("No extracted words available, using fallback timing")
            return self._fallback_timing(lyrics, self._get_audio_duration())

        matched_lyrics_with_timing = []
        duration = self._get_audio_duration()
        
        # Gabungkan semua extracted words menjadi satu string untuk logging
        full_extracted_text = " ".join([w['word'] for w in extracted_words])
        self.logger.info(f"Full extracted text: {full_extracted_text[:200]}...")
        
        # Deteksi overlapping voices dan voice groups
        overlapping_segments, voice_groups = self._detect_overlapping_voices(segments_with_speakers)
        
        # Build Faiss index untuk semantic search dengan speaker info
        text_chunks, chunk_timings = self._build_faiss_index(extracted_words, segments_with_speakers)
        
        # Analisis struktur lagu berdasarkan voice groups
        song_structure = self._analyze_song_structure(voice_groups, lyrics)
        
        # Group lyrics berdasarkan struktur lagu yang terdeteksi
        lyrics_groups = self._group_lyrics_by_structure(lyrics, song_structure)
        
        self.logger.info(f"Grouped lyrics into {len(lyrics_groups)} structure groups")
        
        # Untuk setiap grup struktur, cari matches yang sesuai
        for group_idx, group_info in enumerate(lyrics_groups):
            group_lyrics = group_info['lyrics']
            group_type = group_info['type']  # 'verse', 'chorus', 'bridge', 'outro'
            group_speakers = group_info['speakers']
            
            self.logger.info(f"Processing {group_type} group {group_idx + 1} with {len(group_lyrics)} lyrics and {len(group_speakers)} speakers")
            
            for lyric_idx, lyric_line in enumerate(group_lyrics):
                global_lyric_idx = group_info['start_idx'] + lyric_idx
                
                # Cari semantic matches menggunakan Faiss dengan awareness overlapping voices
                matches = self._semantic_search_lyric(lyric_line, text_chunks, chunk_timings, top_k=7)  # Lebih banyak candidates
                
                best_match = None
                for match in matches:
                    if match['similarity'] >= 0.35:  # Lebih rendah untuk overlapping voices
                        timing = match['timing']
                        # Validasi timing: durasi tidak boleh terlalu pendek atau terlalu panjang
                        lyric_duration = timing['end'] - timing['start']
                        if 0.2 <= lyric_duration <= 15.0:  # Lebih fleksibel untuk overlapping voices
                            best_match = match
                            break
                
                if best_match:
                    timing = best_match['timing']
                    matched_lyrics_with_timing.append({
                        'text': lyric_line,
                        'start': timing['start'],
                        'end': timing['end'],
                        'speaker_id': timing.get('speaker_id', group_speakers[0] if group_speakers else 0),
                        'group_type': group_type,
                        'is_overlapping': len(group_speakers) > 1
                    })
                    self.logger.info(f"✅ Overlapping-voices matched lyric {global_lyric_idx+1}: '{lyric_line}' from {timing['start']:.2f}s to {timing['end']:.2f}s (similarity: {best_match['similarity']:.2f}, {group_type}, speakers: {len(group_speakers)})")
                else:
                    # Jika tidak ada semantic match, gunakan timing berdasarkan struktur lagu
                    start_time = (global_lyric_idx / len(lyrics)) * duration * 0.9
                    end_time = ((global_lyric_idx + 1) / len(lyrics)) * duration * 0.9
                    matched_lyrics_with_timing.append({
                        'text': lyric_line,
                        'start': start_time,
                        'end': end_time,
                        'speaker_id': group_speakers[0] if group_speakers else 0,
                        'group_type': group_type,
                        'is_overlapping': len(group_speakers) > 1
                    })
                    self.logger.info(f"📝 Structure-based timing for lyric {global_lyric_idx+1}: '{lyric_line}' from {start_time:.2f}s to {end_time:.2f}s ({group_type}, speakers: {len(group_speakers)})")
        
        # Validasi dan perbaiki timing dengan awareness overlapping voices
        matched_lyrics_with_timing = self._validate_and_fix_timing_overlapping(matched_lyrics_with_timing, duration)
                
        return matched_lyrics_with_timing

    def _find_word_indices_for_match(self, match, extracted_words, full_extracted_text):
        """Temukan word indices yang sesuai dengan match"""
        # Hitung posisi match dalam full_extracted_text
        match_start = match.b
        match_end = match.b + match.size
        
        # Temukan kata-kata yang sesuai dengan match
        word_indices = []
        current_pos = 0
        
        for i, word_data in enumerate(extracted_words):
            word = word_data['word']
            word_start = current_pos
            word_end = current_pos + len(word)
            
            # Cek apakah kata ini overlap dengan match
            if (word_start < match_end and word_end > match_start):
                word_indices.append(i)
            
            current_pos += len(word) + 1  # +1 untuk spasi
        
        return word_indices

    def _validate_and_fix_timing_multispeaker(self, lyric_data, duration):
        """Validasi dan perbaiki timing dengan awareness multiple speakers"""
        self.logger.info("Validating and fixing timing for multiple speakers...")
        
        # Jika tidak ada data, gunakan fallback
        if not lyric_data:
            return self._fallback_timing([], duration)
        
        # Validasi setiap lyric
        for i, lyric_entry in enumerate(lyric_data):
            # Validasi timing dasar
            if (lyric_entry['start'] < 0 or lyric_entry['end'] < 0 or 
                lyric_entry['start'] >= duration or lyric_entry['end'] > duration):
                
                # Timing tidak valid, gunakan fallback untuk lyric ini
                start_time = (i / len(lyric_data)) * duration * 0.9
                end_time = ((i + 1) / len(lyric_data)) * duration * 0.9
                lyric_entry['start'] = start_time
                lyric_entry['end'] = end_time
                self.logger.info(f"🔧 Fixed invalid timing for lyric {i+1}: '{lyric_entry['text']}' from {start_time:.2f}s to {end_time:.2f}s")
            
            # Validasi durasi lyric
            lyric_duration = lyric_entry['end'] - lyric_entry['start']
            if lyric_duration < 0.3:  # Terlalu pendek untuk multiple speakers
                lyric_entry['end'] = lyric_entry['start'] + 0.3
                self.logger.info(f"🔧 Extended short duration for lyric {i+1}: '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
            elif lyric_duration > 12.0:  # Terlalu panjang untuk multiple speakers
                lyric_entry['end'] = lyric_entry['start'] + 12.0
                self.logger.info(f"🔧 Shortened long duration for lyric {i+1}: '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
        
        # Sort by start time untuk analisis overlap
        lyric_data.sort(key=lambda x: x['start'])
        
        # Perbaiki overlap dengan awareness speaker
        for i in range(1, len(lyric_data)):
            prev_end = lyric_data[i-1]['end']
            curr_start = lyric_data[i]['start']
            prev_speaker = lyric_data[i-1].get('speaker_id', 0)
            curr_speaker = lyric_data[i].get('speaker_id', 0)
            
            if curr_start < prev_end:  # Ada overlap
                overlap_duration = prev_end - curr_start
                prev_duration = lyric_data[i-1]['end'] - lyric_data[i-1]['start']
                
                # Jika speaker sama dan overlap > 30%, perbaiki
                if prev_speaker == curr_speaker and overlap_duration > prev_duration * 0.3:
                    lyric_data[i]['start'] = prev_end
                    self.logger.info(f"🔧 Fixed overlap for same speaker lyric {i+1}: '{lyric_data[i]['text']}' now starts at {prev_end:.2f}s")
                # Jika speaker berbeda, biarkan sedikit overlap untuk transisi natural
                elif prev_speaker != curr_speaker and overlap_duration > 1.0:
                    lyric_data[i]['start'] = prev_end - 0.5  # Biarkan 0.5s overlap untuk transisi
                    self.logger.info(f"🔧 Adjusted overlap for speaker transition lyric {i+1}: '{lyric_data[i]['text']}' now starts at {lyric_data[i]['start']:.2f}s")
        
        # Log final timing dengan speaker info
        for i, lyric_entry in enumerate(lyric_data):
            speaker_id = lyric_entry.get('speaker_id', 0)
            self.logger.info(f"✅ Final timing for lyric {i+1} (Speaker {speaker_id}): '{lyric_entry['text']}' from {lyric_entry['start']:.2f}s to {lyric_entry['end']:.2f}s")
        
        return lyric_data

    def _fallback_timing(self, lyrics, duration):
        """Fallback timing jika AI matching gagal"""
        lyric_timings = []
        if not lyrics:
            return lyric_timings
            
        total_lyric_time = duration * 0.90
        time_per_lyric = total_lyric_time / len(lyrics)
        
        for i in range(len(lyrics)):
            start_time = i * time_per_lyric
            end_time = (i + 1) * time_per_lyric
            lyric_timings.append({'text': lyrics[i], 'start': start_time, 'end': end_time})
        
        return lyric_timings

    def _create_smooth_frames(self, image, duration, fps=30, lyric_data=None):
        frames = []
        total_frames = int(duration * fps)
        
        width, height = image.size
        
        try:
            font = ImageFont.truetype("arial.ttf", 42) # Font size dikurangi 30% dari 60px
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
            except:
                font = ImageFont.load_default()
        
        for i in range(total_frames):
            progress = i / total_frames
            time_in_video = i / fps

            # Complex smooth movement - multiple frequencies
            zoom_factor = 1.0 + 0.1 * math.sin(2 * math.pi * progress * 2) + 0.05 * math.sin(2 * math.pi * progress * 7)
            pan_x = 0.1 * math.sin(2 * math.pi * progress * 1.5)
            pan_y = 0.1 * math.cos(2 * math.pi * progress * 1.2)
            rotation = 2 * math.sin(2 * math.pi * progress * 0.8)
            
            # Apply transformations
            zoomed_width = int(width * zoom_factor)
            zoomed_height = int(height * zoom_factor)
            
            zoomed_image = image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
            
            # Calculate crop area
            crop_x = int((zoomed_width - width) / 2 + pan_x * width)
            crop_y = int((zoomed_height - height) / 2 + pan_y * height)
            
            crop_x = max(0, min(crop_x, zoomed_width - width))
            crop_y = max(0, min(crop_y, zoomed_height - height))
            
            cropped_image = zoomed_image.crop((crop_x, crop_y, crop_x + width, crop_y + height))
            
            # Apply rotation
            rotated_image = cropped_image.rotate(rotation, expand=False, fillcolor=(0, 0, 0))
            
            # Add lyrics to frame
            if lyric_data:
                current_lyric_text = ""
                current_start_time = 0
                current_end_time = 0
                
                for entry in lyric_data:
                    if entry['start'] <= time_in_video < entry['end']:
                        current_lyric_text = entry['text']
                        current_start_time = entry['start']
                        current_end_time = entry['end']
                        break
                
                if current_lyric_text:
                    draw = ImageDraw.Draw(rotated_image)
                    bbox = draw.textbbox((0, 0), current_lyric_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position text at bottom center, lebih tinggi
                    text_x = (width - text_width) // 2
                    text_y = height - text_height - 150 # Lebih tinggi lagi
                    
                    # Smooth fade effect
                    fade_duration = 0.3
                    alpha = 0.0
                    
                    if time_in_video < current_start_time + fade_duration:
                        alpha = (time_in_video - current_start_time) / fade_duration
                    elif time_in_video > current_end_time - fade_duration:
                        alpha = (current_end_time - time_in_video) / fade_duration
                    else:
                        alpha = 1.0
                    
                    alpha_int = int(alpha * 255)
                    
                    # Draw text with stroke
                    draw.text((text_x, text_y), current_lyric_text, 
                             font=font, 
                             fill=(255, 255, 255, alpha_int), 
                             stroke_fill=(0, 0, 0, alpha_int), 
                             stroke_width=2)

            frames.append(rotated_image)
            
            if i % (total_frames // 10) == 0:
                self.logger.info(f"Creating frames: {progress * 100:.1f}%")
        
        self.logger.info("Creating frames: 100.0%")
        return frames

    def generate_video(self, lyrics=None):
        self.logger.info("=== AI Lyric Synchronization Video Generator ===")
        self.logger.info("Using AI text extraction + lyric matching for perfect synchronization...")

        if not os.path.exists(self.image_path):
            self.logger.error(f"Image file not found: {self.image_path}")
            return
        if not os.path.exists(self.audio_path):
            self.logger.error(f"Audio file not found: {self.audio_path}")
            return

        image = Image.open(self.image_path)
        image = image.resize(self.image_size, Image.LANCZOS)

        duration = self._get_audio_duration()
        if duration is None:
            return

        self.logger.info(f"Audio duration: {duration:.2f}s")
        
        # Step 1: AI mengekstrak teks dari audio dengan deteksi multiple speakers
        extracted_words, segments_with_speakers = self._extract_text_from_audio()
        
        # Step 2: AI mencocokkan teks yang diekstrak dengan lirik yang diberikan (multi-speaker)
        lyric_data = self._match_extracted_text_with_lyrics(lyrics, extracted_words, segments_with_speakers)
        
        # Step 3: Buat video dengan text yang sudah di-match
        frames = self._create_smooth_frames(image, duration, fps=30, lyric_data=lyric_data)
        
        # Convert to numpy arrays with memory optimization
        numpy_frames = []
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                self.logger.info(f"Converting frames to numpy: {i/len(frames)*100:.1f}%")
            numpy_frames.append(np.array(frame))
        
        # Create video
        video_clip = ImageSequenceClip(numpy_frames, fps=30)
        audio_clip = AudioFileClip(self.audio_path)
        final_video = video_clip.set_audio(audio_clip)

        try:
            final_video.write_videofile(
                self.output_path, 
                fps=30, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile="temp-audio.m4a", 
                remove_temp=True
            )
            self.logger.info(f"✅ AI-synchronized video created successfully: {self.output_path}")
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")


if __name__ == "__main__":
    image_file = "context/rapper.jpg"
    # File MP3 tidak ditemukan, silakan tambahkan file MP3 ke context folder
    audio_file = "context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3"
    output_video_file = "context/ai_lyric_sync_video.mp4"

    lyrics = [
        # Verse 1
        "Behind the glass towers, money flows in silence",
        "Di balik gedung kaca, uang diam-diam mengalir",
        "Promises to the people sold, conscience ignored",
        "Janji rakyat dijual, suara nurani dikaburkan",
        "Tight structures, broken systems, yet they laugh",
        "Struktur rapat, sistem retak, tapi mereka tertawa",
        "Structured corruption, a culture made real",
        "Korupsi terstruktur, jadi budaya yang nyata",
        
        "Closed doors, contracts played in shadows",
        "Pintu-pintu tertutup, kontrak main belakang",
        "People's projects stolen, while words remain hollow",
        "Proyek rakyat diambil, sementara janji cuma kata",
        "From center to regions, black chains expand",
        "Dari pusat sampai daerah, rantai hitam membentang",
        "Law throws up its hands, justice often lost",
        "Hukum lempar tangan, keadilan sering hilang",
        
        # Chorus
        "Black chains, don't stay silent",
        "Rantai hitam, jangan kau bungkam",
        "People see, but afraid to speak",
        "Rakyat lihat, tapi takut bicara",
        "Dirty money on the table, promises held hostage",
        "Uang kotor di meja, janji jadi sandera",
        "Structured corruption, we fight together",
        "Korupsi terstruktur, kita lawan bersama",
        
        # Verse 2
        "Money fills the pockets, morals vanish somewhere",
        "Duit masuk kantong, moral pergi entah kemana",
        "Officials wear masks, people still suffer",
        "Pejabat ganti topeng, rakyat tetap merana",
        "Audits just formalities, reports full of lies",
        "Audit cuma formalitas, laporan penuh tipu daya",
        "Transparency just slogans, justice is a disguise",
        "Transparansi tinggal slogan, keadilan hanya maya",
        
        "Not just small fry, big bosses play too",
        "Bukan cuma orang kecil, bos besar ikut main",
        "Projects, taxes, permits, all through illegal routes",
        "Proyek, pajak, izin, semua ada jalur haram",
        "If we stay silent, future generations inherit",
        "Kalau diam terus, anak cucu bakal warisi",
        "A culture of corruption, higher, more legitimate",
        "Budaya korupsi yang makin tinggi, makin murni",
        
        # Chorus
        "Black chains, don't stay silent",
        "Rantai hitam, jangan kau bungkam",
        "People see, but afraid to speak",
        "Rakyat lihat, tapi takut bicara",
        "Dirty money on the table, promises held hostage",
        "Uang kotor di meja, janji jadi sandera",
        "Structured corruption, we fight together",
        "Korupsi terstruktur, kita lawan bersama",
        
        # Bridge
        "Raise your voice, don't fear to fight",
        "Bangkit suara, jangan takut lawan",
        "Open eyes, open ears, don't stay silent",
        "Buka mata, buka telinga, jangan diam",
        "If we all care, chains can break",
        "Kalau semua peduli, rantai bisa patah",
        "The system is fragile, but we can change it",
        "Sistem rapuh, tapi kita bisa ubah",
        
        # Outro
        "Structured corruption, but we're no slaves",
        "Korupsi terstruktur, tapi kita bukan budak",
        "People united, they won't win anymore",
        "Rakyat bersatu, mereka takkan menang lagi",
        "Not just a dream, this is reality",
        "Bukan cuma mimpi, ini suara realita",
        "Fight the black chains, for our future",
        "Lawan rantai hitam, demi masa depan kita"
    ]

    generator = AILyricSyncGenerator(image_file, audio_file, output_video_file)
    generator.generate_video(lyrics=lyrics)