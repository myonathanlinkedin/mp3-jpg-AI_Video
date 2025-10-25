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

# Suppress Triton kernel warnings
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
warnings.filterwarnings("ignore", message="falling back to a slower")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.timing")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedLongSentenceLyricSyncGenerator:
    def __init__(self, image_path, audio_path, output_path="context/advanced_long_sentence_video.mp4"):
        self.image_path = image_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.logger = logger
        self.image_size = (1024, 1024)
        
        # File paths untuk menyimpan hasil
        self.extracted_lyrics_file = "context/extracted_lyrics.txt"
        self.comparison_lyrics_file = "context/comparison_lyrics.txt"
        self.matched_lyrics_file = "context/matched_lyrics.txt"
        
        # Load Whisper model
        self.logger.info("Loading Whisper model for advanced text extraction...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=device)
        self.logger.info(f"Whisper model loaded successfully on {device.upper()}!")
        
        # Load Sentence Transformer
        self.logger.info("Loading Sentence Transformer for advanced semantic search...")
        self.sentence_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        self.logger.info("Sentence Transformer loaded successfully!")
        
        self.faiss_index = None

    def _get_audio_duration(self):
        try:
            duration = librosa.get_duration(path=self.audio_path)
            return duration
        except Exception as e:
            self.logger.error(f"Error getting audio duration: {e}")
            return None

    def _detect_language(self, text):
        """Deteksi bahasa dari text (English vs Indonesian)"""
        # Kata-kata umum English
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'behind', 'glass', 'towers', 'money', 'flows', 'silence', 'promises', 'people', 'sold', 'conscience', 'ignored', 'tight', 'structures', 'broken', 'systems', 'yet', 'they', 'laugh', 'structured', 'corruption', 'culture', 'made', 'real', 'closed', 'doors', 'contracts', 'played', 'shadows', 'projects', 'stolen', 'while', 'words', 'remain', 'hollow', 'center', 'regions', 'black', 'chains', 'expand', 'law', 'throws', 'hands', 'justice', 'often', 'lost', 'don', 'stay', 'see', 'afraid', 'speak', 'dirty', 'table', 'held', 'hostage', 'fight', 'together', 'fills', 'pockets', 'morals', 'vanish', 'somewhere', 'officials', 'wear', 'masks', 'still', 'suffer', 'audits', 'just', 'formalities', 'reports', 'full', 'lies', 'transparency', 'slogans', 'disguise', 'small', 'fry', 'big', 'bosses', 'play', 'too', 'taxes', 'permits', 'through', 'illegal', 'routes', 'stay', 'silent', 'future', 'generations', 'inherit', 'higher', 'legitimate', 'raise', 'voice', 'fear', 'open', 'eyes', 'ears', 'care', 'break', 'system', 'fragile', 'change', 'slaves', 'united', 'win', 'anymore', 'dream', 'reality']
        
        # Kata-kata umum Indonesian
        indonesian_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'ini', 'itu', 'akan', 'telah', 'sudah', 'belum', 'tidak', 'bukan', 'atau', 'juga', 'hanya', 'saja', 'lebih', 'sangat', 'sekali', 'masih', 'tetap', 'selalu', 'pernah', 'mungkin', 'bisa', 'dapat', 'harus', 'perlu', 'ingin', 'mau', 'sedang', 'lagi', 'baru', 'lama', 'besar', 'kecil', 'tinggi', 'rendah', 'baik', 'buruk', 'benar', 'salah', 'mudah', 'sulit', 'cepat', 'lambat', 'banyak', 'sedikit', 'semua', 'setiap', 'beberapa', 'ada', 'balik', 'gedung', 'kaca', 'uang', 'diam', 'mengalir', 'janji', 'rakyat', 'dijual', 'suara', 'nurani', 'dikaburkan', 'struktur', 'rapat', 'sistem', 'retak', 'tapi', 'mereka', 'tertawa', 'korupsi', 'terstruktur', 'jadi', 'budaya', 'nyata', 'pintu', 'tertutup', 'kontrak', 'main', 'belakang', 'proyek', 'diambil', 'sementara', 'cuma', 'kata', 'pusat', 'sampai', 'daerah', 'rantai', 'hitam', 'membentang', 'hukum', 'lempar', 'tangan', 'keadilan', 'sering', 'hilang', 'jangan', 'kau', 'bungkam', 'lihat', 'takut', 'bicara', 'kotor', 'meja', 'sandera', 'kita', 'lawan', 'bersama', 'masuk', 'kantong', 'moral', 'pergi', 'entah', 'kemana', 'pejabat', 'ganti', 'topeng', 'tetap', 'merana', 'audit', 'formalitas', 'laporan', 'penuh', 'tipu', 'daya', 'transparansi', 'tinggal', 'slogan', 'hanya', 'maya', 'orang', 'bos', 'ikut', 'pajak', 'izin', 'jalur', 'haram', 'diam', 'terus', 'anak', 'cucu', 'bakal', 'warisi', 'makin', 'tinggi', 'murni', 'bangkit', 'mata', 'telinga', 'semua', 'peduli', 'patah', 'rapuh', 'bisa', 'ubah', 'bukan', 'budak', 'bersatu', 'takkan', 'menang', 'lagi', 'mimpi', 'realita', 'demi', 'masa', 'depan']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        indonesian_count = sum(1 for word in indonesian_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 'unknown'
        
        # Return language dengan confidence
        if english_count > indonesian_count:
            return 'english'
        elif indonesian_count > english_count:
            return 'indonesian'
        else:
            return 'mixed'

    def _extract_text_from_audio(self):
        """AI mengekstrak teks dari audio dengan segment yang lebih besar untuk kalimat panjang"""
        self.logger.info("AI extracting text from audio with enhanced segmentation for long sentences...")
        
        try:
            result = self.whisper_model.transcribe(
                self.audio_path, 
                word_timestamps=True,
                verbose=False
            )
            
            words_with_timestamps = []
            extracted_text_lines = []
            
            # Gabungkan segment yang kecil menjadi segment yang lebih besar untuk kalimat panjang
            combined_segments = []
            current_segment = None
            
            for segment in result['segments']:
                if current_segment is None:
                    current_segment = {
                        'text': segment['text'].strip(),
                        'start': segment['start'],
                        'end': segment['end'],
                        'words': segment.get('words', [])
                    }
                else:
                    # Gabungkan jika segment terlalu pendek atau berdekatan
                    segment_duration = segment['end'] - segment['start']
                    gap = segment['start'] - current_segment['end']
                    
                    if segment_duration < 2.0 and gap < 1.0:  # Gabungkan segment pendek yang berdekatan
                        current_segment['text'] += " " + segment['text'].strip()
                        current_segment['end'] = segment['end']
                        current_segment['words'].extend(segment.get('words', []))
                    else:
                        combined_segments.append(current_segment)
                        current_segment = {
                            'text': segment['text'].strip(),
                            'start': segment['start'],
                            'end': segment['end'],
                            'words': segment.get('words', [])
                        }
            
            if current_segment:
                combined_segments.append(current_segment)
            
            for segment in combined_segments:
                segment_text = segment['text']
                segment_language = self._detect_language(segment_text)
                
                # Simpan segment text dengan timing
                extracted_text_lines.append(f"[{segment['start']:.2f}s-{segment['end']:.2f}s] ({segment_language}) {segment_text}")
                
                for word in segment['words']:
                    word_data = {
                        'word': word['word'].strip(),
                        'start': word['start'],
                        'end': word['end'],
                        'language': segment_language
                    }
                    words_with_timestamps.append(word_data)
            
            # Simpan hasil ekstraksi ke file
            with open(self.extracted_lyrics_file, 'w', encoding='utf-8') as f:
                f.write("=== EXTRACTED LYRICS FROM AUDIO (ENHANCED FOR LONG SENTENCES) ===\n")
                f.write(f"Audio file: {self.audio_path}\n")
                f.write(f"Original segments: {len(result['segments'])}\n")
                f.write(f"Combined segments: {len(combined_segments)}\n")
                f.write(f"Total words: {len(words_with_timestamps)}\n")
                f.write("=" * 60 + "\n\n")
                
                for line in extracted_text_lines:
                    f.write(line + "\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("=== WORD-LEVEL TIMESTAMPS ===\n")
                for word_data in words_with_timestamps:
                    f.write(f"[{word_data['start']:.2f}s-{word_data['end']:.2f}s] {word_data['word']}\n")
            
            self.logger.info(f"✅ Enhanced extracted lyrics saved to: {self.extracted_lyrics_file}")
            self.logger.info(f"AI extracted {len(words_with_timestamps)} words from {len(combined_segments)} combined segments")
            return words_with_timestamps
        except Exception as e:
            self.logger.error(f"Error during AI text extraction: {e}")
            return None

    def _load_comparison_lyrics(self):
        """Load comparison lyrics dari file"""
        if not os.path.exists(self.comparison_lyrics_file):
            self.logger.warning(f"Comparison lyrics file not found: {self.comparison_lyrics_file}")
            return []
        
        try:
            with open(self.comparison_lyrics_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            lyrics = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('='):
                    lyrics.append(line)
            
            self.logger.info(f"✅ Loaded {len(lyrics)} comparison lyrics from: {self.comparison_lyrics_file}")
            return lyrics
        except Exception as e:
            self.logger.error(f"Error loading comparison lyrics: {e}")
            return []

    def _build_multi_scale_faiss_index(self, extracted_words):
        """Build Faiss index dengan multiple scales untuk menangkap kalimat panjang"""
        self.logger.info("Building multi-scale Faiss index for comprehensive long sentence matching...")
        
        text_chunks = []
        chunk_timings = []
        
        # Scale 1: Small chunks (3-6 words) untuk kata kunci
        small_chunk_size = 4
        small_overlap = 2
        
        i = 0
        while i < len(extracted_words):
            chunk_words = extracted_words[i:i + small_chunk_size]
            if len(chunk_words) < 2:
                break
                
            chunk_text = " ".join([w['word'] for w in chunk_words])
            chunk_language = self._detect_language(chunk_text)
            
            text_chunks.append(chunk_text)
            chunk_timings.append({
                'start': chunk_words[0]['start'],
                'end': chunk_words[-1]['end'],
                'words': chunk_words,
                'language': chunk_language,
                'scale': 'small',
                'word_count': len(chunk_words)
            })
            
            i += small_chunk_size - small_overlap
        
        # Scale 2: Medium chunks (6-12 words) untuk frase
        medium_chunk_size = 8
        medium_overlap = 3
        
        i = 0
        while i < len(extracted_words):
            chunk_words = extracted_words[i:i + medium_chunk_size]
            if len(chunk_words) < 4:
                break
                
            chunk_text = " ".join([w['word'] for w in chunk_words])
            chunk_language = self._detect_language(chunk_text)
            
            text_chunks.append(chunk_text)
            chunk_timings.append({
                'start': chunk_words[0]['start'],
                'end': chunk_words[-1]['end'],
                'words': chunk_words,
                'language': chunk_language,
                'scale': 'medium',
                'word_count': len(chunk_words)
            })
            
            i += medium_chunk_size - medium_overlap
        
        # Scale 3: Large chunks (12-20 words) untuk kalimat panjang
        large_chunk_size = 16
        large_overlap = 4
        
        i = 0
        while i < len(extracted_words):
            chunk_words = extracted_words[i:i + large_chunk_size]
            if len(chunk_words) < 8:
                break
                
            chunk_text = " ".join([w['word'] for w in chunk_words])
            chunk_language = self._detect_language(chunk_text)
            
            text_chunks.append(chunk_text)
            chunk_timings.append({
                'start': chunk_words[0]['start'],
                'end': chunk_words[-1]['end'],
                'words': chunk_words,
                'language': chunk_language,
                'scale': 'large',
                'word_count': len(chunk_words)
            })
            
            i += large_chunk_size - large_overlap
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(text_chunks)
        
        # Create Faiss index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        self.logger.info(f"Multi-scale Faiss index built with {len(text_chunks)} chunks (small: {small_chunk_size}, medium: {medium_chunk_size}, large: {large_chunk_size})")
        return text_chunks, chunk_timings

    def _multi_scale_semantic_search(self, lyric_text, text_chunks, chunk_timings, top_k=25):
        """Multi-scale semantic search untuk menangkap kalimat panjang"""
        lyric_language = self._detect_language(lyric_text)
        word_count = len(lyric_text.split())
        
        lyric_embedding = self.sentence_model.encode([lyric_text])
        faiss.normalize_L2(lyric_embedding)
        
        similarities, indices = self.faiss_index.search(lyric_embedding, top_k)
        
        best_matches = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(chunk_timings):
                chunk_language = chunk_timings[idx]['language']
                chunk_scale = chunk_timings[idx]['scale']
                chunk_word_count = chunk_timings[idx]['word_count']
                
                # Boost similarity berdasarkan berbagai faktor
                adjusted_similarity = similarity
                
                # 1. Language match boost
                if lyric_language == chunk_language:
                    adjusted_similarity *= 1.4
                elif lyric_language == 'mixed' or chunk_language == 'mixed':
                    adjusted_similarity *= 1.2
                
                # 2. Scale preference untuk kalimat panjang
                if word_count > 8:  # Long sentence
                    if chunk_scale == 'large':
                        adjusted_similarity *= 1.3
                    elif chunk_scale == 'medium':
                        adjusted_similarity *= 1.1
                else:  # Short sentence
                    if chunk_scale == 'small':
                        adjusted_similarity *= 1.2
                    elif chunk_scale == 'medium':
                        adjusted_similarity *= 1.1
                
                # 3. Word count similarity boost
                word_count_ratio = min(word_count, chunk_word_count) / max(word_count, chunk_word_count)
                if word_count_ratio > 0.7:  # Similar word count
                    adjusted_similarity *= 1.2
                
                best_matches.append({
                    'similarity': float(adjusted_similarity),
                    'original_similarity': float(similarity),
                    'text': text_chunks[idx],
                    'timing': chunk_timings[idx],
                    'language_match': lyric_language == chunk_language,
                    'scale_match': chunk_scale,
                    'word_count': word_count,
                    'chunk_word_count': chunk_word_count
                })
        
        return best_matches

    def _calculate_adaptive_duration(self, lyric_text, timing, extracted_words):
        """Hitung durasi yang adaptif berdasarkan panjang kalimat dan konteks"""
        lyric_language = self._detect_language(lyric_text)
        word_count = len(lyric_text.split())
        
        # Base duration dengan formula yang lebih canggih
        if word_count <= 3:
            base_duration = word_count * 0.6  # Very short
        elif word_count <= 6:
            base_duration = word_count * 0.5  # Short
        elif word_count <= 10:
            base_duration = word_count * 0.4  # Medium
        elif word_count <= 15:
            base_duration = word_count * 0.35  # Long
        else:
            base_duration = word_count * 0.3  # Very long
        
        # Language adjustment
        if lyric_language == 'english':
            base_duration *= 1.05
        elif lyric_language == 'indonesian':
            base_duration *= 1.25
        
        # Contextual adjustment berdasarkan kecepatan vocal
        nearby_words = [w for w in extracted_words if abs(w['start'] - timing['start']) < 10.0]
        if nearby_words:
            avg_word_duration = sum(w['end'] - w['start'] for w in nearby_words) / len(nearby_words)
            if avg_word_duration < 0.25:  # Very fast vocals
                base_duration *= 0.6
            elif avg_word_duration < 0.4:  # Fast vocals
                base_duration *= 0.8
            elif avg_word_duration > 0.8:  # Slow vocals
                base_duration *= 1.3
        
        # Adaptive min/max duration
        min_duration = max(1.0, word_count * 0.2)
        max_duration = min(20.0, word_count * 0.8)
        
        return max(min_duration, min(max_duration, base_duration))

    def _match_lyrics_advanced(self, lyrics, extracted_words):
        """Advanced lyric matching dengan multi-scale approach"""
        self.logger.info("Advanced lyric matching with multi-scale approach for long sentences...")
        
        if not extracted_words:
            self.logger.warning("No extracted words available, using fallback timing")
            return self._fallback_timing(lyrics, self._get_audio_duration())

        matched_lyrics_with_timing = []
        duration = self._get_audio_duration()
        
        # Build multi-scale Faiss index
        text_chunks, chunk_timings = self._build_multi_scale_faiss_index(extracted_words)
        
        # Match lyrics dengan multi-scale semantic search
        matched_indices = set()
        
        for i, lyric_line in enumerate(lyrics):
            lyric_language = self._detect_language(lyric_line)
            word_count = len(lyric_line.split())
            self.logger.info(f"Processing lyric {i+1} ({lyric_language}, {word_count} words): '{lyric_line}'")
            
            # Adaptive top_k berdasarkan panjang kalimat
            top_k = 30 if word_count > 12 else 20 if word_count > 8 else 15
            matches = self._multi_scale_semantic_search(lyric_line, text_chunks, chunk_timings, top_k=top_k)
            
            best_match = None
            for match in matches:
                # Adaptive threshold berdasarkan panjang kalimat
                if word_count > 12:
                    threshold = 0.15  # Very low threshold for very long sentences
                elif word_count > 8:
                    threshold = 0.2   # Low threshold for long sentences
                else:
                    threshold = 0.25  # Normal threshold for short sentences
                
                if match['similarity'] >= threshold:
                    timing = match['timing']
                    
                    # Hitung durasi adaptif
                    optimal_duration = self._calculate_adaptive_duration(lyric_line, timing, extracted_words)
                    
                    # Validasi durasi dengan range yang sangat luas
                    min_duration = max(0.8, word_count * 0.15)
                    max_duration = min(25.0, word_count * 1.0)
                    
                    if min_duration <= optimal_duration <= max_duration:
                        best_match = match
                        best_match['optimal_duration'] = optimal_duration
                        break
            
            if best_match:
                timing = best_match['timing']
                optimal_duration = best_match['optimal_duration']
                
                # Adjust timing dengan durasi optimal
                adjusted_end = timing['start'] + optimal_duration
                
                matched_lyrics_with_timing.append({
                    'text': lyric_line,
                    'start': timing['start'],
                    'end': adjusted_end,
                    'language': lyric_language,
                    'word_count': word_count,
                    'match_method': 'multi_scale_semantic',
                    'similarity': best_match['similarity'],
                    'language_match': best_match['language_match'],
                    'scale_match': best_match['scale_match']
                })
                matched_indices.add(i)
                self.logger.info(f"✅ Multi-scale matched lyric {i+1} ({lyric_language}, {word_count} words, {best_match['scale_match']}): '{lyric_line}' from {timing['start']:.2f}s to {adjusted_end:.2f}s (similarity: {best_match['similarity']:.2f})")
        
        # Enhanced fallback untuk lyrics yang belum match
        unmatched_lyrics = []
        for i, lyric in enumerate(lyrics):
            if i not in matched_indices:
                unmatched_lyrics.append((i, lyric))
        
        if unmatched_lyrics:
            self.logger.info(f"Using enhanced fallback timing for {len(unmatched_lyrics)} unmatched lyrics...")
            for lyric_idx, lyric_line in unmatched_lyrics:
                lyric_language = self._detect_language(lyric_line)
                word_count = len(lyric_line.split())
                
                # Enhanced fallback dengan distribusi yang lebih baik
                start_time = (lyric_idx / len(lyrics)) * duration * 0.9
                
                # Adaptive duration per word berdasarkan bahasa dan panjang
                if lyric_language == 'english':
                    duration_per_word = 0.35 if word_count > 12 else 0.4 if word_count > 8 else 0.5
                elif lyric_language == 'indonesian':
                    duration_per_word = 0.45 if word_count > 12 else 0.5 if word_count > 8 else 0.6
                else:
                    duration_per_word = 0.4 if word_count > 12 else 0.45 if word_count > 8 else 0.55
                
                lyric_duration = word_count * duration_per_word
                end_time = start_time + lyric_duration
                
                matched_lyrics_with_timing.append({
                    'text': lyric_line,
                    'start': start_time,
                    'end': end_time,
                    'language': lyric_language,
                    'word_count': word_count,
                    'match_method': 'enhanced_fallback'
                })
                self.logger.info(f"📝 Enhanced fallback for lyric {lyric_idx+1} ({lyric_language}, {word_count} words): '{lyric_line}' from {start_time:.2f}s to {end_time:.2f}s")
        
        # Sort by start time
        matched_lyrics_with_timing.sort(key=lambda x: x['start'])
        
        # Simpan hasil matching ke file
        self._save_matched_lyrics(matched_lyrics_with_timing)
        
        # Validasi dan perbaiki timing
        return self._validate_and_fix_advanced_timing(matched_lyrics_with_timing, duration)

    def _save_matched_lyrics(self, matched_lyrics):
        """Simpan hasil matching ke file dengan detail yang lebih lengkap"""
        try:
            with open(self.matched_lyrics_file, 'w', encoding='utf-8') as f:
                f.write("=== ADVANCED MATCHED LYRICS RESULT ===\n")
                f.write(f"Total matched lyrics: {len(matched_lyrics)}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, lyric_entry in enumerate(matched_lyrics):
                    f.write(f"Lyric {i+1}:\n")
                    f.write(f"  Text: {lyric_entry['text']}\n")
                    f.write(f"  Timing: {lyric_entry['start']:.2f}s - {lyric_entry['end']:.2f}s\n")
                    f.write(f"  Language: {lyric_entry['language']}\n")
                    f.write(f"  Word Count: {lyric_entry['word_count']}\n")
                    f.write(f"  Match Method: {lyric_entry['match_method']}\n")
                    if 'similarity' in lyric_entry:
                        f.write(f"  Similarity: {lyric_entry['similarity']:.2f}\n")
                    if 'scale_match' in lyric_entry:
                        f.write(f"  Scale Match: {lyric_entry['scale_match']}\n")
                    f.write("\n")
            
            self.logger.info(f"✅ Advanced matched lyrics saved to: {self.matched_lyrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving matched lyrics: {e}")

    def _validate_and_fix_advanced_timing(self, lyric_data, duration):
        """Validasi dan perbaiki timing dengan advanced algorithm"""
        self.logger.info("Validating and fixing advanced timing...")
        
        if not lyric_data:
            return self._fallback_timing([], duration)
        
        # Validasi durasi lyric dengan awareness bahasa dan panjang kalimat
        for i, lyric_entry in enumerate(lyric_data):
            lyric_duration = lyric_entry['end'] - lyric_entry['start']
            language = lyric_entry.get('language', 'unknown')
            word_count = lyric_entry.get('word_count', 0)
            
            # Adaptive durasi minimum dan maksimum
            if language == 'english':
                min_duration = max(0.8, word_count * 0.2) if word_count > 8 else max(0.6, word_count * 0.25)
                max_duration = min(15.0, word_count * 0.7) if word_count > 8 else min(8.0, word_count * 0.6)
            elif language == 'indonesian':
                min_duration = max(1.0, word_count * 0.25) if word_count > 8 else max(0.8, word_count * 0.3)
                max_duration = min(18.0, word_count * 0.8) if word_count > 8 else min(10.0, word_count * 0.7)
            else:
                min_duration = max(0.9, word_count * 0.22) if word_count > 8 else max(0.7, word_count * 0.27)
                max_duration = min(16.0, word_count * 0.75) if word_count > 8 else min(9.0, word_count * 0.65)
            
            if lyric_duration < min_duration:
                lyric_entry['end'] = lyric_entry['start'] + min_duration
                self.logger.info(f"🔧 Extended short duration for {language} lyric {i+1} ({word_count} words): '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
            elif lyric_duration > max_duration:
                lyric_entry['end'] = lyric_entry['start'] + max_duration
                self.logger.info(f"🔧 Shortened long duration for {language} lyric {i+1} ({word_count} words): '{lyric_entry['text']}' now {lyric_entry['end'] - lyric_entry['start']:.2f}s")
        
        # Sort by start time
        lyric_data.sort(key=lambda x: x['start'])
        
        # Perbaiki overlap dengan advanced algorithm
        for i in range(1, len(lyric_data)):
            prev_end = lyric_data[i-1]['end']
            curr_start = lyric_data[i]['start']
            prev_lang = lyric_data[i-1].get('language', 'unknown')
            curr_lang = lyric_data[i].get('language', 'unknown')
            prev_words = lyric_data[i-1].get('word_count', 0)
            curr_words = lyric_data[i].get('word_count', 0)
            
            # Adaptive overlap threshold
            overlap_threshold = 0.4 if curr_words > 12 else 0.3 if curr_words > 8 else 0.2
            
            if curr_start < prev_end - overlap_threshold:
                # Advanced overlap resolution
                if prev_lang == curr_lang:
                    # Same language - minimal overlap
                    lyric_data[i]['start'] = prev_end
                else:
                    # Different language - allow transition overlap
                    overlap_allowance = 0.8 if curr_words > 12 else 0.6 if curr_words > 8 else 0.4
                    lyric_data[i]['start'] = prev_end - overlap_allowance
                
                self.logger.info(f"🔧 Fixed overlap for {prev_lang}->{curr_lang} lyric {i+1} ({curr_words} words): '{lyric_data[i]['text']}' now starts at {lyric_data[i]['start']:.2f}s")
        
        # Log final timing dengan detail lengkap
        for i, lyric_entry in enumerate(lyric_data):
            language = lyric_entry.get('language', 'unknown')
            word_count = lyric_entry.get('word_count', 0)
            match_method = lyric_entry.get('match_method', 'unknown')
            self.logger.info(f"✅ Final timing for lyric {i+1} ({language}, {word_count} words, {match_method}): '{lyric_entry['text']}' from {lyric_entry['start']:.2f}s to {lyric_entry['end']:.2f}s")
        
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

    def _create_smooth_frames(self, image, duration, fps=24, lyric_data=None):
        frames = []
        total_frames = int(duration * fps)
        
        width, height = image.size
        
        try:
            font = ImageFont.truetype("arial.ttf", 42)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
            except:
                font = ImageFont.load_default()
        
        # GPU optimization untuk RTX 4090
        batch_size = 100
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for i in range(batch_start, batch_end):
                progress = i / total_frames
                time_in_video = i / fps

                # Complex smooth movement
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
                    current_language = "unknown"
                    
                    for entry in lyric_data:
                        if entry['start'] <= time_in_video < entry['end']:
                            current_lyric_text = entry['text']
                            current_start_time = entry['start']
                            current_end_time = entry['end']
                            current_language = entry.get('language', 'unknown')
                            break
                    
                    if current_lyric_text:
                        draw = ImageDraw.Draw(rotated_image)
                        bbox = draw.textbbox((0, 0), current_lyric_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Position text at bottom center
                        text_x = (width - text_width) // 2
                        text_y = height - text_height - 150
                        
                        # Adaptive fade effect berdasarkan panjang kalimat
                        word_count = len(current_lyric_text.split())
                        if current_language == 'indonesian':
                            fade_duration = 1.0 if word_count > 12 else 0.8 if word_count > 8 else 0.6
                        else:
                            fade_duration = 0.8 if word_count > 12 else 0.6 if word_count > 8 else 0.4
                        
                        alpha = 0.0
                        
                        if time_in_video < current_start_time + fade_duration:
                            alpha = (time_in_video - current_start_time) / fade_duration
                        elif time_in_video > current_end_time - fade_duration:
                            alpha = (current_end_time - time_in_video) / fade_duration
                        else:
                            alpha = 1.0
                        
                        alpha_int = int(alpha * 255)
                        
                        # Color coding berdasarkan bahasa
                        if current_language == 'english':
                            text_color = (255, 255, 255, alpha_int)  # White for English
                        elif current_language == 'indonesian':
                            text_color = (255, 255, 0, alpha_int)   # Yellow for Indonesian
                        else:
                            text_color = (255, 255, 255, alpha_int)  # White for mixed/unknown
                        
                        # Draw text with stroke
                        draw.text((text_x, text_y), current_lyric_text, 
                                 font=font, 
                                 fill=text_color, 
                                 stroke_fill=(0, 0, 0, alpha_int), 
                                 stroke_width=2)

                frames.append(rotated_image)
            
            # Progress logging
            batch_progress = (batch_end / total_frames) * 100
            self.logger.info(f"Creating frames: {batch_progress:.1f}% (Batch {batch_start//batch_size + 1})")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info("Creating frames: 100.0%")
        return frames

    def generate_video(self):
        self.logger.info("=== Advanced Long Sentence AI Lyric Synchronization Video Generator ===")
        self.logger.info("Using multi-scale approach with enhanced long sentence matching...")

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
        
        # Step 1: AI mengekstrak teks dari audio dengan enhanced segmentation
        extracted_words = self._extract_text_from_audio()
        
        # Step 2: Load comparison lyrics dari file
        comparison_lyrics = self._load_comparison_lyrics()
        
        if not comparison_lyrics:
            self.logger.error("No comparison lyrics found. Please create comparison_lyrics.txt file.")
            return
        
        # Step 3: AI mencocokkan teks dengan multi-scale approach
        lyric_data = self._match_lyrics_advanced(comparison_lyrics, extracted_words)
        
        # Step 4: Buat video dengan advanced text
        frames = self._create_smooth_frames(image, duration, fps=24, lyric_data=lyric_data)
        
        # Convert to numpy arrays dengan GPU optimization
        numpy_frames = []
        batch_size = 50
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                batch_numpy = []
                for frame in batch_frames:
                    batch_numpy.append(np.array(frame))
                
                numpy_frames.extend(batch_numpy)
                
                torch.cuda.empty_cache()
            else:
                for frame in batch_frames:
                    numpy_frames.append(np.array(frame))
            
            progress = (i + len(batch_frames)) / len(frames) * 100
            self.logger.info(f"Converting frames to numpy: {progress:.1f}% (Batch {i//batch_size + 1})")
        
        # Create video
        video_clip = ImageSequenceClip(numpy_frames, fps=24)
        audio_clip = AudioFileClip(self.audio_path)
        final_video = video_clip.set_audio(audio_clip)

        try:
            final_video.write_videofile(
                self.output_path, 
                fps=24,
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile="temp-audio.m4a", 
                remove_temp=True
            )
            self.logger.info(f"✅ Advanced long sentence AI-synchronized video created successfully: {self.output_path}")
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")


if __name__ == "__main__":
    image_file = "context/rapper.jpg"
    audio_file = "context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3"
    output_video_file = "context/advanced_long_sentence_video.mp4"

    generator = AdvancedLongSentenceLyricSyncGenerator(image_file, audio_file, output_video_file)
    generator.generate_video()
