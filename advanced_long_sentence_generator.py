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

    def _calculate_text_similarity(self, text1, text2):
        """Hitung similarity antara dua text dengan fuzzy matching"""
        # Normalize text
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return 1.0
        
        # Check substring match (high similarity)
        if t1 in t2 or t2 in t1:
            return 0.95
        
        # Word overlap ratio
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        word_overlap = len(intersection) / len(union) if union else 0.0
        
        # Character similarity (fuzzy)
        char_similarity = SequenceMatcher(None, t1, t2).ratio()
        
        # Combined score (word overlap lebih penting)
        return (word_overlap * 0.7) + (char_similarity * 0.3)

    def _exact_word_match(self, lyric_text, extracted_words):
        """Cari exact word match dalam extracted words untuk akurasi maksimal"""
        lyric_words = [w.lower().strip() for w in lyric_text.split()]
        if not lyric_words:
            return None
        
        # Cari sequence words yang match
        best_match = None
        best_score = 0
        
        # Sliding window untuk mencari sequence
        for i in range(len(extracted_words) - len(lyric_words) + 1):
            window = extracted_words[i:i+len(lyric_words)]
            window_text = " ".join([w['word'].lower().strip() for w in window])
            lyric_text_lower = " ".join(lyric_words)
            
            # Hitung similarity
            similarity = self._calculate_text_similarity(lyric_text_lower, window_text)
            
            if similarity > best_score:
                best_score = similarity
                # Jika similarity tinggi (>0.7), gunakan timing dari extracted words
                if similarity > 0.7:
                    start_time = window[0]['start']
                    end_time = window[-1]['end']
                    best_match = {
                        'start': start_time,
                        'end': end_time,
                        'similarity': similarity,
                        'method': 'exact_word_match',
                        'words': window
                    }
        
        return best_match

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
                chunk_text = text_chunks[idx].lower().strip()
                lyric_text_lower = lyric_text.lower().strip()
                
                # Tambahkan fuzzy text similarity untuk meningkatkan akurasi
                fuzzy_sim = self._calculate_text_similarity(lyric_text_lower, chunk_text)
                
                # Boost similarity berdasarkan berbagai faktor
                adjusted_similarity = similarity
                
                # 0. Fuzzy text matching boost (sangat penting)
                if fuzzy_sim > 0.8:
                    adjusted_similarity = max(adjusted_similarity, fuzzy_sim * 1.5)
                elif fuzzy_sim > 0.6:
                    adjusted_similarity = max(adjusted_similarity, fuzzy_sim * 1.2)
                
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
                    'fuzzy_similarity': fuzzy_sim,
                    'text': text_chunks[idx],
                    'timing': chunk_timings[idx],
                    'language_match': lyric_language == chunk_language,
                    'scale_match': chunk_scale,
                    'word_count': word_count,
                    'chunk_word_count': chunk_word_count
                })
        
        return best_matches

    def _phonetic_similarity(self, word1, word2):
        """Hitung phonetic similarity menggunakan simple phonetic algorithm"""
        def simple_phonetic(word):
            """Simple phonetic representation (Soundex-like)"""
            if not word:
                return ""
            word = str(word).lower().strip()
            if not word:
                return ""
            
            # Remove common suffixes
            word = re.sub(r'(ng|nya|mu|ku|lah|kah|pun)$', '', word)
            
            if not word:
                return ""
            
            # Convert to phonetic code
            phonetic = word[0]  # Keep first letter
            for char in word[1:]:
                if char in 'aeiouy':
                    phonetic += '0'  # Vowels
                elif char in 'bp':
                    phonetic += '1'
                elif char in 'ckq':
                    phonetic += '2'
                elif char in 'dt':
                    phonetic += '3'
                elif char in 'fg':
                    phonetic += '4'
                elif char in 'hj':
                    phonetic += '5'
                elif char in 'lm':
                    phonetic += '6'
                elif char in 'nr':
                    phonetic += '7'
                elif char in 'sx':
                    phonetic += '8'
                elif char in 'wz':
                    phonetic += '9'
                else:
                    phonetic += char
            return phonetic
        
        ph1 = simple_phonetic(word1)
        ph2 = simple_phonetic(word2)
        
        # Exact phonetic match
        if ph1 == ph2:
            return 1.0
        
        # Character similarity on phonetic codes
        if not ph1 or not ph2:
            return 0.0
        
        similarity = SequenceMatcher(None, ph1, ph2).ratio()
        return similarity
    
    def _hierarchical_forced_alignment(self, lyric_text, extracted_words):
        """Forced alignment dengan hierarchical approach: sentence -> phrase -> word"""
        lyric_words = [w.lower().strip() for w in lyric_text.split()]
        if not lyric_words:
            return None
        
        # Step 1: Find sentence-level match using sliding window
        # Try different window sizes for better matching
        min_window = max(1, len(lyric_words) - 2)
        max_window = min(len(extracted_words), len(lyric_words) + 5)
        
        best_match = None
        best_score = 0.0
        
        # Sliding window dengan berbagai ukuran
        for window_size in range(min_window, max_window + 1):
            for i in range(len(extracted_words) - window_size + 1):
                window = extracted_words[i:i+window_size]
                window_text = " ".join([w['word'].lower().strip() for w in window])
                lyric_text_lower = " ".join(lyric_words)
                
                # Calculate multiple similarity metrics
                # 1. Text similarity
                text_sim = self._calculate_text_similarity(lyric_text_lower, window_text)
                
                # 2. Phonetic similarity (word by word)
                phonetic_sims = []
                for j, lyric_word in enumerate(lyric_words):
                    if j < len(window):
                        phonetic_sims.append(self._phonetic_similarity(lyric_word, window[j]['word']))
                
                avg_phonetic_sim = sum(phonetic_sims) / len(phonetic_sims) if phonetic_sims else 0
                
                # 3. Word overlap
                lyric_words_set = set(lyric_words)
                window_words_set = set([w['word'].lower().strip() for w in window])
                overlap = len(lyric_words_set.intersection(window_words_set))
                word_overlap_ratio = overlap / max(len(lyric_words_set), len(window_words_set)) if (lyric_words_set or window_words_set) else 0
                
                # 4. Position similarity (favor matches that are in sequence)
                position_penalty = 0
                if len(window) == len(lyric_words):
                    # Check if words are in similar relative positions
                    for j, lyric_word in enumerate(lyric_words):
                        if j < len(window):
                            ph_sim = self._phonetic_similarity(lyric_word, window[j]['word'])
                            if ph_sim > 0.6:
                                position_penalty += 0.1
                    position_penalty = min(position_penalty / len(lyric_words), 0.3)
                
                # Combined score dengan weights
                combined_score = (
                    text_sim * 0.4 +           # Text similarity
                    avg_phonetic_sim * 0.3 +    # Phonetic similarity
                    word_overlap_ratio * 0.2 + # Word overlap
                    position_penalty           # Position bonus
                )
                
                # Boost score jika ada banyak matches
                if word_overlap_ratio > 0.5 and avg_phonetic_sim > 0.6:
                    combined_score *= 1.2
                
                if combined_score > best_score:
                    best_score = combined_score
                    start_time = window[0]['start']
                    end_time = window[-1]['end']
                    best_match = {
                        'start': start_time,
                        'end': end_time,
                        'similarity': combined_score,
                        'text_similarity': text_sim,
                        'phonetic_similarity': avg_phonetic_sim,
                        'word_overlap': word_overlap_ratio,
                        'method': 'hierarchical_forced_alignment',
                        'words': window
                    }
        
        return best_match if best_score > 0.5 else None  # Lower threshold untuk forced alignment
    
    def _sliding_window_forced_match(self, lyric_text, extracted_words, start_search_time=0, end_search_time=None):
        """Sliding window forced matching dengan DTW-like approach"""
        lyric_words = [w.lower().strip() for w in lyric_text.split()]
        if not lyric_words or not extracted_words:
            return None
        
        # Filter extracted words by time range
        filtered_words = [w for w in extracted_words if w['start'] >= start_search_time]
        if end_search_time:
            filtered_words = [w for w in filtered_words if w['start'] <= end_search_time]
        
        if len(filtered_words) < len(lyric_words):
            # If not enough words, use all available words with padding
            filtered_words = extracted_words
        
        if not filtered_words:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Try multiple window sizes around the expected length
        for offset in range(-3, 4):  # Allow -3 to +3 word difference
            window_size = len(lyric_words) + offset
            if window_size < 1 or window_size > len(filtered_words):
                continue
            
            for i in range(len(filtered_words) - window_size + 1):
                window = filtered_words[i:i+window_size]
                window_text = " ".join([w['word'].lower().strip() for w in window])
                lyric_text_lower = " ".join(lyric_words)
                
                # Calculate similarity
                text_sim = self._calculate_text_similarity(lyric_text_lower, window_text)
                
                # Phonetic matching for individual words
                phonetic_matches = 0
                for j, lyric_word in enumerate(lyric_words):
                    # Try to match with nearby words in window
                    search_range = min(3, len(window) - j)
                    for k in range(search_range):
                        if j + k < len(window):
                            ph_sim = self._phonetic_similarity(lyric_word, window[j+k]['word'])
                            if ph_sim > 0.6:
                                phonetic_matches += 1
                                break
                
                phonetic_ratio = phonetic_matches / len(lyric_words) if lyric_words else 0
                
                # Combined score
                combined_score = text_sim * 0.6 + phonetic_ratio * 0.4
                
                # Boost jika ada banyak phonetic matches
                if phonetic_ratio > 0.7:
                    combined_score *= 1.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    start_time = window[0]['start']
                    end_time = window[-1]['end']
                    best_match = {
                        'start': start_time,
                        'end': end_time,
                        'similarity': combined_score,
                        'text_similarity': text_sim,
                        'phonetic_ratio': phonetic_ratio,
                        'method': 'sliding_window_forced',
                        'words': window
                    }
        
        return best_match if best_score > 0.4 else None  # Lower threshold untuk forced alignment
    
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
        
        # Track last matched time for sequential matching
        last_matched_end = 0.0
        
        for i, lyric_line in enumerate(lyrics):
            lyric_language = self._detect_language(lyric_line)
            word_count = len(lyric_line.split())
            self.logger.info(f"Processing lyric {i+1} ({lyric_language}, {word_count} words): '{lyric_line}'")
            
            matched = False
            
            # STEP 1: Coba forced alignment dengan hierarchical approach (terbaik untuk missing lyrics)
            forced_match = self._hierarchical_forced_alignment(lyric_line, extracted_words)
            if forced_match and forced_match['similarity'] > 0.5:
                # Use forced alignment result
                matched_lyrics_with_timing.append({
                    'text': lyric_line,
                    'start': forced_match['start'],
                    'end': forced_match['end'],
                    'language': lyric_language,
                    'word_count': word_count,
                    'match_method': 'forced_alignment_hierarchical',
                    'similarity': forced_match['similarity'],
                    'phonetic_similarity': forced_match.get('phonetic_similarity', 0.0),
                    'word_overlap': forced_match.get('word_overlap', 0.0)
                })
                matched_indices.add(i)
                matched = True
                last_matched_end = max(last_matched_end, forced_match['end'])
                self.logger.info(f"✅ Forced alignment matched lyric {i+1} ({lyric_language}, {word_count} words): '{lyric_line}' from {forced_match['start']:.2f}s to {forced_match['end']:.2f}s (sim: {forced_match['similarity']:.2f}, phonetic: {forced_match.get('phonetic_similarity', 0):.2f})")
            
            # STEP 2: Jika forced alignment gagal, coba sliding window forced match
            if not matched:
                # Search from last matched position forward
                sliding_match = self._sliding_window_forced_match(lyric_line, extracted_words, start_search_time=max(0, last_matched_end - 2.0))
                if sliding_match and sliding_match['similarity'] > 0.4:
                    matched_lyrics_with_timing.append({
                        'text': lyric_line,
                        'start': sliding_match['start'],
                        'end': sliding_match['end'],
                        'language': lyric_language,
                        'word_count': word_count,
                        'match_method': 'forced_alignment_sliding',
                        'similarity': sliding_match['similarity'],
                        'phonetic_ratio': sliding_match.get('phonetic_ratio', 0.0)
                    })
                    matched_indices.add(i)
                    matched = True
                    last_matched_end = max(last_matched_end, sliding_match['end'])
                    self.logger.info(f"✅ Sliding window forced match lyric {i+1} ({lyric_language}, {word_count} words): '{lyric_line}' from {sliding_match['start']:.2f}s to {sliding_match['end']:.2f}s (sim: {sliding_match['similarity']:.2f}, phonetic: {sliding_match.get('phonetic_ratio', 0):.2f})")
            
            # STEP 3: Coba exact word match (paling akurat tapi mungkin terlewat jika transcription salah)
            if not matched:
                exact_match = self._exact_word_match(lyric_line, extracted_words)
                if exact_match and exact_match['similarity'] > 0.7:
                    matched_lyrics_with_timing.append({
                        'text': lyric_line,
                        'start': exact_match['start'],
                        'end': exact_match['end'],
                        'language': lyric_language,
                        'word_count': word_count,
                        'match_method': 'exact_word_match',
                        'similarity': exact_match['similarity']
                    })
                    matched_indices.add(i)
                    matched = True
                    last_matched_end = max(last_matched_end, exact_match['end'])
                    self.logger.info(f"✅ Exact word matched lyric {i+1} ({lyric_language}, {word_count} words): '{lyric_line}' from {exact_match['start']:.2f}s to {exact_match['end']:.2f}s (similarity: {exact_match['similarity']:.2f})")
            
            # STEP 4: Jika semua forced alignment gagal, gunakan semantic search (fallback)
            if not matched:
                # Adaptive top_k berdasarkan panjang kalimat
                top_k = 30 if word_count > 12 else 20 if word_count > 8 else 15
                matches = self._multi_scale_semantic_search(lyric_line, text_chunks, chunk_timings, top_k=top_k)
                
                best_match = None
                for match in matches:
                    # Threshold yang lebih ketat untuk akurasi lebih baik
                    # Prioritaskan fuzzy similarity yang tinggi
                    fuzzy_sim = match.get('fuzzy_similarity', 0.0)
                
                    # Jika fuzzy similarity tinggi, gunakan threshold lebih rendah
                    if fuzzy_sim > 0.7:
                        threshold = 0.3  # Lebih ketat untuk fuzzy match yang baik
                    elif word_count > 12:
                        threshold = 0.4  # Lebih tinggi untuk very long sentences
                    elif word_count > 8:
                        threshold = 0.45  # Lebih tinggi untuk long sentences
                    else:
                        threshold = 0.5  # Lebih tinggi untuk short sentences
                    
                    # Prioritaskan match dengan fuzzy similarity tinggi
                    combined_score = match['similarity']
                    if fuzzy_sim > 0.7:
                        combined_score = max(combined_score, fuzzy_sim * 1.3)
                    
                    if combined_score >= threshold:
                        timing = match['timing']
                        
                        # Gunakan timing langsung dari extracted words jika memungkinkan
                        # Hitung durasi berdasarkan actual word timings
                        actual_duration = timing['end'] - timing['start']
                        
                        # Jika durasi terlalu pendek, perpanjang sedikit berdasarkan word count
                        expected_duration = word_count * 0.4  # Estimasi per kata
                        if actual_duration < expected_duration * 0.5:  # Durasi terlalu pendek
                            optimal_duration = self._calculate_adaptive_duration(lyric_line, timing, extracted_words)
                            # Gunakan maksimum antara actual dan calculated
                            optimal_duration = max(actual_duration, optimal_duration)
                        else:
                            # Gunakan actual duration dengan sedikit margin
                            optimal_duration = actual_duration * 1.1  # Tambah 10% margin
                        
                        # Validasi durasi dengan range yang wajar
                        min_duration = max(0.5, word_count * 0.2)
                        max_duration = min(15.0, word_count * 0.7)
                        
                        if min_duration <= optimal_duration <= max_duration:
                            best_match = match
                            best_match['optimal_duration'] = optimal_duration
                            break
                
                if best_match:
                    timing = best_match['timing']
                    optimal_duration = best_match['optimal_duration']
                    fuzzy_sim = best_match.get('fuzzy_similarity', 0.0)
                    
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
                        'fuzzy_similarity': fuzzy_sim,
                        'language_match': best_match['language_match'],
                        'scale_match': best_match['scale_match']
                    })
                    matched_indices.add(i)
                    matched = True
                    last_matched_end = max(last_matched_end, adjusted_end)
                    match_info = f"fuzzy:{fuzzy_sim:.2f}" if fuzzy_sim > 0 else f"similarity:{best_match['similarity']:.2f}"
                    self.logger.info(f"✅ Multi-scale matched lyric {i+1} ({lyric_language}, {word_count} words, {best_match['scale_match']}): '{lyric_line}' from {timing['start']:.2f}s to {adjusted_end:.2f}s ({match_info})")
        
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
                    if 'fuzzy_similarity' in lyric_entry:
                        f.write(f"  Fuzzy Similarity: {lyric_entry['fuzzy_similarity']:.2f}\n")
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

    def _wrap_text(self, text, max_chars_per_line, font, max_width):
        """Wrap text ke beberapa baris berdasarkan max characters dan lebar layar"""
        words = text.split()
        lines = []
        current_line = []
        
        # Buat dummy image untuk mengukur lebar teks
        dummy_img = Image.new('RGB', (100, 100))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        for word in words:
            # Cek panjang word jika ditambahkan ke current line
            test_line = ' '.join(current_line + [word])
            
            # Test lebar aktual dengan font
            try:
                bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
                test_width = bbox[2] - bbox[0]
            except:
                # Fallback ke estimasi karakter (asumsi rata-rata 10 pixel per karakter)
                test_width = len(test_line) * 10
            
            # Cek apakah perlu wrap: berdasarkan panjang karakter atau lebar pixel
            char_limit = len(test_line) > max_chars_per_line
            width_limit = test_width > max_width
            
            if (char_limit or width_limit) and current_line:
                # Simpan current line dan mulai baris baru
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Tambahkan word ke current line
                current_line.append(word)
        
        # Tambahkan baris terakhir jika ada
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]

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
                        
                        # Wrap text function - maksimum 50 karakter per baris
                        max_chars_per_line = 50
                        wrapped_lines = self._wrap_text(current_lyric_text, max_chars_per_line, font, width - 100)
                        
                        # Calculate total text height for multi-line text
                        # Ukur tinggi aktual dari satu baris teks
                        test_bbox = draw.textbbox((0, 0), "Test", font=font)
                        single_line_height = test_bbox[3] - test_bbox[1]
                        line_height = single_line_height + 10  # Spacing antar baris
                        total_text_height = len(wrapped_lines) * line_height
                        
                        # Position text lebih ke bawah untuk menghindari overlap dengan kontrol video
                        # 250 pixels dari bawah untuk memberikan ruang yang cukup
                        bottom_margin = 250
                        text_y = height - bottom_margin - total_text_height
                        
                        # Pastikan teks tidak terpotong di atas (minimal 20 pixels dari atas)
                        min_y = 20
                        if text_y < min_y:
                            text_y = min_y
                        
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
                        
                        # Draw multi-line text dengan proper centering
                        for line_num, line in enumerate(wrapped_lines):
                            bbox = draw.textbbox((0, 0), line, font=font)
                            line_width = bbox[2] - bbox[0]
                            text_x = (width - line_width) // 2
                            
                            current_y = text_y + (line_num * line_height)
                            
                            # Draw text with stroke untuk setiap baris
                            draw.text((text_x, current_y), line, 
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
    image_file = "context/chat.jpg"
    audio_file = "context/Chatmu Kayak Janji Negara.mp3"
    output_video_file = "context/advanced_long_sentence_video.mp4"

    generator = AdvancedLongSentenceLyricSyncGenerator(image_file, audio_file, output_video_file)
    generator.generate_video()
