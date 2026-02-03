import argparse
from pathlib import Path
import tempfile
import os
import sys
import shutil
import traceback
import subprocess

from agent_analyst import AgentAnalyst
from agent_analyst.chunking import ChunkingConfig, split_text_words, merge_analyses

from moviepy.editor import VideoFileClip
import whisper
import numpy as np

# ============================================
# SOLUTION ULTIME POUR FFMPEG
# ============================================

def force_ffmpeg_setup():
    """Forcer la configuration de FFmpeg pour Whisper"""
    # Trouver ffmpeg
    ffmpeg_path = None
    
    # Chercher via imageio
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
    
    # Chercher dans PATH
    if not ffmpeg_path:
        ffmpeg_path = shutil.which("ffmpeg")
    
    # Chercher dans emplacements courants
    if not ffmpeg_path:
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
    
    if not ffmpeg_path:
        print("❌ FFmpeg non trouvé!")
        print("   Installez: winget install ffmpeg")
        print("   ou téléchargez depuis: https://ffmpeg.org/download.html")
        return None
    
    print(f"✅ FFmpeg: {ffmpeg_path}")
    
    # 1. Pour moviepy
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    
    # 2. Pour whisper (TRÈS IMPORTANT)
    # Whisper utilise ffmpeg-python qui cherche dans le PATH
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    # 3. Forcer la réinitialisation de ffmpeg-python
    try:
        import ffmpeg
        # Réinitialiser le cache
        ffmpeg._run.LOGLEVEL = "error"
    except:
        pass
    
    return ffmpeg_path

# Configurer FFmpeg
ffmpeg_bin = force_ffmpeg_setup()

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}


def iter_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Lister toutes les vidéos"""
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]

    videos = [p for p in files if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    return sorted(videos)


def extract_and_transcribe_directly(video_path: Path) -> str:
    """Extraire et transcrire en une seule étape - ÉVITE les problèmes de fichiers temporaires"""
    print(f"  🎵 Traitement audio direct...")
    
    try:
        # OPTION 1: Whisper directement sur la vidéo
        global whisper_model
        
        print(f"  🔊 Transcription directe...")
        result = whisper_model.transcribe(
            str(video_path),  # Whisper peut lire directement les vidéos!
            language="fr",
            task="transcribe",
            fp16=False,
            verbose=False
        )
        
        text = result["text"].strip()
        print(f"  📝 {len(text.split())} mots")
        return text
        
    except Exception as e:
        print(f"  ⚠️  Méthode directe échouée: {e}")
        print(f"  🔄 Tentative avec extraction manuelle...")
        
        # OPTION 2: Extraire manuellement et transcrire
        return extract_and_transcribe_manual(video_path)


def extract_and_transcribe_manual(video_path: Path) -> str:
    """Alternative: extraire l'audio manuellement"""
    # Créer fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    
    try:
        # Extraire avec moviepy
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                raise ValueError("Pas d'audio")
            
            clip.audio.write_audiofile(
                str(audio_path),
                verbose=False,
                logger=None,
                fps=16000,
                codec='pcm_s16le'
            )
        
        # Transcrire avec numpy array pour éviter ffmpeg
        global whisper_model
        
        # Charger avec scipy ou wave
        try:
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(str(audio_path))
            audio_float = audio_data.astype(np.float32) / 32768.0
        except:
            # Fallback simple
            import wave
            import struct
            with wave.open(str(audio_path), 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio_float = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcrire
        result = whisper_model.transcribe(
            audio_float,
            language="fr",
            fp16=False
        )
        
        text = result["text"].strip()
        
        # Nettoyer
        os.unlink(audio_path)
        
        return text
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if audio_path.exists():
            try:
                os.unlink(audio_path)
            except:
                pass
        raise RuntimeError(f"Échec traitement manuel: {e}")


def transcribe_with_ffmpeg_fallback(video_path: Path) -> str:
    """Dernière option: utiliser ffmpeg en ligne de commande"""
    print(f"  🔧 Utilisation de FFmpeg en ligne de commande...")
    
    # Créer fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    
    try:
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg non disponible")
        
        # Extraire audio avec ffmpeg
        cmd = [
            ffmpeg_bin,
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-y",
            str(wav_path)
        ]
        
        # Exécuter
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg: {result.stderr[:200]}")
        
        # Charger et transcrire
        global whisper_model
        
        # Charger le WAV
        import wave
        import struct
        with wave.open(str(wav_path), 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcrire
        result = whisper_model.transcribe(
            audio_array,
            language="fr",
            fp16=False
        )
        
        text = result["text"].strip()
        
        # Nettoyer
        os.unlink(wav_path)
        
        return text
        
    except Exception as e:
        if wav_path.exists():
            try:
                os.unlink(wav_path)
            except:
                pass
        raise RuntimeError(f"Échec FFmpeg CLI: {e}")


def analyze_one(
    agent: AgentAnalyst,
    video_path: Path,
    output_path: Path,
    overwrite: bool,
    chunk_words: int,
    overlap_words: int,
    min_words_to_chunk: int,
) -> None:
    """Analyser une vidéo"""
    if output_path.exists() and not overwrite:
        print(f"⏭️  Déjà fait: {output_path.name}")
        return

    print(f"\n🎬 {video_path.name}")
    
    try:
        # Essayer plusieurs méthodes
        text = None
        methods = [
            ("Directe", extract_and_transcribe_directly),
            ("Manuelle", extract_and_transcribe_manual),
            ("FFmpeg CLI", transcribe_with_ffmpeg_fallback),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"  🔄 Méthode: {method_name}")
                text = method_func(video_path)
                if text and len(text.strip()) > 10:  # Au moins 10 caractères
                    print(f"  ✅ {method_name} réussie")
                    break
            except Exception as e:
                print(f"  ⚠️  {method_name} échouée: {str(e)[:100]}")
                continue
        
        if not text or len(text.strip()) == 0:
            print(f"  ❌ Toutes les méthodes ont échoué")
            return
        
        # Analyse
        cfg = ChunkingConfig(
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words_to_chunk=min_words_to_chunk,
        )

        chunks = split_text_words(text, cfg)
        if len(chunks) == 1:
            print(f"  🧠 Analyse...")
            analysis = agent.analyze_text(text)
        else:
            print(f"  🧩 {len(chunks)} segments")
            analyses = []
            for i, chunk in enumerate(chunks, start=1):
                print(f"    - Segment {i}...")
                analyses.append(agent.analyze_text(chunk))
            analysis = merge_analyses(analyses)

        # Sauvegarder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            analysis.model_dump_json(indent=2, ensure_ascii=False , exclude={"raw_model_output"}),
            encoding="utf-8",
        )

        print(f"✅ Succès: {output_path.name}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        traceback.print_exc()  # Pour débogage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyser des vidéos avec multiples méthodes de secours"
    )
    parser.add_argument("--input-dir", type=str, default="videos")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--min-words-to-chunk", type=int, default=600)
    parser.add_argument("--whisper-model", type=str, default="base")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Dossier manquant: {input_dir}")
        return

    videos = iter_videos(input_dir, recursive=args.recursive)
    if not videos:
        print("ℹ️  Pas de vidéos")
        return

    # Charger Whisper
    print(f"\n🤖 Whisper '{args.whisper_model}'...")
    global whisper_model
    whisper_model = whisper.load_model(args.whisper_model)
    
    agent = AgentAnalyst()

    print(f"\n📂 Entrée: {input_dir}")
    print(f"📁 Sortie: {output_dir}")
    print(f"🎥 Vidéos: {len(videos)}")
    print("-" * 50)

    success = 0
    errors = 0

    for video in videos:
        out = output_dir / f"{video.stem}.json"
        try:
            analyze_one(
                agent,
                video,
                out,
                overwrite=args.overwrite,
                chunk_words=args.chunk_words,
                overlap_words=args.overlap_words,
                min_words_to_chunk=args.min_words_to_chunk,
            )
            success += 1
        except KeyboardInterrupt:
            print("\n⏹️  Arrêt")
            break
        except Exception:
            errors += 1

    print("\n" + "=" * 50)
    print(f"📊 Résultat: {success}✅ {errors}❌ sur {len(videos)}🎥")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Au revoir")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Erreur: {e}")
        sys.exit(1)