import os
import random
import time
import wave
import sys
import numpy as np

# Adicionar o diretório raiz ao path para facilitar imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mispronunciation_detector import MispronunciationDetector

# Lista de perguntas simples para prática de inglês
QUESTIONS = [
    "How are you today?",
    "What is your favorite color?",
    "Where are you from?",
    "What do you like to do in your free time?",
    "What is the weather like today?",
    "Can you describe your best friend?",
    "What did you have for breakfast?",
    "What is your favorite movie?",
    "Do you like to travel?",
    "What is your dream job?"
]

def record_audio(output_path, duration=5, sample_rate=16000):
    """Grava áudio do microfone e salva em um arquivo WAV."""
    try:
        import sounddevice as sd
        from scipy.io import wavfile
    except ImportError:
        print("\n[ERRO] Bibliotecas necessárias para gravação não encontradas.")
        print("Por favor, instale: pip install sounddevice scipy")
        print("No macOS, você também pode precisar de: brew install portaudio")
        return False

    print(f"\nPreparando para gravar por {duration} segundos...")
    print("Fale agora!")
    
    # Gravação
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    
    # Barra de progresso simples
    for i in range(duration):
        time.sleep(1)
        print(f"{duration - i}...", end=" ", flush=True)
    
    sd.wait()
    print("\nGravação concluída.")
    
    # Salvar arquivo
    wavfile.write(output_path, sample_rate, recording)
    return True

def main():
    print("="*50)
    print("   TF-PLN: Verificador de Pronúncia Interativo")
    print("="*50)
    
    # Inicializar detector (pode demorar um pouco devido aos modelos)
    try:
        detector = MispronunciationDetector()
    except Exception as e:
        print(f"Erro ao carregar os modelos: {e}")
        return

    while True:
        question = random.choice(QUESTIONS)
        print(f"\nPERGUNTA: {question}")
        input("Pressione ENTER para começar a gravar sua resposta...")
        
        temp_wav = "temp_recording.wav"
        if record_audio(temp_wav):
            print("\nIniciando análise...")
            try:
                # O método detect() agora lida com toda a impressão dos resultados
                detector.detect(temp_wav)
            except Exception as e:
                print(f"Erro durante a análise: {e}")
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
        
        cont = input("\nDeseja tentar outra pergunta? (s/n): ").lower()
        if cont != 's':
            break

    print("\nObrigado por praticar!")

if __name__ == "__main__":
    main()
