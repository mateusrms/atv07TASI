# main.py

import whisper
import subprocess
import os
import requests


def converter_audio(input_file, output_file="audio_convertido.wav"):
    """
    Converte áudio para o formato WAV usando ffmpeg.
    """
    print("Convertendo o áudio para WAV...")
    comando = ["ffmpeg", "-y", "-i", input_file, output_file]
    subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file

def transcrever_audio(audio_file):
    """
    Transcreve o áudio usando o modelo Whisper.
    """
    print("Carregando modelo Whisper...")
    model = whisper.load_model("base")
    print("Transcrevendo o áudio...")
    result = model.transcribe(audio_file)
    return result['text']

def extrair_pontos_chave(texto_transcrito):
    """
    Envia o texto transcrito para o Ollama para gerar pontos-chave.
    """
    print("Enviando texto para o Ollama...")
    # Ajuste conforme seu modelo Ollama disponível localmente
    ollama_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3",  # ou outro modelo instalado
        "prompt": f"Extraia os principais pontos-chave do seguinte texto:\n\n{texto_transcrito}"
    }

    response = requests.post(ollama_url, json=payload)
    if response.status_code == 200:
        resumo = response.json()['response']
        return resumo
    else:
        print("Erro ao se comunicar com o Ollama.")
        return None


def pipeline(audio_path):
    """
    Pipeline completo: Conversão -> Transcrição -> Extração de Pontos-Chave
    """
    audio_convertido = converter_audio(audio_path)
    texto = transcrever_audio(audio_convertido)
    resumo = extrair_pontos_chave(texto)

    print("\n\n==== Transcrição ====\n")
    print(texto)
    print("\n\n==== Pontos-Chave ====\n")
    print(resumo)


if __name__ == "__main__":
    caminho_audio = input("Informe o caminho do arquivo de áudio (.wav ou .mp3): ")
    pipeline(caminho_audio)
