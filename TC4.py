# Instalação de bibliotecas

!pip install deepface
!pip install keras
!pip install opencv-python-headless tf-keras

# Variáveis de entrada e saída
input_video_path="/content/drive/MyDrive/Colab Notebooks/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
output_video_path_emotions="/content/output_video_emotions.mp4"
output_video_path_pose="/content/output_video_pose.mp4"
output_text_path="/content/output_text.txt"
output_audio_path="/content/output_audio.wav"
output_text_path_sentences="/content/output_text_sentences.txt"
output_text_path_punctuation="/content/output_text_punctuation.txt"
output_text_path_summarization="/content/output_text_summarization.txt"
output_text_path_video_emotions="/content/output_text_path_video_emotions.txt"
output_text_path_video_emotions_summarization="/content/output_text_path_video_emotions_summarization.txt"

# Detecção de emoção

import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm

def detect_emotions(video_path, output_path):
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print("Error opening video file")
    return

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  dominant_emotions = []
  for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
      break

    results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            align=False
        )

        # Caso retorne um único dicionário, converte para lista
    if isinstance(results, dict):
        results = [results]


    for face in results:
      x = face['region']['x']
      y = face['region']['y']
      w = face['region']['w']
      h = face['region']['h']

      dominant_emotion = face['dominant_emotion']
      dominant_emotions.append({
          "frame": _,
          "emotion": dominant_emotion
      })


      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

  cap.release()
  out.release()
  # cv2.destroyAllWindows()

    # with open(output_text_path_video_emotions, "w", encoding="utf-8") as f:
    # for item in dominant_emotions:
    #     f.write(f"At frame {item['frame']}, a person expressed {item['emotion']}.")

  # Após processar todos os frames:
  emotion_counts = {}
  emotion_sequences = []
  current_emotion = None
  start_frame = 0
  
  for item in dominant_emotions:
      frame = item['frame']
      emotion = item['emotion']
      
      # Contabilizar emoções
      if emotion not in emotion_counts:
          emotion_counts[emotion] = 0
      emotion_counts[emotion] += 1
      
      # Detectar mudanças de emoção para criar sequências
      if emotion != current_emotion:
          if current_emotion is not None:
              emotion_sequences.append({
                  'emotion': current_emotion,
                  'start_frame': start_frame,
                  'end_frame': frame - 1,
                  'duration': frame - start_frame
              })
          current_emotion = emotion
          start_frame = frame
  
  # Adicionar a última sequência
  if current_emotion is not None:
      emotion_sequences.append({
          'emotion': current_emotion,
          'start_frame': start_frame,
          'end_frame': len(dominant_emotions) - 1,
          'duration': len(dominant_emotions) - start_frame
      })
  
  # Escrever resumo estruturado
  with open(output_text_path_video_emotions, "w", encoding="utf-8") as f:
      # Resumo geral
      f.write("EMOTION ANALYSIS SUMMARY\n\n")
      f.write("Overall emotion distribution:\n")
      for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
          percentage = (count / len(dominant_emotions)) * 100
          f.write(f"- {emotion}: {count} frames ({percentage:.1f}%)\n")
      
      f.write("\nEmotion sequences:\n")
      for i, seq in enumerate(emotion_sequences):
          if seq['duration'] > 10:  # Filtrar sequências muito curtas
              f.write(f"Sequence {i+1}: {seq['emotion']} from frame {seq['start_frame']} to {seq['end_frame']} (duration: {seq['duration']} frames)\n")
      
      f.write("\nDetailed frame analysis:\n")
      # Agrupar por grupos de 30 frames para reduzir verbosidade
      for i in range(0, len(dominant_emotions), 30):
          group = dominant_emotions[i:i+30]
          main_emotion = max(set([g['emotion'] for g in group]), key=[g['emotion'] for g in group].count)
          f.write(f"From frame {group[0]['frame']} to frame {group[-1]['frame']}: predominantly {main_emotion}\n")


detect_emotions(input_video_path, output_video_path_emotions)

# Detecção de pose

!pip install mediapipe

import mediapipe as mp
import cv2
from tqdm import tqdm
from google.colab.patches import cv2_imshow

def detect_pose(video_path, output_path):
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file")
    return

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
      break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
      mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)


  cap.release()
  out.release()
  # cv2.destroyAllWindows()

detect_pose(input_video_path, output_video_path_pose)

#  Transcrição de áudio

!pip install moviepy speechrecognition pydub
!pip install deepmultilingualpunctuation

import  moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import os
from tqdm import tqdm
from deepmultilingualpunctuation import PunctuationModel

def extract_audio_form_video(video_path, output_audio_path):
  clip = mp.VideoFileClip(video_path)
  clip.audio.write_audiofile(output_audio_path)

def transcribe_audio(audio_path, output_text_path):
  recognizer = sr.Recognizer()
  with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data, language="en-US")
    with open(output_text_path, "w", encoding="utf-8") as text_file:
      text_file.write(text)

def insert_text_punctuation(input_text_path, output_text_path_punctuation):

  with open(input_text_path, "r", encoding="utf-8") as f:
    text = f.read()

  model = PunctuationModel()
  text_pontuado = model.restore_punctuation(text)

  with open(output_text_path_punctuation, "w", encoding="utf-8") as f:
    f.write(text_pontuado)

extract_audio_form_video(input_video_path, output_audio_path)
transcribe_audio(output_audio_path, output_text_path)
insert_text_punctuation(output_text_path, output_text_path_punctuation)

# Categorização
!pip install scikit-learn gensim nltk scipy

import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

def classify_sentences (input_text, output_text_sentences):
  text = []
  labels = [
      "technology",
      "technology",
      "technology",
      "reading",
      "conversation",
      "movement",
      "emotion",
      "emotion",
      "details",
      "diversity",
      "technology",
      "technology"
  ]

  with open(input_text, "r", encoding="utf-8") as f:
    raw_text = f.read()

  # Quebra em frases
  text = [s.strip() for s in raw_text.split('.') if s.strip()]

  print(len(text))
  x_train, x_test, y_train, y_test = train_test_split(text[:9], labels[:9], test_size=0.2, random_state=42)

  model = make_pipeline(TfidfVectorizer(), MultinomialNB())
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)
  with open(output_text_sentences, "w", encoding="utf-8") as f:
    f.write("Metrics:\n")
    f.write(metrics.classification_report(y_test, predicted, zero_division=0))

    # Acurácia
    accuracy = metrics.accuracy_score(y_test, predicted)
    f.write(f"Acurácia: {accuracy:.2f}")

    test_sentences = [
        "a man focuses intently on a tablet, his brow furrowed in concentration",
        "two friends laugh loudly while watching a funny video together",
        "a woman flips through a book, absorbed by its content",
        "a child waves enthusiastically at the camera, smiling widely",
        "people move through the space naturally, creating a constant shift in the environment",
        "a man and woman have a heated discussion, their gestures growing more animated",
        "a face scanner tracks multiple individuals entering a busy lobby",
        "subtle facial expressions reveal a range of unspoken thoughts",
        "visual markers trace the contours of each unique face in the crowd",
        "the diversity of ages and appearances highlights human uniqueness"
    ]

    f.write("\n\nNew predictions:\n")
    predictions = model.predict(test_sentences)
    for sentence, label in zip(test_sentences, predictions):
        f.write(f"{sentence} ---> {label}\n")

classify_sentences(output_text_path_punctuation, output_text_path_sentences)

# Sumarização

!pip install transformers

import re
from collections import defaultdict
from transformers import pipeline

# Inicialize o modelo de sumarização
summarizer = pipeline("summarization", device=-1)

def extract_emotion_patterns(text):
    """
    Extrai padrões e tendências das sequências de emoção.
    """
    # Extrair as distribuições gerais
    distribution_match = re.search(r"Overall emotion distribution:(.*?)Emotion sequences:", text, re.DOTALL)
    distributions = {}
    if distribution_match:
        dist_text = distribution_match.group(1).strip()
        for line in dist_text.split('\n'):
            if line.strip():
                match = re.search(r'- (\w+): (\d+) frames \((\d+\.\d+)%\)', line)
                if match:
                    emotion, frames, percentage = match.groups()
                    distributions[emotion] = (int(frames), float(percentage))
    
    # Extrair sequências significativas
    sequences = []
    seq_pattern = re.compile(r'Sequence \d+: (\w+) from frame (\d+) to (\d+) \(duration: (\d+) frames\)')
    seq_matches = seq_pattern.finditer(text)
    
    for match in seq_matches:
        emotion, start, end, duration = match.groups()
        sequences.append({
            'emotion': emotion,
            'start': int(start),
            'end': int(end),
            'duration': int(duration)
        })
    
    # Ordenar sequências por duração (descendente)
    sequences.sort(key=lambda x: x['duration'], reverse=True)
    
    # Extrair análise detalhada de frames
    frame_analysis = []
    analysis_pattern = re.compile(r'From frame (\d+) to frame (\d+): predominantly (\w+)')
    analysis_matches = analysis_pattern.finditer(text)
    
    for match in analysis_matches:
        start, end, emotion = match.groups()
        frame_analysis.append({
            'start': int(start),
            'end': int(end),
            'emotion': emotion
        })
    
    return {
        'distributions': distributions,
        'sequences': sequences,
        'frame_analysis': frame_analysis
    }

def create_narrative_chunks(data):
    """
    Cria chunks de texto narrativo a partir dos dados extraídos.
    """
    chunks = []
    
    # Chunk 1: Visão geral das emoções
    overview = "# Análise Emocional do Vídeo\n\n"
    overview += "## Distribuição Geral de Emoções\n\n"
    
    # Ordenar emoções por percentual
    sorted_emotions = sorted(data['distributions'].items(), 
                            key=lambda x: x[1][1], 
                            reverse=True)
    
    for emotion, (frames, percentage) in sorted_emotions:
        overview += f"- **{emotion.capitalize()}**: {percentage}% do vídeo ({frames} frames)\n"
    
    chunks.append(overview)
    
    # Chunk 2: Principais sequências emocionais
    top_sequences = "## Principais Sequências Emocionais\n\n"
    top_sequences += "As sequências emocionais mais longas identificadas no vídeo são:\n\n"
    
    # Pegar as 10 sequências mais longas
    for i, seq in enumerate(data['sequences'][:10]):
        emotion = seq['emotion'].capitalize()
        duration_sec = seq['duration'] / 30.0  # Assumindo 30 fps, converter para segundos
        top_sequences += f"- **Sequência {i+1}**: {emotion} por {duration_sec:.1f} segundos (frames {seq['start']}-{seq['end']})\n"
    
    chunks.append(top_sequences)
    
    # Chunk 3: Padrões e transições
    transitions = "## Padrões e Transições Emocionais\n\n"
    
    # Analisar transições entre emoções
    emotion_transitions = defaultdict(int)
    prev_emotion = None
    
    for analysis in data['frame_analysis']:
        if prev_emotion and prev_emotion != analysis['emotion']:
            transition = f"{prev_emotion} → {analysis['emotion']}"
            emotion_transitions[transition] += 1
        prev_emotion = analysis['emotion']
    
    # Pegar as transições mais comuns
    common_transitions = sorted(emotion_transitions.items(), key=lambda x: x[1], reverse=True)[:5]
    
    transitions += "As transições emocionais mais frequentes observadas foram:\n\n"
    for transition, count in common_transitions:
        transitions += f"- **{transition.capitalize()}**: {count} vezes\n"
    
    # Identificar segmentos emocionais do vídeo
    video_segments = []
    current_segment = {'emotion': data['frame_analysis'][0]['emotion'], 'start': data['frame_analysis'][0]['start']}
    
    for i in range(1, len(data['frame_analysis'])):
        if data['frame_analysis'][i]['emotion'] != current_segment['emotion']:
            current_segment['end'] = data['frame_analysis'][i-1]['end']
            video_segments.append(current_segment)
            current_segment = {'emotion': data['frame_analysis'][i]['emotion'], 'start': data['frame_analysis'][i]['start']}
    
    # Adicionar o último segmento
    if 'end' not in current_segment:
        current_segment['end'] = data['frame_analysis'][-1]['end']
        video_segments.append(current_segment)
    
    # Encontrar os 3 segmentos mais longos
    video_segments.sort(key=lambda x: x['end'] - x['start'], reverse=True)
    
    transitions += "\nOs segmentos emocionais mais longos do vídeo foram:\n\n"
    for i, segment in enumerate(video_segments[:3]):
        emotion = segment['emotion'].capitalize()
        frame_count = segment['end'] - segment['start']
        start_time = segment['start'] / 30.0  # Convertendo para segundos
        end_time = segment['end'] / 30.0
        transitions += f"- **{emotion}**: {frame_count} frames ({start_time:.1f}s - {end_time:.1f}s do vídeo)\n"
    
    chunks.append(transitions)
    
    # Chunk 4: Resumo narrativo
    narrative = "## Narrativa Emocional\n\n"
    
    # Dividir o vídeo em terços para análise narrativa
    total_frames = data['frame_analysis'][-1]['end']
    first_third = total_frames // 3
    second_third = 2 * (total_frames // 3)
    
    # Contar emoções em cada terço
    emotions_by_third = [defaultdict(int), defaultdict(int), defaultdict(int)]
    
    for analysis in data['frame_analysis']:
        start = analysis['start']
        if start < first_third:
            section = 0
        elif start < second_third:
            section = 1
        else:
            section = 2
            
        emotions_by_third[section][analysis['emotion']] += 1
    
    # Determinar emoção dominante de cada terço
    dominant_emotions = []
    for third in emotions_by_third:
        if third:
            dominant = max(third.items(), key=lambda x: x[1])[0]
            dominant_emotions.append(dominant)
        else:
            dominant_emotions.append("não identificada")
    
    narrative += f"No início do vídeo, a emoção predominante foi **{dominant_emotions[0]}**. "
    narrative += f"Na parte intermediária, observou-se principalmente **{dominant_emotions[1]}**. "
    narrative += f"Na parte final, a emoção dominante passou a ser **{dominant_emotions[2]}**.\n\n"
    
    # Adicionar insights sobre picos emocionais
    emotional_peaks = []
    for emotion in ['happy', 'sad', 'fear', 'surprise', 'angry']:
        peaks = [seq for seq in data['sequences'] if seq['emotion'] == emotion and seq['duration'] > 20]
        if peaks:
            longest_peak = max(peaks, key=lambda x: x['duration'])
            emotional_peaks.append({
                'emotion': emotion,
                'duration': longest_peak['duration'],
                'start': longest_peak['start'],
                'end': longest_peak['end']
            })
    
    if emotional_peaks:
        narrative += "Momentos emocionais significativos incluíram:\n\n"
        for peak in emotional_peaks:
            emotion = peak['emotion'].capitalize()
            start_time = peak['start'] / 30.0
            end_time = peak['end'] / 30.0
            narrative += f"- Um pico de **{emotion}** entre {start_time:.1f}s e {end_time:.1f}s\n"
    
    chunks.append(narrative)
    
    return chunks

def summarize_emotion_analysis(input_file_path, output_file_path, max_length=150):
    """
    Função principal para resumir a análise de emoções.
    """
    # Ler o arquivo de entrada
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Extrair dados estruturados do texto
    data = extract_emotion_patterns(text)
    
    # Criar chunks narrativos
    narrative_chunks = create_narrative_chunks(data)
    
    # Processar cada chunk narrativo
    processed_chunks = []
    
    for chunk in narrative_chunks:
        # Verificar se o chunk precisa ser resumido
        if len(chunk.split()) > 100:
            try:
                # Aplicar o modelo de summarization apenas para chunks longos
                result = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)
                if result and len(result) > 0:
                    # Preservar títulos de seção e formatar o resumo
                    title_match = re.search(r'^(#+\s.*?)$', chunk, re.MULTILINE)
                    title = title_match.group(1) if title_match else ""
                    
                    summary_text = result[0]["summary_text"]
                    # Melhorar a formatação do resumo
                    summary_text = summary_text.replace(" . ", ". ")
                    summary_text = summary_text.replace(" , ", ", ")
                    
                    processed_text = f"{title}\n\n{summary_text}" if title else summary_text
                    processed_chunks.append(processed_text)
                else:
                    processed_chunks.append(chunk)
            except Exception as e:
                print(f"Erro ao resumir chunk: {str(e)[:100]}...")
                processed_chunks.append(chunk)
        else:
            # Manter chunks curtos intactos
            processed_chunks.append(chunk)
    
    # Adicionar conclusão
    conclusion = "\n\n## Conclusão\n\n"
    conclusion += "Esta análise emocional revela padrões significativos nas expressões faciais capturadas no vídeo. "
    conclusion += "As transições entre diferentes estados emocionais fornecem insights sobre a dinâmica do conteúdo apresentado."
    
    processed_chunks.append(conclusion)
    
    # Combinar tudo em um texto final
    final_text = "\n\n".join(processed_chunks)
    
    # Salvar o resultado
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    return final_text

# Função para testar sem depender do pipeline transformer
def summarize_without_model(input_file_path, output_file_path):
    """
    Versão simplificada que não usa o modelo de transformers.
    Útil quando há problemas com o modelo ou para testes rápidos.
    """
    # Ler o arquivo de entrada
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Extrair dados estruturados do texto
    data = extract_emotion_patterns(text)
    
    # Criar chunks narrativos (sem resumir)
    narrative_chunks = create_narrative_chunks(data)
    
    # Combinar tudo em um texto final
    final_text = "\n\n".join(narrative_chunks)
    
    # Adicionar conclusão
    conclusion = "\n\n## Conclusão\n\n"
    conclusion += "Esta análise emocional revela padrões significativos nas expressões faciais capturadas no vídeo. "
    conclusion += "As transições entre diferentes estados emocionais fornecem insights sobre a dinâmica do conteúdo apresentado."
    
    final_text += conclusion
    
    # Salvar o resultado
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    return final_text

# Usando o modelo transformer (pode falhar com o erro de índice)
try:
    summarize_emotion_analysis("output_text_path_video_emotions.txt", 
                              "output_text_path_video_emotions_summarization.txt")
except Exception as e:
    print(f"Erro ao usar o modelo transformer: {str(e)}")
    print("Usando método alternativo...")
    
    # Versão alternativa sem depender do modelo
    summarize_without_model("output_text_path_video_emotions.txt", 
                           "output_text_path_video_emotions_summarization.txt")