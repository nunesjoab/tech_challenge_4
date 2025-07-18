# Instalação de bibliotecas

!pip install deepface
!pip install keras
!pip install opencv-python-headless tf-keras

# Variáveis de entrada e saída
input_video_path="/content/drive/MyDrive/Colab Notebooks/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4"
output_video_path_emotions="/content/output_video_emotions.mp4"
output_video_path_pose="/content/output_video_pose.mp4"
output_video_path_anomalies = "/content/output_video_anomalies.mp4"
output_text_path_anomalies = "/content/output_text_anomalies.txt"
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

#Detecção de Anomalias
def detect_movement_anomalies(video_path, output_path, output_text_path_anomalies, sensitivity=0.15, window_size=10):
    """
    Detecta anomalias de movimento no vídeo analisando variações bruscas nas poses.
    
    Args:
        video_path: Caminho para o vídeo de entrada
        output_path: Caminho para o vídeo de saída com marcações de anomalias
        output_text_path_anomalies: Caminho para o arquivo de texto com descrição das anomalias
        sensitivity: Limiar de sensibilidade para detecção (0.05-0.2 recomendado)
        window_size: Tamanho da janela para suavização de movimentos
    """
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
    
    # Armazenar histórico de landmarks para detectar movimentos bruscos
    landmark_history = []
    anomalies = []
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting anomalies"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Visualizar pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extrair coordenadas dos landmarks principais
            current_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                current_landmarks.append((landmark.x, landmark.y, landmark.z))
                
            # Calcular velocidade de movimento se tivermos histórico suficiente
            if len(landmark_history) > 0:
                # Calcular variação média entre frames consecutivos
                avg_movement = 0
                prev_landmarks = landmark_history[-1]
                
                for i, (curr_x, curr_y, curr_z) in enumerate(current_landmarks):
                    if i < len(prev_landmarks):
                        prev_x, prev_y, prev_z = prev_landmarks[i]
                        # Calcular distância euclidiana
                        distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)**0.5
                        avg_movement += distance
                
                avg_movement /= len(current_landmarks)
                
                # Verificar se o movimento é anômalo
                is_anomaly = False
                if len(landmark_history) >= window_size:
                    # Calcular média de movimento na janela recente
                    window_movements = []
                    for j in range(1, min(window_size, len(landmark_history))):
                        window_prev = landmark_history[-j-1]
                        window_curr = landmark_history[-j]
                        window_movement = 0
                        for i, (curr_x, curr_y, curr_z) in enumerate(window_curr):
                            if i < len(window_prev):
                                prev_x, prev_y, prev_z = window_prev[i]
                                window_movement += ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)**0.5
                        window_movement /= len(window_curr)
                        window_movements.append(window_movement)
                    
                    avg_window_movement = sum(window_movements) / len(window_movements)
                    
                    # Detectar se o movimento atual é significativamente maior que a média recente
                    if avg_movement > (avg_window_movement * (1 + sensitivity)) and avg_movement > 0.01:  # Limiar mínimo para ignorar pequenas variações
                        is_anomaly = True
                        anomalies.append({
                            "frame": frame_idx,
                            "timestamp": frame_idx / fps,
                            "movement_intensity": avg_movement,
                            "baseline_movement": avg_window_movement,
                            "ratio": avg_movement / avg_window_movement
                        })
                        
                        # Marcar anomalia no frame
                        cv2.putText(frame, "ANOMALY DETECTED", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Adicionar borda vermelha
                        cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 3)
            
            # Adicionar landmarks atuais ao histórico
            landmark_history.append(current_landmarks)
            # Limitar o tamanho do histórico
            if len(landmark_history) > window_size:
                landmark_history.pop(0)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Agrupar anomalias próximas (dentro de 1 segundo)
    grouped_anomalies = []
    if anomalies:
        current_group = [anomalies[0]]
        for i in range(1, len(anomalies)):
            if anomalies[i]["frame"] - current_group[-1]["frame"] <= fps:  # 1 segundo
                current_group.append(anomalies[i])
            else:
                # Finalizar grupo atual
                start_time = current_group[0]["timestamp"]
                end_time = current_group[-1]["timestamp"]
                max_intensity = max([a["ratio"] for a in current_group])
                grouped_anomalies.append({
                    "start_frame": current_group[0]["frame"],
                    "end_frame": current_group[-1]["frame"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "max_intensity": max_intensity,
                    "anomaly_count": len(current_group)
                })
                # Iniciar novo grupo
                current_group = [anomalies[i]]
        
        # Adicionar último grupo
        if current_group:
            start_time = current_group[0]["timestamp"]
            end_time = current_group[-1]["timestamp"]
            max_intensity = max([a["ratio"] for a in current_group])
            grouped_anomalies.append({
                "start_frame": current_group[0]["frame"],
                "end_frame": current_group[-1]["frame"],
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "max_intensity": max_intensity,
                "anomaly_count": len(current_group)
            })
    
    # Escrever análise de anomalias
    with open(output_text_path_anomalies, "w", encoding="utf-8") as f:
        f.write("MOVEMENT ANOMALY ANALYSIS\n\n")
        
        if not grouped_anomalies:
            f.write("No significant movement anomalies detected in the video.\n")
        else:
            f.write(f"Detected {len(grouped_anomalies)} anomaly events:\n\n")
            
            # Ordenar por intensidade
            grouped_anomalies.sort(key=lambda x: x["max_intensity"], reverse=True)
            
            for i, anomaly in enumerate(grouped_anomalies):
                severity = "High" if anomaly["max_intensity"] > 2.0 else ("Medium" if anomaly["max_intensity"] > 1.5 else "Low")
                f.write(f"Anomaly {i+1} (Severity: {severity}):\n")
                f.write(f"  Time: {anomaly['start_time']:.2f}s to {anomaly['end_time']:.2f}s (duration: {anomaly['duration']:.2f}s)\n")
                f.write(f"  Frames: {anomaly['start_frame']} to {anomaly['end_frame']}\n")
                f.write(f"  Movement intensity: {anomaly['max_intensity']:.2f}x normal\n")
                f.write(f"  Consecutive anomalous frames: {anomaly['anomaly_count']}\n\n")
            
            # Adicionar estatísticas gerais
            total_anomaly_time = sum([a["duration"] for a in grouped_anomalies])
            total_video_time = total_frames / fps
            anomaly_percentage = (total_anomaly_time / total_video_time) * 100
            
            f.write("\nSummary Statistics:\n")
            f.write(f"Total video duration: {total_video_time:.2f} seconds\n")
            f.write(f"Total time with anomalies: {total_anomaly_time:.2f} seconds ({anomaly_percentage:.1f}% of video)\n")
            
            # Classificar o vídeo baseado na presença de anomalias
            if anomaly_percentage > 15:
                f.write("\nAssessment: High level of anomalous movement detected throughout the video.\n")
            elif anomaly_percentage > 5:
                f.write("\nAssessment: Moderate level of anomalous movement detected in the video.\n")
            else:
                f.write("\nAssessment: Low level of anomalous movement detected, mostly normal activity patterns.\n")
    
    return anomalies

# Detecção de Pose
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

# Detecção de anomalias
detect_movement_anomalies(input_video_path, output_video_path_anomalies, output_text_path_anomalies)

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



def summarize_emotion_analysis(input_file_path, output_file_path, anomalies_file_path=None, max_length=150):
    """
    Função principal para resumir a análise de emoções, agora com suporte para anomalias.
    
    Args:
        input_file_path: Caminho para o arquivo de análise de emoções
        output_file_path: Caminho para salvar o resumo gerado
        anomalies_file_path: (Opcional) Caminho para o arquivo de análise de anomalias
        max_length: Comprimento máximo para o resumo
    """
    # Código original
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    data = extract_emotion_patterns(text)
    narrative_chunks = create_narrative_chunks(data)
    
    # Adicionar informações de anomalias se disponível
    if anomalies_file_path and os.path.exists(anomalies_file_path):
        try:
            with open(anomalies_file_path, "r", encoding="utf-8") as f:
                anomalies_text = f.read()
                
            # Extrair informações de anomalias
            anomaly_count_match = re.search(r"Detected (\d+) anomaly events:", anomalies_text)
            anomaly_count = int(anomaly_count_match.group(1)) if anomaly_count_match else 0
            
            anomaly_percentage_match = re.search(r"Total time with anomalies: (\d+\.\d+) seconds \((\d+\.\d+)% of video\)", anomalies_text)
            anomaly_percentage = float(anomaly_percentage_match.group(2)) if anomaly_percentage_match else 0
            
            assessment_match = re.search(r"Assessment: (.*?)\n", anomalies_text)
            assessment = assessment_match.group(1) if assessment_match else "No assessment available"
            
            # Extrair as anomalias mais significativas
            top_anomalies = []
            anomaly_pattern = re.compile(r"Anomaly (\d+) \(Severity: (.*?)\):\n  Time: (\d+\.\d+)s to (\d+\.\d+)s.*?\n  Frames: (\d+) to (\d+)\n  Movement intensity: (\d+\.\d+)x", re.DOTALL)
            for match in anomaly_pattern.finditer(anomalies_text):
                _, severity, start_time, end_time, start_frame, end_frame, intensity = match.groups()
                top_anomalies.append({
                    "severity": severity,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "intensity": float(intensity)
                })
            
            # Limitar aos 5 principais
            top_anomalies = sorted(top_anomalies, key=lambda x: x["intensity"], reverse=True)[:5]
            
            # Criar seção de anomalias
            anomaly_section = "## Anomalias de Movimento Detectadas\n\n"
            if anomaly_count > 0:
                anomaly_section += f"Foram detectados {anomaly_count} eventos anômalos de movimento, ocupando {anomaly_percentage:.1f}% do tempo total do vídeo.\n\n"
                anomaly_section += f"{assessment}\n\n"
                
                if top_anomalies:
                    anomaly_section += "Principais anomalias detectadas:\n\n"
                    for i, anomaly in enumerate(top_anomalies):
                        anomaly_section += f"- **Anomalia {i+1} ({anomaly['severity']})**: Em {anomaly['start_time']:.1f}s-{anomaly['end_time']:.1f}s, "
                        anomaly_section += f"intensidade {anomaly['intensity']:.1f}x acima do normal\n"
            else:
                anomaly_section += "Não foram detectadas anomalias significativas de movimento no vídeo. "
                anomaly_section += "Os padrões de movimento observados estão dentro dos limites normais de variação."
            
            # Inserir a seção de anomalias após o terceiro chunk (após padrões e transições)
            if len(narrative_chunks) >= 3:
                narrative_chunks.insert(3, anomaly_section)
            else:
                narrative_chunks.append(anomaly_section)
                
        except Exception as e:
            print(f"Erro ao processar arquivo de anomalias: {str(e)}")
            # Continuar sem as informações de anomalias
    
    # Resto do código original
    processed_chunks = []
    for chunk in narrative_chunks:
        # ... (código existente para processamento de chunks)
        if len(chunk.split()) > 100:
            try:
                # Aplicar o modelo de summarization apenas para chunks longos
                result = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)
                if result and len(result) > 0:
                    title_match = re.search(r'^(#+\s.*?)$', chunk, re.MULTILINE)
                    title = title_match.group(1) if title_match else ""
                    
                    summary_text = result[0]["summary_text"]
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
            processed_chunks.append(chunk)
    
    # Adicionar conclusão atualizada com menção a anomalias
    conclusion = "\n\n## Conclusão\n\n"
    if anomalies_file_path and os.path.exists(anomalies_file_path):
        conclusion += "Esta análise revela padrões significativos nas expressões faciais e movimentos capturados no vídeo. "
        conclusion += "As transições entre diferentes estados emocionais, juntamente com a detecção de anomalias de movimento, "
        conclusion += "fornecem insights valiosos sobre a dinâmica comportamental apresentada."
    else:
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
    summarize_emotion_analysis(
        "output_text_path_video_emotions.txt",
        "output_text_path_video_emotions_summarization.txt",
        anomalies_file_path="output_text_anomalies.txt"
    )
except Exception as e:
    print(f"Erro ao usar o modelo transformer: {str(e)}")
    print("Usando método alternativo...")
    
    # Versão alternativa sem depender do modelo
    summarize_without_model(
        "output_text_path_video_emotions.txt",
        "output_text_path_video_emotions_summarization.txt"
    )