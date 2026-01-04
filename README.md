
# 🧠 Tech Challenge – Análise Inteligente de Vídeo com IA (Visão Computacional)

Este projeto foi desenvolvido como parte do **Tech Challenge da FIAP (Fase 4 – Inteligência Artificial)** e tem como objetivo a criação de uma **aplicação de análise automática de vídeo**, utilizando técnicas avançadas de **Visão Computacional, Deep Learning e IA Generativa**.

A aplicação é capaz de:
- Identificar pessoas em vídeo
- Analisar expressões emocionais faciais
- Detectar e categorizar atividades humanas
- Detectar comportamentos anômalos
- Gerar automaticamente um resumo estruturado do conteúdo analisado

---

## 🎯 Objetivo do Projeto

Aplicar na prática os conhecimentos adquiridos ao longo da fase, integrando múltiplos modelos de IA para realizar uma **análise semântica e comportamental de vídeos**, simulando cenários reais como reuniões de trabalho, uso de computadores, interações sociais, atividades expressivas (dança, gestos) e situações fora do padrão.

---

## 📁 Estrutura do Projeto

```
Tech-Challenge-4-IA-FIAP/
├── assets/
│   └── input_video.mp4
├── outputs/
│   ├── annotated_video.mp4
│   ├── report.txt
│   └── events.json
├── src/
│   ├── main.py
│   ├── pipeline/
│   │   ├── person_detector.py
│   │   ├── clip_zeroshot.py
│   │   ├── action_recog.py
│   │   ├── emotion_deepface.py
│   │   ├── anomaly.py
│   │   └── summarizer.py
│   └── utils/
│       └── video_utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Como Executar

### 0) Pré-requisitos
- **Python 3.11** (recomendado)  
- Windows PowerShell (ou terminal VS Code)  
- Vídeo em `assets/input_video.mp4`

### 1) Criar ambiente virtual
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Instalar dependências do projeto
```powershell
pip install -r requirements.txt
```

---

## 🚀 Rodar em GPU NVIDIA CUDA (Recomendado)

GPU antiga:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU nova por exemplo RTX 5070 arquitetura Blackwell e tem CUDA capability sm_120
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 🔎 Verificação - validar GPU (PyTorch)
```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('cap:', torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_name(0))"
```

```powershell
python -m src.main `
  --video "assets/input_video.mp4" `
  --out "outputs" `
  --device cuda `
  --frame-skip 2 `
  --clip-len 16
```


> 💡 **Observação importante (RTX 5070 / sm_120):**  
> Algumas GPUs muito novas podem exigir uma versão do PyTorch com suporte atualizado. Se aparecer erro de compatibilidade, instale uma build mais recente (nightly) conforme instrução acima, ou use CPU temporariamente.

---

## 🖥️ Rodar em CPU (Fallback – mais lento)

Se você não tiver GPU NVIDIA (ou não estiver configurada), rode em CPU:

```bash
pip install torch torchvision torchaudio
```

```powershell
python -m src.main `
  --video "assets/input_video.mp4" `
  --out "outputs" `
  --device cpu `
  --frame-skip 2 `
  --clip-len 16
```

⚠️ **Aviso:** em CPU o processamento demora mais (pode levar de **20 a 40+ minutos**, dependendo do vídeo e das configurações).

---

## 🔧 Parâmetros úteis (para qualidade x performance)

- `--frame-skip 2`  
  Analisa 1 frame a cada 2 (menos custo, mais rápido).  
  Para mais qualidade, use `--frame-skip 1`.

- `--clip-len 16`  
  Número de frames por “clip” para ações.  
  Aumentar ajuda ações contínuas, mas custa mais.

---

## 📊 Saídas Geradas

Após rodar, você terá:

- `outputs/annotated_video.mp4`  
  Vídeo com caixas (pessoa/face), IDs, atividade e emoção.

- `outputs/report.txt`  
  Relatório automático com:
  - total de frames analisados
  - número de anomalias detectadas
  - ranking de atividades
  - emoções por pessoa
  - atividades por pessoa
  - amostras de anomalias

- `outputs/events.json`  
  Log detalhado por frame (útil para auditoria/debug).

---

## 🧠 Técnicas Utilizadas

### 1) Detecção & Tracking de Pessoas
- **YOLOv8 (Ultralytics)** para detectar pessoas
- **ByteTrack** para manter um ID consistente ao longo do vídeo

### 2) Emoções Faciais (por pessoa)
- **DeepFace** para inferir emoções (happy, sad, angry, fear, surprise, neutral, etc.)
- Associação emoção ↔ pessoa via proximidade box pessoa / face

### 3) Atividades (por pessoa e no geral)
Abordagem híbrida (mais robusta que “um modelo só”):
- **CLIP Zero-Shot (OpenCLIP)** com prompts em inglês (mais “humanos”) e **labels final em português**
- **Action Recognition (R3D-18 / Kinetics400)** como *fallback* quando o CLIP não está confiante
- Heurísticas simples para atividades “contextuais”, ex:
  - **reunião / conversa** (pessoas próximas, postura, baixa movimentação)
  - **usando computador / digitando** (pessoa sentada + mãos perto da região de teclado/mesa + objetos próximos)

### 4) Anomalias
- Anomalia = movimento fora do padrão geral do vídeo (gestos bruscos, mudanças abruptas etc.)
- Implementação: **z-score** do deslocamento/variação de posição ao longo do tempo

### 5) Suavização temporal (anti “alucinação”)
- Votação/janela temporal para reduzir troca de labels frame a frame
- “cooldown” mínimo antes de mudar a atividade dominante

---


## 📚 Bibliotecas Principais

- `torch` - Backend de deep learning utilizado para executar modelos de IA em CPU ou GPU (CUDA), incluindo Action Recognition e CLIP
- `ultralytics` - Implementação do YOLOv8 para detecção e tracking de pessoas em vídeos
- `open-clip-torch` - Implementação do CLIP Zero-Shot, utilizada para classificação semântica de atividades em linguagem natural
- `deepface` - Biblioteca para análise de expressões emocionais faciais, baseada em modelos pré-treinados
- `opencv-python` - – Processamento de vídeo, leitura de frames, escrita de vídeo anotado e operações de imagem
- `numpy` - – Operações numéricas, manipulação de arrays e cálculos estatísticos (ex: detecção de anomalias)
- `tqdm` – Exibição de barras de progresso durante o processamento do vídeo
- `mediapipe` - Extração de landmarks corporais e faciais, auxiliando na análise de postura e movimentos
- `protobuf` - Serialização de dados utilizada internamente pelo MediaPipe e TensorFlow
- `keras` - API de alto nível para construção e execução de modelos neurais utilizados pelo DeepFace
- `gast` - Dependência do ecossistema TensorFlow para análise e transformação de grafos computacionais
- `tensorboard` - Ferramenta de visualização e monitoramento utilizada pelo TensorFlow
- `pillow` - Manipulação e conversão de imagens, suporte auxiliar ao OpenCV e CLIP

---

## 👨‍💻 Autor

**Phellype Guilherme Pereira da Silva**  
**RM:** 361625  
**Projeto:** Fase 4 - Pós Tech FIAP - Inteligência Artificial  
**Instituição:** [FIAP – Faculdade de Informática e Administração Paulista](https://www.fiap.com.br/)
